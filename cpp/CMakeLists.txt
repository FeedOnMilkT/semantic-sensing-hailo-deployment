cmake_minimum_required(VERSION 3.16)
project(sscodec LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(WITH_HAILORT   "Build the HailoRT sender backend"          OFF)
option(HAILORT_LINK   "LINK the HailoRT backend against libhailort" OFF)
option(BUILD_TESTS    "Build the test suite"                       ON)
option(BUILD_TOOLS    "Build the CLI tools"                        ON)
option(BUILD_SENDER   "Build the sender pipeline"                  ON)
option(STRICT_FP      "Deterministic FP flags (ON is strongly recommended)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Deterministic floating point.
#
# The h_s evaluation MUST be bit-identical between the sender build (possibly
# aarch64 / Raspberry Pi) and the receiver build (x86-64).  The flags below
# are the whole reason:
#
#   -ffp-contract=off   GCC on aarch64 contracts a*b+c into an FMA BY
#                        DEFAULT.  With contraction on, the aarch64 sender and
#                        the x86-64 receiver would produce different h_s
#                        outputs, and ~17% of the CDF indexes would differ ->
#                        rANS desync.  This flag is MANDATORY.
#   -fno-fast-math       no reassociation, no reciprocal substitution, no
#                        finite-math-only assumptions.
#   -fno-unsafe-math-optimizations
#   -fno-associative-math
#
# Note: -ffast-math is never used.  -O3 alone does NOT break reproducibility
# here because the h_s loops are hand-written with a fixed accumulation
# order (no vectorisation-dependent reductions) -- but -ffp-contract=off is
# still required because FMA contraction changes the ROUNDING of the
# accumulation itself.
# ---------------------------------------------------------------------------
set(SSCODEC_FP_FLAGS "")
if(STRICT_FP)
  list(APPEND SSCODEC_FP_FLAGS
       -ffp-contract=off
       -fno-fast-math
       -fno-unsafe-math-optimizations
       -fno-associative-math)
endif()
message(STATUS "sscodec deterministic FP flags: ${SSCODEC_FP_FLAGS}")

# ---------------------------------------------------------------------------
# libsscodec
# ---------------------------------------------------------------------------
set(SSCODEC_SOURCES
    src/rans.cpp
    src/hyper_synthesis.cpp
    src/codec.cpp
    src/sscodec_api.cpp)

add_library(sscodec_obj OBJECT ${SSCODEC_SOURCES})
target_include_directories(sscodec_obj PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/compressai_rans)
target_compile_options(sscodec_obj PRIVATE ${SSCODEC_FP_FLAGS} -Wall -Wextra)

add_library(sscodec SHARED $<TARGET_OBJECTS:sscodec_obj>)
target_include_directories(sscodec PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/compressai_rans)
target_link_libraries(sscodec PRIVATE sscodec_obj)
set_target_properties(sscodec PROPERTIES OUTPUT_NAME sscodec)

add_library(sscodec_static STATIC $<TARGET_OBJECTS:sscodec_obj>)
target_include_directories(sscodec_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/compressai_rans)
target_link_libraries(sscodec_static PRIVATE sscodec_obj)
set_target_properties(sscodec_static PROPERTIES OUTPUT_NAME sscodec)

# ---------------------------------------------------------------------------
# CLI tools
# ---------------------------------------------------------------------------
if(BUILD_TOOLS)
  add_executable(ss_encode_cli tools/ss_encode_cli.cpp)
  target_link_libraries(ss_encode_cli PRIVATE sscodec_static)
  target_compile_options(ss_encode_cli PRIVATE ${SSCODEC_FP_FLAGS})

  add_executable(ss_decode_cli tools/ss_decode_cli.cpp)
  target_link_libraries(ss_decode_cli PRIVATE sscodec_static)
  target_compile_options(ss_decode_cli PRIVATE ${SSCODEC_FP_FLAGS})
endif()

# ---------------------------------------------------------------------------
# Sender pipeline
# ---------------------------------------------------------------------------
if(BUILD_SENDER)
  set(SENDER_CORE_SOURCES
      sender/replay_backend.cpp
      sender/sender_pipeline.cpp
      sender/sha256.cpp)

  if(NOT WITH_HAILORT)
    # The stub factory only exists when the real HailoRT backend is absent.
    # If it were compiled in unconditionally, the static linker would resolve
    # `MakeHailoRTBackend` from sender_core FIRST (it comes earlier on the
    # link line) and the real backend in hailtort_backend.a would never be
    # pulled in -- the binary would silently keep the stub.
    list(APPEND SENDER_CORE_SOURCES sender/inference_backend.cpp)
  endif()

  add_library(sender_core STATIC ${SENDER_CORE_SOURCES})
  target_include_directories(sender_core PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/sender
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_options(sender_core PRIVATE ${SSCODEC_FP_FLAGS})
  target_link_libraries(sender_core PUBLIC sscodec_static)

  if(WITH_HAILORT)
    # The HailoRT headers are only used for a COMPILE CHECK by default: this
    # machine has no Hailo device and no libhailort, so the backend cannot be
    # linked or run.  Point HAILORT_INCLUDE_DIR at the hailoRT checkout.
    #
    # With -DHAILORT_LINK=ON the backend is also LINKED against libhailort,
    # which is what the Raspberry Pi / board build needs.
    if(NOT HAILORT_INCLUDE_DIR)
      message(FATAL_ERROR
        "WITH_HAILORT=ON requires -DHAILORT_INCLUDE_DIR=<hailort checkout>/hailort")
    endif()
    add_library(hailort_backend STATIC sender/hailort_backend.cpp)
    target_include_directories(hailort_backend PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/sender
        ${HAILORT_INCLUDE_DIR})
    target_compile_definitions(hailort_backend PRIVATE SSCODEC_WITH_HAILORT=1)
    target_compile_options(hailort_backend PRIVATE ${SSCODEC_FP_FLAGS})
    target_link_libraries(hailort_backend PUBLIC sender_core)
    set(SSCODEC_SENDER_BACKEND hailort_backend)
    if(HAILORT_LINK)
      find_library(HAILORT_LIB NAMES hailort
                   HINTS ${HAILORT_INCLUDE_DIR}/../lib
                         ${HAILORT_ROOT}/lib)
      if(NOT HAILORT_LIB)
        message(FATAL_ERROR
          "HAILORT_LINK=ON but libhailort was not found -- point "
          "-DHAILORT_ROOT=<hailort install prefix> at the install tree")
      endif()
      target_link_libraries(hailort_backend PUBLIC ${HAILORT_LIB})
      message(STATUS "sscodec: linking the HailoRT backend against ${HAILORT_LIB}")
    else()
      message(STATUS
        "sscodec: WITH_HAILORT=ON -- COMPILE CHECK ONLY. The HailoRT backend "
        "is compiled but NOT linked (no libhailort available); configure with "
        "-DHAILORT_LINK=ON on a machine that has it.")
    endif()
  else()
    set(SSCODEC_SENDER_BACKEND "")
  endif()

  add_executable(sender_cpp sender/sender_main.cpp)
  target_link_libraries(sender_cpp PRIVATE sender_core ${SSCODEC_SENDER_BACKEND})
  target_compile_options(sender_cpp PRIVATE ${SSCODEC_FP_FLAGS})
  if(WITH_HAILORT AND NOT HAILORT_LINK)
    # The compile-check build cannot link the real backend (no libhailort),
    # so the sender binary keeps the stub factory and reports a clear error
    # if the hailo backend is requested.  This keeps the *build* green while
    # still compiling the real HailoRT code path.
    target_sources(sender_cpp PRIVATE sender/inference_backend.cpp)
    target_compile_definitions(sender_cpp PRIVATE SSCODEC_HAILORT_COMPILE_CHECK=1)
  endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
  enable_testing()

  # The C++ tests need the shared test vectors and the codec tables.  Both live
  # outside the repo (the tables are a build artefact, the vectors are derived
  # from the Hailo emulator dump), so they are passed in explicitly:
  #
  #   -DSSCODEC_TABLES=<path to codec_tables.bin>
  #   -DSSCODEC_VECTORS=<path to vectors250.bin>
  #   -DSSCODEC_N_VECTORS=<number of images in the vector file>
  #
  # If they are not given, the tests are still built but `ctest` reports them
  # as SKIPPED (a missing vector file is not a build failure).
  set(SSCODEC_TABLES "" CACHE FILEPATH "codec_tables.bin for the C++ tests")
  set(SSCODEC_VECTORS "" CACHE FILEPATH "test vectors for the C++ tests")
  set(SSCODEC_N_VECTORS 250 CACHE STRING "number of images in the vector file")

  add_executable(ss_test_roundtrip tests/test_roundtrip.cpp)
  target_link_libraries(ss_test_roundtrip PRIVATE sscodec_static)
  target_compile_options(ss_test_roundtrip PRIVATE ${SSCODEC_FP_FLAGS})

  add_executable(ss_test_determinism tests/test_determinism.cpp)
  target_link_libraries(ss_test_determinism PRIVATE sscodec_static)
  target_compile_options(ss_test_determinism PRIVATE ${SSCODEC_FP_FLAGS})

  if(SSCODEC_TABLES AND SSCODEC_VECTORS AND EXISTS "${SSCODEC_TABLES}"
     AND EXISTS "${SSCODEC_VECTORS}")
    # `ctest` runs the test binary from the build dir; the arguments are
    # absolute paths so that works from anywhere.
    add_test(NAME roundtrip
             COMMAND ss_test_roundtrip "${SSCODEC_TABLES}" "${SSCODEC_VECTORS}"
                     "${SSCODEC_N_VECTORS}")
    add_test(NAME determinism
             COMMAND ss_test_determinism --dump
                     --out "${CMAKE_CURRENT_BINARY_DIR}/ctest_dump.bin"
                     --tables "${SSCODEC_TABLES}"
                     --vectors "${SSCODEC_VECTORS}"
                     --n "${SSCODEC_N_VECTORS}")
    set_tests_properties(roundtrip PROPERTIES TIMEOUT 1800)
    set_tests_properties(determinism PROPERTIES TIMEOUT 1800)
  else()
    message(STATUS
        "sscodec: SSCODEC_TABLES / SSCODEC_VECTORS not set -- "
        "the C++ tests are built but cannot be run by ctest automatically")
  endif()
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS sscodec sscodec_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(FILES include/sscodec.h DESTINATION include)
