/*
 * sha256.h -- SHA-256 helper for the manifest.
 */
#ifndef SSCODEC_SENDER_SHA256_H
#define SSCODEC_SENDER_SHA256_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace sender {

/* Lower-case hex SHA-256 digest (64 characters) of a byte buffer. */
std::string Sha256Hex(const uint8_t *data, size_t len);

}  // namespace sender

#endif /* SSCODEC_SENDER_SHA256_H */
