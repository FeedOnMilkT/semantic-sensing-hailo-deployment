"""Merge Hailo SDK log files into one timestamped, provenance-preserving file."""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

SDK_LOG_PATTERNS = (
    "command.log",
    "hailo_sdk*.log",
    "acceleras.log",
    "allocator.log",
    "hailort.log",
    "pyhailort.log",
)
DEFAULT_LOG_ROOT = Path(
    "/local/shared_with_docker/deplyment-demo/artifacts/sdk_logs"
)


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M")


def collect(source_dir: Path) -> list[Path]:
    return sorted({path for pattern in SDK_LOG_PATTERNS for path in source_dir.glob(pattern)})


def unique_path(directory: Path, stem: str) -> Path:
    candidate = directory / f"{stem}.log"
    duplicate = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{duplicate:02d}.log"
        duplicate += 1
    return candidate


def consolidate(
    source_dir: Path,
    output_dir: Path,
    stem: str,
    archive_source: bool = False,
) -> Path:
    """Write one text bundle, optionally moving its raw source files to legacy/."""
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    sources = collect(source_dir)
    if not sources:
        raise FileNotFoundError(f"no Hailo SDK logs found under {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = unique_path(output_dir, stem)
    with destination.open("w", encoding="utf-8", buffering=1) as output:
        output.write("# Consolidated Hailo SDK logs\n")
        output.write(
            f"# generated_at: {datetime.now().astimezone().isoformat(timespec='minutes')}\n"
        )
        output.write(f"# source_dir: {source_dir}\n")
        output.write(f"# source_count: {len(sources)}\n")
        for source in sources:
            stat = source.stat()
            output.write("\n" + "=" * 88 + "\n")
            output.write(f"# SOURCE: {source.name}\n")
            output.write(
                "# mtime: "
                f"{datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec='seconds')}\n"
            )
            output.write(f"# bytes: {stat.st_size}\n")
            output.write("=" * 88 + "\n")
            content = source.read_text(encoding="utf-8", errors="replace")
            output.write(content)
            if not content.endswith("\n"):
                output.write("\n")

    if archive_source:
        archive_dir = output_dir / "legacy" / destination.stem
        archive_dir.mkdir(parents=True, exist_ok=False)
        for source in sources:
            shutil.move(str(source), archive_dir / source.name)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consolidate Hailo SDK logs and optionally clear the source directory."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/workspace/demo/deplyment-demo"),
        help="directory currently containing hailo_sdk*.log (default: deployment root)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--stem",
        default=None,
        help="output basename without .log (default: sdk_history_<minute>)",
    )
    parser.add_argument(
        "--archive-source",
        action="store_true",
        help="move raw source logs under output-dir/legacy after bundling",
    )
    args = parser.parse_args()

    destination = consolidate(
        args.source_dir,
        args.output_dir,
        args.stem or f"sdk_history_{timestamp()}",
        archive_source=args.archive_source,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
