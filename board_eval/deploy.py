#!/usr/bin/env python3
"""Deploy one immutable HEF + memory-only runner to the Hailo board over SCP."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hailo_deploy.io.hash import sha256


def common_options(args) -> list[str]:
    # A board powered from a marginal adapter may retain a Tailnet presence
    # while no longer forwarding an SSH data stream.  Bound that failure so a
    # deployment cannot leave a half-uploaded HEF or block the automation.
    options = [
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
    ]
    if args.identity_file:
        options += ["-i", str(args.identity_file), "-o", "IdentitiesOnly=yes"]
    if args.proxy_command:
        options += ["-o", f"ProxyCommand={args.proxy_command}"]
    if args.known_hosts:
        options += ["-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={args.known_hosts}"]
    return options


def run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def stream_upload(source: Path, destination: str, options: list[str], target: str) -> None:
    """Upload through SSH with resume support and an atomic final rename.

    Some marginal Tailscale/board links complete ordinary SSH commands but hang
    in the legacy SCP sub-protocol.  This keeps the transport encrypted and
    authenticated while avoiding SCP. Interrupted byte streams retain only a
    ``.uploading`` file; the next identical upload resumes from its exact byte
    count, and ``mv`` ensures no partial HEF is selected by a runner.
    """
    remote_dir, filename = destination.rsplit("/", 1)
    staged = f"{remote_dir}/.{filename}.uploading"
    source_digest = sha256(source)
    existing = subprocess.run(
        ["ssh", *options, target,
         f"if test -f {shlex.quote(destination)}; then sha256sum {shlex.quote(destination)} | cut -d' ' -f1; fi"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()
    if existing == source_digest:
        return
    probe = subprocess.run(
        ["ssh", *options, target,
         f"mkdir -p {shlex.quote(remote_dir)}; "
         f"if test -f {shlex.quote(staged)}; then stat -c%s {shlex.quote(staged)}; else echo 0; fi"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        offset = int(probe.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"invalid remote staging size for {destination!r}: {probe.stdout!r}") from exc
    source_bytes = source.stat().st_size
    if offset < 0 or offset > source_bytes:
        raise RuntimeError(f"invalid remote staging size {offset} for {source} ({source_bytes} bytes)")
    if offset == source_bytes:
        command = f"mv -f {shlex.quote(staged)} {shlex.quote(destination)}"
        completed = subprocess.run(["ssh", *options, target, command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        operator = ">" if offset == 0 else ">>"
        command = f"cat {operator} {shlex.quote(staged)} && mv -f {shlex.quote(staged)} {shlex.quote(destination)}"
        with source.open("rb") as handle:
            handle.seek(offset)
            completed = subprocess.run(
                ["ssh", *options, target, command], stdin=handle,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace")
        raise subprocess.CalledProcessError(completed.returncode, command, stderr=message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--remote-root", default=".local/share/semantic-sensing-board")
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--proxy-command")
    parser.add_argument("--known-hosts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    hef = Path(manifest["candidate"]["hef"]["path"])
    runner = Path(__file__).with_name("board_runner.py")
    if sha256(hef) != manifest["candidate"]["hef"]["sha256"]:
        raise ValueError("local HEF hash differs from frozen manifest")
    remote_dir = f"{args.remote_root.rstrip('/')}/{manifest['run_tag']}"
    opts = common_options(args)
    run(["ssh", *opts, args.ssh_target, f"mkdir -p {shlex.quote(remote_dir)}"])
    remote_manifest = dict(manifest)
    remote_manifest["board"] = dict(manifest["board"])
    remote_manifest["board"]["hef_filename"] = hef.name
    with tempfile.TemporaryDirectory(dir=args.manifest.parent) as directory:
        temp_manifest = Path(directory) / "manifest.json"
        temp_manifest.write_text(json.dumps(remote_manifest, indent=2, sort_keys=True), encoding="utf-8")
        for source, name in ((hef, hef.name), (runner, "board_runner.py"), (temp_manifest, "manifest.json")):
            stream_upload(source, f"{remote_dir}/{name}", opts, args.ssh_target)
    remote_sha = run(["ssh", *opts, args.ssh_target,
                      f"sha256sum {shlex.quote(remote_dir + '/' + hef.name)}"]).stdout.split()[0]
    if remote_sha != sha256(hef):
        raise ValueError(f"board HEF hash mismatch: {remote_sha}")
    preflight = run(["ssh", *opts, args.ssh_target,
                     "python3 -c 'import hailo_platform, numpy; print(hailo_platform.__file__)'; "
                     f"hailortcli parse-hef {shlex.quote(remote_dir + '/' + hef.name)}"])
    report = {
        "run_tag": manifest["run_tag"], "remote_dir": remote_dir,
        "hef": {"sha256": remote_sha, "bytes": hef.stat().st_size},
        "ssh_target": args.ssh_target, "preflight_stdout": preflight.stdout,
        "preflight_stderr": preflight.stderr,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
