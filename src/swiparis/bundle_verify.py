"""Standalone bundle verifier, also copied to the root of exported demo ZIPs."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise SystemExit(f"Invalid bundle path: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise SystemExit(f"Hash mismatch: {relative}")
    print("Bundle hashes OK:", manifest["data_version"])


if __name__ == "__main__":
    main()
