"""Export only the demo catalog and wheel; never raw data, logs, environment or keys."""
import argparse
from importlib.resources import files
import json
from pathlib import Path
import zipfile

from .catalog import Catalog
from .prepare_data import sha256


def build(catalog_path, wheel_path, output):
    catalog_path, wheel_path, output = map(Path, (catalog_path, wheel_path, output))
    if output.exists():
        raise ValueError("Bundle already exists; choose another output")
    if not wheel_path.is_file() or not wheel_path.name.startswith("swiparis-") or wheel_path.suffix != ".whl":
        raise ValueError("Expected a built SwiParis wheel")
    catalog = Catalog(catalog_path)
    entries = {"data/processed/common.sqlite": catalog_path.read_bytes(),
               "wheels/" + wheel_path.name: wheel_path.read_bytes(),
               "start.py": files("swiparis").joinpath("bundle_start.py").read_bytes(),
               "verify_bundle.py": files("swiparis").joinpath("bundle_verify.py").read_bytes(),
               "README.txt": ("SwiParis offline demo. Prerequisite: Python 3.10+ installed.\n"
                              "Extract ZIP, open PowerShell here, run: py -3 start.py\n"
                              "Check without leaving a server running: py -3 start.py --check-only\n"
                              "Custom interpreter: & 'C:\\path\\python.exe' start.py\n"
                              "Map optional: set $env:GOOGLE_MAPS_API_KEY before starting.\n"
                              "Open http://127.0.0.1:8000 . Ctrl+C stops the local server.\n"
                              "Snapshot dates appear on screen. No trained model included.\n"
                              "Data sources: Foursquare (Yang et al.), City of Paris / Paris Data.\n"
                              "Internal student demo: retain original source terms; see project docs.\n"
                              "No raw files, pilot logs, API keys or external runtime packages included.\n").encode()}
    import hashlib
    manifest = dict(data_version=catalog.version,
                    files_sha256={name: hashlib.sha256(data).hexdigest() for name, data in entries.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    return dict(bundle=str(output), bytes=output.stat().st_size, sha256=sha256(output), data_version=catalog.version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/common.sqlite"))
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("dist/swiparis-demo.zip"))
    args = parser.parse_args()
    print(json.dumps(build(args.catalog, args.wheel, args.output), indent=2))


if __name__ == "__main__":
    main()
