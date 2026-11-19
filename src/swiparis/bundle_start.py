"""Standalone offline launcher copied as start.py into the demo bundle."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10+ is required")
    root = Path(__file__).resolve().parent
    started = time.perf_counter()
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "PIP_PREFIX"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run([sys.executable, str(root / "verify_bundle.py")], check=True, cwd=root, env=env, timeout=30)
    venv = root / ".venv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    fresh = not python.is_file()
    if fresh:
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, cwd=root, env=env, timeout=180)
    wheels = list((root / "wheels").glob("swiparis-*.whl"))
    if len(wheels) != 1:
        raise SystemExit("Expected exactly one SwiParis wheel")
    subprocess.run([str(python), "-m", "pip", "--isolated", "install", "--no-index", "--no-deps", "--force-reinstall", str(wheels[0])], check=True, cwd=root, env=env, timeout=120)
    imported = subprocess.check_output([str(python), "-c", "import swiparis; print(swiparis.__file__)"], cwd=root, env=env, text=True).strip()
    if not Path(imported).resolve().is_relative_to(venv):
        raise SystemExit("Package imported outside the bundle venv")
    smoke = subprocess.check_output([str(python), "-m", "swiparis.smoke", "--catalog", str(root / "data/processed/common.sqlite")], cwd=root, env=env, timeout=120, text=True, encoding="utf-8")
    elapsed = round(time.perf_counter() - started, 3)
    report = dict(status="ok" if elapsed < 600 else "too_slow", total_seconds=elapsed,
                  under_ten_minutes=elapsed < 600, fresh_venv=fresh, imported_package=imported,
                  network_packages_downloaded=False, separate_physical_machine_tested=False, smoke=json.loads(smoke))
    (root / "bundle_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if elapsed >= 600:
        raise SystemExit("Preparation exceeded ten minutes")
    if not args.check_only:
        try:
            subprocess.run([str(python), "-m", "swiparis.demo", "--catalog", str(root / "data/processed/common.sqlite"), "--logs", str(root / "data/pilot/interactions.sqlite")], check=True, cwd=root, env=env)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
