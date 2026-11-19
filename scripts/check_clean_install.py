"""Create a fresh isolated venv, install the wheel offline and time real HTTP checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/common.sqlite"))
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/clean-install"))
    parser.add_argument("--report", type=Path, default=Path("outputs/clean_install_report.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    work = args.work_dir.resolve()
    if work.exists():
        raise SystemExit("Work directory exists; choose a new --work-dir")
    work.mkdir(parents=True)
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "PIP_PREFIX"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run([sys.executable, "-m", "venv", str(work / "venv")], check=True, env=env, timeout=120)
    python = work / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "--isolated", "install", "--no-index", "--no-deps", str(args.wheel.resolve())], check=True, env=env, cwd=work, timeout=120)
    imported = subprocess.check_output([str(python), "-c", "import swiparis; print(swiparis.__file__)"], env=env, cwd=work, text=True).strip()
    if not Path(imported).resolve().is_relative_to(work / "venv"):
        raise RuntimeError("Smoke test imported code outside the clean venv")
    smoke = subprocess.check_output([str(python), "-m", "swiparis.smoke", "--catalog", str(args.catalog.resolve())], env=env, cwd=work, timeout=120, text=True, encoding="utf-8")
    elapsed = round(time.perf_counter() - started, 3)
    report = dict(status="ok" if elapsed < 600 else "too_slow", total_seconds=elapsed, under_ten_minutes=elapsed < 600,
                  python=sys.version, clean_venv=True, imported_package=imported,
                  network_packages_downloaded=False, separate_physical_machine_tested=False,
                  note="Fresh venv on current Windows host, not a separate computer; Python and prepared catalog are prerequisites.",
                  smoke=json.loads(smoke))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if elapsed >= 600:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
