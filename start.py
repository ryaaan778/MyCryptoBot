#!/usr/bin/env python3
"""One command to get the desk running on a fresh clone.

    python start.py                     # paper trading, synthetic feed
    python start.py --provider binance  # paper trading, live market data
    python start.py --port 8080

Checks the Python dependencies, builds the voxel world if it has not been built,
then hands off to ``run_bot.py``. Everything it does, it says first — a script
that silently installs things on your machine is not a convenience.

If you would rather do it by hand, the three steps are exactly:

    pip install -r requirements.txt
    cd web && npm install && npm run build && cd ..
    python run_bot.py --provider simulated
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
DIST = WEB / "dist"

#: Import name -> what to tell the user if it is missing.
REQUIRED = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "numpy": "numpy",
    "aiosqlite": "aiosqlite",
    "websockets": "websockets",
}


def say(message: str) -> None:
    print(f"  {message}", flush=True)


def missing_python_packages() -> list[str]:
    import importlib.util

    return [name for name in REQUIRED if importlib.util.find_spec(name) is None]


def install_python_dependencies() -> bool:
    say("installing Python dependencies (pip install -r requirements.txt)")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
        cwd=ROOT,
    )
    return result.returncode == 0


def npm_command() -> str | None:
    """npm is a shell script on Windows, so ``shutil.which`` needs the real name."""
    for candidate in ("npm", "npm.cmd"):
        if shutil.which(candidate):
            return candidate
    return None


def build_frontend(npm: str) -> bool:
    if not (WEB / "node_modules").exists():
        say("installing frontend packages (npm install) — this takes a minute the first time")
        if subprocess.run([npm, "install"], cwd=WEB).returncode != 0:
            return False
    say("building the voxel world (npm run build)")
    return subprocess.run([npm, "run", "build"], cwd=WEB).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set up if needed, then start the trading desk.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", default="simulated",
                        choices=["auto", "binance", "simulated"],
                        help="simulated is the fastest way to see everything move")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--skip-setup", action="store_true",
                        help="assume dependencies and the build are already in place")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the frontend even if it already exists")
    args = parser.parse_args()

    print("\nJOJO Trading Command Center — setup check\n")

    if not args.skip_setup:
        gaps = missing_python_packages()
        if gaps:
            say(f"missing Python packages: {', '.join(sorted(gaps))}")
            if not install_python_dependencies():
                say("pip failed. Install by hand: pip install -r requirements.txt")
                return 1
        else:
            say("Python dependencies: ok")

        if args.rebuild or not DIST.exists():
            npm = npm_command()
            if npm is None:
                # Not fatal. The API and the 2D fallback both work without it,
                # and saying so beats refusing to start.
                say("npm not found — skipping the frontend build.")
                say("The API will run, but the world needs Node.js: https://nodejs.org")
            elif not build_frontend(npm):
                say("the frontend build failed; starting the API anyway.")
                say("Build it by hand later: cd web && npm install && npm run build")
        else:
            say("voxel world: already built (--rebuild to force)")

    print()
    command = [
        sys.executable, str(ROOT / "run_bot.py"),
        "--provider", args.provider, "--port", str(args.port), "--host", args.host,
    ]
    say(f"starting: {' '.join(command[1:])}")
    say(f"open http://localhost:{args.port} once it says the engine is up")
    print()
    try:
        return subprocess.run(command, cwd=ROOT).returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
