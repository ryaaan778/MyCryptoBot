#!/usr/bin/env python3
"""JOJO Trading Command Center — launcher.

Boots the trading engine and serves the API, the WebSocket and (when it has
been built) the voxel world from a single process.

    python run_bot.py                        # paper trading, live market data
    python run_bot.py --provider simulated   # paper trading, synthetic feed
    python run_bot.py --mode live            # requires the live-trading gate

The live-trading gate needs all three of:
  1. ``"live_enabled": true`` under ``trading`` in config.json
  2. an API key and secret under ``exchange``
  3. ``LIVE_TRADING_CONFIRM=I_UNDERSTAND_THE_RISK`` in the environment
Anything less runs the paper matching engine instead.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JOJO Trading Command Center",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["paper", "live"], default="paper",
        help="paper runs the simulated matching engine (default); live requires the gate",
    )
    parser.add_argument(
        "--provider", choices=["auto", "binance", "simulated"], default="auto",
        help="market data source; auto tries Binance and falls back to synthetic",
    )
    parser.add_argument("--config", default="config.json", help="configuration file path")
    parser.add_argument("--host", default=None, help="bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="port (default 8000)")
    parser.add_argument("--data-dir", default=None, help="directory for the SQLite database")
    parser.add_argument("--seed", type=int, default=None, help="seed for the synthetic feed")
    parser.add_argument(
        "--balance", type=float, default=None, help="starting paper balance",
    )
    parser.add_argument(
        "--log-level", default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    try:
        import uvicorn
    except ImportError:
        print(
            "\n❌ Dependencies are missing. Install them first:\n"
            "     pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        return 1

    from backend.app import WEB_DIST, create_app
    from backend.config import load_settings

    overrides: dict[str, object] = {"provider": args.provider}
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.balance is not None:
        overrides["initial_balance"] = args.balance
    if args.host:
        overrides["server.host"] = args.host
    if args.port:
        overrides["server.port"] = args.port
    if args.mode == "paper":
        # An explicit --mode paper must win over whatever config.json says.
        overrides["trading.live_enabled"] = False

    settings = load_settings(REPO_ROOT / args.config, overrides=overrides)

    allowed, reason = settings.live_gate_status()
    if args.mode == "live" and not allowed:
        print(f"\n⚠️  Live trading requested but the gate is closed: {reason}")
        print("   Continuing in PAPER mode. No real orders will be placed.\n")
    elif allowed:
        print("\n" + "=" * 72)
        print("  ⚠️  LIVE TRADING IS ENABLED — REAL ORDERS WILL BE PLACED")
        print(f"  exchange: {settings.exchange.name}  testnet: {settings.exchange.testnet}")
        print("=" * 72 + "\n")

    host = settings.server.host
    port = settings.server.port

    print("\n🌸 JOJO TRADING COMMAND CENTER")
    print(f"   execution : {'LIVE' if allowed else 'PAPER'}")
    print(f"   market    : {settings.provider}")
    print(f"   bots      : {', '.join(b.name for b in settings.bots)}")
    print(f"   API       : http://{host}:{port}/api/health")
    print(f"   WebSocket : ws://{host}:{port}/ws")
    if WEB_DIST.exists():
        print(f"   World     : http://{host}:{port}/")
    else:
        print("   World     : not built yet — run `cd web && npm install && npm run build`")
        print("               (or `npm run dev` for the Vite dev server on :5173)")
    print()

    os.environ.setdefault("JOJO_PROVIDER", settings.provider)
    uvicorn.run(create_app(settings), host=host, port=port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
