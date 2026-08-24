#!/usr/bin/env python3
"""Research CLI — datasets, splits, baselines, reports.

Everything here is offline. No command in this file starts a server, places an
order, or touches ``config.json``; the backtester forces ``live_enabled=False``
regardless of what the settings say.

    python research.py venues                     # what ccxt can reach
    python research.py synth --bars 60000         # deterministic simulated bars
    python research.py collect --venue bybit --symbol BTC/USDT --from 2023-01-01
    python research.py datasets                   # what is on disk and where it came from
    python research.py features --dataset <id>
    python research.py regimes  --dataset <id>
    python research.py split    --dataset <id>
    python research.py backtest --dataset <id> --policy mean_reversion
    python research.py baselines --dataset <id>   # every policy, walk-forward
    python research.py report   --dataset <id>

Training, promotion and the RL agents arrive in later phases; those subcommands
are deliberately absent rather than stubbed, so ``--help`` describes what the
system can actually do today.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

from backend.config import load_settings
from research.backtest import BacktestConfig, run_backtest, run_walk_forward
from research.collect import collect_exchange, generate_synthetic, parse_date
from research.datasets import (
    Dataset, DataSource, known_exchanges, list_datasets, assert_single_source,
)
from research.experiments import ExperimentStore
from research.features import FEATURE_SET_VERSION, available_features, build_features
from research.metrics import aggregate, summarise
from research.policy import BASELINE_POLICIES, STRATEGY_POLICIES, make_policy
from research.regimes import Regime, label_regimes
from research.splits import (
    SplitConfig, SplitPlan, default_config, make_walk_forward, scale_config,
)

DEFAULT_ROOT = Path("data")
NO_EXIT_POLICIES = {"buy_and_hold", "always_short", "flat"}


def log_setup(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def load(root: Path, dataset_id: str) -> Dataset:
    try:
        return Dataset.load(root, dataset_id)
    except FileNotFoundError:
        available = [m.dataset_id for m in list_datasets(root)]
        raise SystemExit(
            f"no dataset {dataset_id!r} under {root}."
            + (f" Available: {', '.join(available)}" if available
               else " Run `python research.py synth` first.")
        )


def resolve_dataset(root: Path, dataset_id: str | None) -> Dataset:
    if dataset_id:
        return load(root, dataset_id)
    manifests = list_datasets(root)
    if not manifests:
        raise SystemExit(f"no datasets under {root}. Run `python research.py synth` first.")
    if len(manifests) > 1:
        raise SystemExit(
            "several datasets are available; name one with --dataset:\n  "
            + "\n  ".join(m.dataset_id for m in manifests)
        )
    return load(root, manifests[0].dataset_id)


def config_for(args, policy_name: str | None = None) -> BacktestConfig:
    settings = load_settings()
    stop = args.stop_loss if args.stop_loss is not None else settings.risk.stop_loss_pct
    target = args.take_profit if args.take_profit is not None else settings.risk.take_profit_pct
    # A hold-forever baseline with a 1% stop is not a hold-forever baseline; it
    # gets stopped out and re-entered dozens of times and stops being the thing
    # a learned policy is supposed to have to beat.
    if policy_name in NO_EXIT_POLICIES:
        stop = target = None
    return BacktestConfig(
        starting_equity=args.equity, risk_per_trade=args.risk, leverage=args.leverage,
        stop_loss_pct=stop, take_profit_pct=target,
        spread_bps=args.spread, fee_rate=args.fee,
    )


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_venues(args) -> int:
    exchanges = sorted(known_exchanges())
    if not exchanges:
        print("ccxt is not installed; `pip install ccxt` to collect real market data.")
        return 1
    print(f"{len(exchanges)} venues reachable through ccxt:\n")
    for i in range(0, len(exchanges), 6):
        print("  " + "".join(f"{name:<18}" for name in exchanges[i:i + 6]))
    print("\nAny of these is a valid --venue. Data from different venues is kept")
    print("separate: fees, spreads and leverage limits differ, so a figure averaged")
    print("across venues describes no venue that actually exists.")
    return 0


def cmd_synth(args) -> int:
    dataset = generate_synthetic(
        symbol=args.symbol, timeframe=args.timeframe, bars=args.bars, seed=args.seed
    )
    directory = dataset.save(args.root)
    with ExperimentStore(Path(args.root) / "research.db") as store:
        store.register_dataset(dataset.manifest)
    print(dataset.manifest.describe())
    print(f"  -> {directory}")
    print("\nSYNTHETIC data tests the harness. It is not evidence about live markets,")
    print("and the reporting layer will refuse to average it with real-venue results.")
    return 0


def cmd_collect(args) -> int:
    since = parse_date(args.since) if args.since else None
    until = parse_date(args.until) if args.until else None
    try:
        dataset = asyncio.run(collect_exchange(
            args.venue, args.symbol, args.timeframe, since=since, until=until,
            testnet=args.testnet, market_type=args.market_type,
            cache_dir=args.root, max_pages=args.max_pages,
        ))
    except Exception as exc:                       # network, auth, unknown venue
        print(f"collection failed: {exc}", file=sys.stderr)
        return 1
    directory = dataset.save(args.root)
    with ExperimentStore(Path(args.root) / "research.db") as store:
        store.register_dataset(dataset.manifest)
    print(dataset.manifest.describe())
    print(f"  -> {directory}")
    if dataset.manifest.gaps:
        print(f"  NOTE: {dataset.manifest.gaps} missing bars. An indicator computed across "
              "a gap blends two non-adjacent points in time.")
    return 0


def cmd_datasets(args) -> int:
    manifests = list_datasets(args.root)
    if not manifests:
        print(f"no datasets under {args.root}")
        return 0
    for manifest in manifests:
        print(manifest.describe())
    sources = sorted({m.source for m in manifests})
    print(f"\n{len(manifests)} dataset(s) from {len(sources)} source(s): {', '.join(sources)}")
    return 0


def cmd_features(args) -> int:
    dataset = resolve_dataset(Path(args.root), args.dataset)
    matrix = build_features(dataset)
    print(f"{dataset.manifest.dataset_id}  feature set {FEATURE_SET_VERSION}")
    print(f"{matrix.n_features} features over {len(matrix):,} bars, "
          f"usable from bar {matrix.warmup:,}\n")
    print(f"{'feature':<16}{'first':>7}{'mean':>11}{'std':>11}{'min':>11}{'max':>11}")
    print("-" * 67)
    for j, name in enumerate(matrix.names):
        column = matrix.values[matrix.warmup:, j]
        first = int(np.argmax(np.isfinite(matrix.values[:, j])))
        print(f"{name:<16}{first:>7}{column.mean():>11.4f}{column.std():>11.4f}"
              f"{column.min():>11.4f}{column.max():>11.4f}")
    return 0


def cmd_regimes(args) -> int:
    dataset = resolve_dataset(Path(args.root), args.dataset)
    labels = label_regimes(dataset)
    print(f"{dataset.manifest.dataset_id}")
    print(labels.summary())
    segments = [s for s in labels.segments() if s[2] is not Regime.UNKNOWN]
    if segments:
        lengths = np.array([stop - start for start, stop, _ in segments])
        print(f"\nsegment length: median {int(np.median(lengths))} bars, "
              f"p90 {int(np.quantile(lengths, 0.9))}, longest {int(lengths.max())}")
    return 0


def cmd_split(args) -> int:
    dataset = resolve_dataset(Path(args.root), args.dataset)
    config = default_config(dataset, warmup_bars=args.warmup)
    if args.scale != 1.0:
        config = scale_config(config, args.scale)
    if args.expanding:
        config = SplitConfig(**{**config.__dict__, "expanding": True})
    plan = make_walk_forward(dataset, config)
    if args.save:
        print(f"saved to {plan.save(args.root)}")
    print(plan.summary())
    return 0


def cmd_backtest(args) -> int:
    dataset = resolve_dataset(Path(args.root), args.dataset)
    settings = load_settings()
    policy = make_policy(args.policy, settings.strategy_parameters, seed=args.seed)
    result = run_backtest(policy, dataset, config=config_for(args, args.policy),
                          seed=args.seed)
    print(f"{dataset.manifest.dataset_id}  [{dataset.manifest.source}]")
    print(f"{args.policy}: {result.metrics.describe()}")
    if result.blocked_by_risk:
        print(f"\nrisk engine refused {result.blocked_by_risk} entries:")
        for reason, count in sorted(result.block_reasons.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {count:>6}x  {reason}")
    return 0


def _walk_forward_plan(dataset, args) -> SplitPlan:
    config = default_config(dataset, warmup_bars=args.warmup)
    if args.scale != 1.0:
        config = scale_config(config, args.scale)
    return make_walk_forward(dataset, config)


def cmd_baselines(args) -> int:
    """Every strategy and every trivial baseline, through the same walk-forward."""
    root = Path(args.root)
    dataset = resolve_dataset(root, args.dataset)
    settings = load_settings()
    plan = _walk_forward_plan(dataset, args)
    seeds = tuple(range(args.seeds))

    print(f"{dataset.manifest.dataset_id}  [{dataset.manifest.source}]")
    print(f"{len(plan)} windows x {len(seeds)} seed(s), "
          f"{len(plan.windows[0].validate):,} validation bars each")
    print(f"holdout: {plan.holdout_size:,} bars SEALED\n")

    store = ExperimentStore(root / "research.db") if args.record else None
    if store:
        store.register_dataset(dataset.manifest)

    names = list(STRATEGY_POLICIES) + list(BASELINE_POLICIES)
    table = []
    for name in names:
        config = config_for(args, name)
        results = run_walk_forward(
            lambda n=name: make_policy(n, settings.strategy_parameters, seed=args.seed),
            dataset, plan, config=config, seeds=seeds,
        )
        summary = aggregate([r.metrics for r in results])
        table.append((name, summary, results))
        if store:
            policy_id = store.register_policy(
                agent=name, kind="strategy" if name in STRATEGY_POLICIES else "baseline")
            for result in results:
                store.record_metrics(
                    policy_id=policy_id, dataset_id=dataset.manifest.dataset_id,
                    metrics=result.metrics, split_id=plan.split_id,
                    window=result.window, seed=result.seed,
                )
        print(f"  {name:<21} done "
              f"({int(sum(r.metrics['trades'] for r in results)):,} trades)")

    print(f"\n{'policy':<21}{'median ret%':>12}{'IQR':>8}{'sortino':>9}{'maxDD%':>8}"
          f"{'win%':>7}{'R:R':>6}{'trades':>8}{'TP%':>6}{'SL%':>6}")
    print("-" * 91)
    for name, summary, results in sorted(table, key=lambda row: -row[1]["total_return_pct"].median):
        total = int(sum(r.metrics["trades"] for r in results))
        weighted = lambda key: (                       # noqa: E731 - local helper
            sum(r.metrics[key] * r.metrics["trades"] for r in results) / total if total else 0.0
        )
        print(f"{name:<21}{summary['total_return_pct'].median:>+12.2f}"
              f"{summary['total_return_pct'].iqr:>8.2f}"
              f"{summary['sortino'].median:>9.2f}"
              f"{summary['max_drawdown_pct'].median:>8.2f}"
              f"{summary['win_rate_pct'].median:>7.1f}"
              f"{summary['realised_rr'].median:>6.2f}"
              f"{total:>8}"
              f"{weighted('exit_take_profit_pct'):>6.1f}"
              f"{weighted('exit_stop_loss_pct'):>6.1f}")

    all_trades = sum(int(sum(r.metrics["trades"] for r in results)) for _, _, results in table)
    tp_hits = sum(
        sum(r.metrics["exit_take_profit_pct"] * r.metrics["trades"] for r in results)
        for _, _, results in table
    )
    if all_trades:
        print(f"\ntake-profit hit rate across every policy: "
              f"{tp_hits / all_trades:.2f}% of {all_trades:,} trades")
    if store:
        print(f"\nrecorded to {root / 'research.db'}: {store.counts()}")
        store.close()
    return 0


def cmd_report(args) -> int:
    root = Path(args.root)
    store = ExperimentStore(root / "research.db")
    try:
        manifests = [
            m for m in list_datasets(root)
            if args.dataset is None or m.dataset_id == args.dataset
        ]
        if not manifests:
            print("nothing recorded yet — run `python research.py baselines --record`")
            return 0
        try:
            source = assert_single_source(manifests, allow_cross_venue=args.cross_venue)
        except ValueError as exc:
            print(f"cannot produce one report: {exc}", file=sys.stderr)
            return 1

        banner = ("SYNTHETIC — simulator output, not evidence about live markets"
                  if source.is_synthetic else f"REAL MARKET — {source}")
        print(f"=== {banner} ===\n")
        for manifest in manifests:
            print("  " + manifest.describe())

        board = store.leaderboard(args.metric, dataset_id=args.dataset, limit=args.limit)
        if not board:
            print(f"\nno {args.metric} recorded yet")
            return 0
        print(f"\n{args.metric}, by median across windows and seeds:\n")
        print(f"{'policy':<24}{'median':>10}{'q25':>10}{'q75':>10}{'n':>6}")
        print("-" * 60)
        for row in board:
            print(f"{row['policy_id']:<24}{row['median']:>10.3f}{row['q25']:>10.3f}"
                  f"{row['q75']:>10.3f}{row['n']:>6}")
        return 0
    finally:
        store.close()


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="data directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_backtest_flags(p):
        p.add_argument("--equity", type=float, default=10_000.0)
        p.add_argument("--risk", type=float, default=0.02, help="risk per trade, as a fraction")
        p.add_argument("--leverage", type=float, default=1.0)
        p.add_argument("--stop-loss", type=float, default=None, dest="stop_loss")
        p.add_argument("--take-profit", type=float, default=None, dest="take_profit")
        p.add_argument("--spread", type=float, default=2.0, help="assumed spread, bps")
        p.add_argument("--fee", type=float, default=None, help="fee rate; default from config")
        p.add_argument("--seed", type=int, default=0)

    def add_split_flags(p):
        p.add_argument("--warmup", type=int, default=600)
        p.add_argument("--scale", type=float, default=1.0,
                       help="shrink every window span, for shorter datasets")

    sub.add_parser("venues", help="list every venue ccxt can reach").set_defaults(func=cmd_venues)

    p = sub.add_parser("synth", help="generate deterministic synthetic bars")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--bars", type=int, default=60_000)
    p.add_argument("--seed", type=int, default=20250328)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("collect", help="download real OHLCV from any ccxt venue")
    p.add_argument("--venue", required=True, help="ccxt exchange id, e.g. binance, bybit, okx")
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--from", dest="since", default=None, help="YYYY-MM-DD")
    p.add_argument("--to", dest="until", default=None, help="YYYY-MM-DD")
    p.add_argument("--market-type", default="spot", dest="market_type")
    p.add_argument("--testnet", action="store_true")
    p.add_argument("--max-pages", type=int, default=None, dest="max_pages")
    p.set_defaults(func=cmd_collect)

    sub.add_parser("datasets", help="list datasets and their provenance").set_defaults(
        func=cmd_datasets)

    p = sub.add_parser("features", help="build features and show their distributions")
    p.add_argument("--dataset", default=None)
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("regimes", help="label market regimes causally")
    p.add_argument("--dataset", default=None)
    p.set_defaults(func=cmd_regimes)

    p = sub.add_parser("split", help="build a purged walk-forward plan")
    p.add_argument("--dataset", default=None)
    p.add_argument("--expanding", action="store_true")
    p.add_argument("--save", action="store_true")
    add_split_flags(p)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("backtest", help="run one policy over a whole dataset")
    p.add_argument("--dataset", default=None)
    p.add_argument("--policy", required=True,
                   choices=sorted(STRATEGY_POLICIES + BASELINE_POLICIES))
    add_backtest_flags(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("baselines", help="every policy through the same walk-forward")
    p.add_argument("--dataset", default=None)
    p.add_argument("--seeds", type=int, default=1, help="number of seeds per window")
    p.add_argument("--record", action="store_true", help="write results to research.db")
    add_backtest_flags(p)
    add_split_flags(p)
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("report", help="leaderboard from recorded results")
    p.add_argument("--dataset", default=None)
    p.add_argument("--metric", default="total_return_pct")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--cross-venue", action="store_true", dest="cross_venue",
                   help="deliberately compare across exchanges")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_setup(args.verbose)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
