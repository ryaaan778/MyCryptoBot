"""Dataset registry with provenance.

Every dataset carries a ``source`` — the sentinel ``SYNTHETIC`` or **a ccxt
exchange id** — and a sha256 over its canonical bytes. That label is not
decoration: it propagates to every metric row, and the reporting layer refuses
to aggregate across sources. There is no code path that produces a metric
without a dataset, and therefore none that produces a metric without a source.
Synthetic results cannot be presented as evidence about real markets by
accident.

The source is an open set rather than a two-value enum because the system
targets every venue ccxt reaches, not one exchange. Adding Bybit or OKX is a
string, not a code change. What the type does enforce is that the string is a
*real* venue: an unrecognised id is rejected at construction, so a typo can
never quietly become a new "exchange" that nothing else in the system knows
about.

Two segregation rules follow from this, and :func:`assert_single_source`
enforces both. Synthetic never aggregates with real — that one is absolute.
And real venues do not silently blend with one another either: fees, spreads,
liquidity and leverage caps differ per exchange, so a figure averaged across
venues describes no venue that actually exists. Cross-venue comparison is
available, but only as a deliberate, labelled operation.

Storage is compressed ``.npz`` rather than parquet: OHLCV is a pure numeric
matrix, and npz avoids a ~90MB pandas/pyarrow dependency for no practical loss.
``to_csv`` exists for interoperability with other tools.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from backend.market.base import TIMEFRAME_MS

logger = logging.getLogger("research.datasets")

COLLECTOR_VERSION = "1.0.0"

# Datasets shorter than this can't support a walk-forward split worth trusting.
MIN_USABLE_ROWS = 500


SYNTHETIC_ID = "SYNTHETIC"

_EXCHANGE_ID = re.compile(r"^[a-z][a-z0-9_]{1,31}$")

# ccxt is a declared dependency of the trading server, but research code must
# still be able to *read* an archived dataset on a machine without it. So the
# strict check runs when ccxt is importable and a syntactic check stands in
# when it is not — a saved dataset never becomes unloadable because of a
# missing optional import.
_ccxt_ids: frozenset[str] | None = None
_warned_no_ccxt = False


def known_exchanges() -> frozenset[str]:
    """Every ccxt exchange id, or an empty set when ccxt is unavailable."""
    global _ccxt_ids
    if _ccxt_ids is None:
        try:
            import ccxt

            _ccxt_ids = frozenset(ccxt.exchanges)
        except Exception:
            _ccxt_ids = frozenset()
    return _ccxt_ids


class DataSource(str):
    """Where the bars came from. The distinction is load-bearing.

    A ``str`` subclass rather than an ``Enum`` because the set of venues is
    open — ccxt exposes ~100 exchanges and the point of this system is to reach
    all of them — while the *validity* of a venue id is still checked. It
    serialises as a plain string, compares as one, and adds the two properties
    the rest of the codebase actually branches on.
    """

    __slots__ = ()

    def __new__(cls, value: "str | DataSource") -> "DataSource":
        raw = str(value).strip()
        if not raw:
            raise ValueError("data source cannot be empty")
        if raw.upper() == SYNTHETIC_ID:
            return super().__new__(cls, SYNTHETIC_ID)

        exchange_id = raw.lower()
        if not _EXCHANGE_ID.match(exchange_id):
            raise ValueError(
                f"invalid data source {value!r}: expected {SYNTHETIC_ID} or a ccxt "
                "exchange id such as 'binance', 'bybit', 'okx', 'kraken'"
            )

        known = known_exchanges()
        if known:
            if exchange_id not in known:
                near = sorted(e for e in known if e.startswith(exchange_id[:3]))[:6]
                raise ValueError(
                    f"unknown exchange {exchange_id!r} — not in ccxt's {len(known)} "
                    + (f"exchanges. Did you mean one of {near}?" if near
                       else "supported exchanges.")
                )
        else:
            global _warned_no_ccxt
            if not _warned_no_ccxt:
                _warned_no_ccxt = True
                logger.warning(
                    "ccxt is not importable; exchange ids are being accepted on shape "
                    "alone. Install ccxt to validate %r against the real venue list.",
                    exchange_id,
                )
        return super().__new__(cls, exchange_id)

    # ``.value`` keeps parity with the Enum this replaced, so call sites that
    # were written against it keep working.
    @property
    def value(self) -> str:
        return str(self)

    @property
    def is_synthetic(self) -> bool:
        return str(self) == SYNTHETIC_ID

    @property
    def is_real_market(self) -> bool:
        return not self.is_synthetic

    @property
    def exchange_id(self) -> str:
        """The ccxt id. Raises for synthetic data, which has no venue."""
        if self.is_synthetic:
            raise ValueError("synthetic data has no exchange")
        return str(self)

    def __repr__(self) -> str:
        return f"DataSource({str(self)!r})"


DataSource.SYNTHETIC = DataSource(SYNTHETIC_ID)   # type: ignore[attr-defined]


class DataQualityError(ValueError):
    """Raised when candles violate an invariant a backtest depends on."""


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    source: str
    symbol: str
    timeframe: str
    start_ts: int
    end_ts: int
    rows: int
    sha256: str
    collector_version: str
    created_at: int
    seed: int | None = None
    gaps: int = 0
    notes: str = ""

    @property
    def data_source(self) -> DataSource:
        return DataSource(self.source)

    @property
    def is_real_market(self) -> bool:
        return self.data_source.is_real_market

    @property
    def is_synthetic(self) -> bool:
        return self.data_source.is_synthetic

    def describe(self) -> str:
        span_days = (self.end_ts - self.start_ts) / 86_400_000
        return (
            f"{self.dataset_id}  {self.source:<11} {self.symbol:<10} {self.timeframe:<4} "
            f"{self.rows:>7,} rows  {span_days:>7.1f}d  gaps={self.gaps}"
        )


@dataclass
class Dataset:
    """Immutable OHLCV bars plus provenance.

    Arrays are parallel and index-aligned; ``ts`` is the bar OPEN time in
    milliseconds, which matters for look-ahead reasoning: a bar indexed ``i`` is
    only fully known after ``ts[i] + timeframe_ms``.
    """

    manifest: DatasetManifest
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    # ---- construction ------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        *,
        ts: Sequence[int] | np.ndarray,
        open: Sequence[float] | np.ndarray,
        high: Sequence[float] | np.ndarray,
        low: Sequence[float] | np.ndarray,
        close: Sequence[float] | np.ndarray,
        volume: Sequence[float] | np.ndarray,
        source: DataSource,
        symbol: str,
        timeframe: str,
        seed: int | None = None,
        notes: str = "",
        validate: bool = True,
    ) -> "Dataset":
        ts_a = np.asarray(ts, dtype=np.int64)
        arrays = {
            "open": np.asarray(open, dtype=np.float64),
            "high": np.asarray(high, dtype=np.float64),
            "low": np.asarray(low, dtype=np.float64),
            "close": np.asarray(close, dtype=np.float64),
            "volume": np.asarray(volume, dtype=np.float64),
        }
        for name, arr in arrays.items():
            if arr.shape != ts_a.shape:
                raise DataQualityError(
                    f"{name} has {arr.shape} rows but ts has {ts_a.shape}"
                )

        digest = _checksum(ts_a, arrays)
        gaps = count_gaps(ts_a, timeframe) if ts_a.size else 0
        manifest = DatasetManifest(
            dataset_id=_dataset_id(source, symbol, timeframe, ts_a, digest),
            source=source.value,
            symbol=symbol,
            timeframe=timeframe,
            start_ts=int(ts_a[0]) if ts_a.size else 0,
            end_ts=int(ts_a[-1]) if ts_a.size else 0,
            rows=int(ts_a.size),
            sha256=digest,
            collector_version=COLLECTOR_VERSION,
            created_at=int(time.time() * 1000),
            seed=seed,
            gaps=gaps,
            notes=notes,
        )
        dataset = cls(manifest=manifest, ts=ts_a, **arrays)
        if validate:
            dataset.validate()
        return dataset

    @classmethod
    def from_candles(
        cls,
        candles: Iterable,
        *,
        source: DataSource,
        symbol: str,
        timeframe: str,
        seed: int | None = None,
        notes: str = "",
    ) -> "Dataset":
        """Build from anything with ts/open/high/low/close/volume attributes."""
        rows = list(candles)
        return cls.from_arrays(
            ts=[c.ts for c in rows],
            open=[c.open for c in rows],
            high=[c.high for c in rows],
            low=[c.low for c in rows],
            close=[c.close for c in rows],
            volume=[c.volume for c in rows],
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            seed=seed,
            notes=notes,
        )

    # ---- invariants --------------------------------------------------------

    def validate(self) -> None:
        """Fail loudly on anything a backtest would silently misinterpret."""
        if self.manifest.rows == 0:
            raise DataQualityError("dataset is empty")

        if np.any(np.diff(self.ts) <= 0):
            raise DataQualityError("timestamps must be strictly increasing")

        for name in ("open", "high", "low", "close", "volume"):
            arr = getattr(self, name)
            if not np.all(np.isfinite(arr)):
                bad = int(np.count_nonzero(~np.isfinite(arr)))
                raise DataQualityError(f"{name} contains {bad} non-finite values")

        for name in ("open", "high", "low", "close"):
            if np.any(getattr(self, name) <= 0):
                raise DataQualityError(f"{name} contains non-positive prices")
        if np.any(self.volume < 0):
            raise DataQualityError("volume contains negative values")

        # OHLC consistency — a violated bar makes stop/target logic meaningless.
        body_high = np.maximum(self.open, self.close)
        body_low = np.minimum(self.open, self.close)
        if np.any(self.high < body_high - 1e-9):
            raise DataQualityError("high is below the candle body on some bars")
        if np.any(self.low > body_low + 1e-9):
            raise DataQualityError("low is above the candle body on some bars")

    # ---- access ------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.ts.size)

    @property
    def timeframe_ms(self) -> int:
        return TIMEFRAME_MS.get(self.manifest.timeframe, 300_000)

    @property
    def bars_per_day(self) -> float:
        return 86_400_000 / self.timeframe_ms

    def slice(self, start: int, stop: int) -> "Dataset":
        """A view over ``[start, stop)`` that keeps provenance but is not re-registered."""
        stop = min(stop, len(self))
        start = max(0, start)
        return Dataset(
            manifest=self.manifest,
            ts=self.ts[start:stop],
            open=self.open[start:stop],
            high=self.high[start:stop],
            low=self.low[start:stop],
            close=self.close[start:stop],
            volume=self.volume[start:stop],
        )

    def matrix(self) -> np.ndarray:
        """`(rows, 5)` OHLCV, for feature builders that want one array."""
        return np.column_stack([self.open, self.high, self.low, self.close, self.volume])

    # ---- persistence -------------------------------------------------------

    def save(self, root: str | Path) -> Path:
        directory = Path(root) / "datasets" / self.manifest.dataset_id
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "candles.npz",
            ts=self.ts, open=self.open, high=self.high,
            low=self.low, close=self.close, volume=self.volume,
        )
        (directory / "manifest.json").write_text(
            json.dumps(asdict(self.manifest), indent=2) + "\n"
        )
        logger.info("saved %s (%d rows) to %s", self.manifest.dataset_id, len(self), directory)
        return directory

    @classmethod
    def load(cls, root: str | Path, dataset_id: str, *, verify: bool = True) -> "Dataset":
        directory = Path(root) / "datasets" / dataset_id
        if not directory.exists():
            raise FileNotFoundError(f"no dataset {dataset_id!r} under {root}")

        manifest = DatasetManifest(**json.loads((directory / "manifest.json").read_text()))
        with np.load(directory / "candles.npz") as payload:
            dataset = cls(
                manifest=manifest,
                ts=payload["ts"], open=payload["open"], high=payload["high"],
                low=payload["low"], close=payload["close"], volume=payload["volume"],
            )

        if verify:
            arrays = {
                name: getattr(dataset, name)
                for name in ("open", "high", "low", "close", "volume")
            }
            actual = _checksum(dataset.ts, arrays)
            if actual != manifest.sha256:
                # Silent corruption would invalidate every experiment that cited
                # this dataset, so it is an error rather than a warning.
                raise DataQualityError(
                    f"{dataset_id} checksum mismatch: manifest says {manifest.sha256[:12]}, "
                    f"data hashes to {actual[:12]}"
                )
        return dataset

    def to_csv(self, path: str | Path) -> Path:
        """Export for inspection in other tools — never read back as canonical."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = np.column_stack(
            [self.ts, self.open, self.high, self.low, self.close, self.volume]
        )
        np.savetxt(
            target, rows, delimiter=",", comments="",
            header="ts,open,high,low,close,volume",
            fmt=["%d", "%.10g", "%.10g", "%.10g", "%.10g", "%.10g"],
        )
        return target


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def list_datasets(root: str | Path) -> list[DatasetManifest]:
    directory = Path(root) / "datasets"
    if not directory.exists():
        return []
    out: list[DatasetManifest] = []
    for entry in sorted(directory.iterdir()):
        manifest_path = entry / "manifest.json"
        if manifest_path.is_file():
            try:
                out.append(DatasetManifest(**json.loads(manifest_path.read_text())))
            except Exception:
                logger.warning("skipping unreadable manifest at %s", manifest_path)
    return out


def find_dataset(
    root: str | Path,
    *,
    source: DataSource | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> list[DatasetManifest]:
    return [
        m for m in list_datasets(root)
        if (source is None or m.source == source.value)
        and (symbol is None or m.symbol == symbol)
        and (timeframe is None or m.timeframe == timeframe)
    ]


def count_gaps(ts: np.ndarray, timeframe: str) -> int:
    """Missing bars, inferred from spacing.

    Gaps matter: an indicator computed across one silently blends two
    non-adjacent points in time, and the backtest cannot tell.
    """
    if ts.size < 2:
        return 0
    step = TIMEFRAME_MS.get(timeframe)
    if not step:
        return 0
    deltas = np.diff(ts)
    missing = np.maximum(0, np.round(deltas / step).astype(np.int64) - 1)
    return int(missing.sum())


def _checksum(ts: np.ndarray, arrays: dict[str, np.ndarray]) -> str:
    """sha256 over canonical bytes, in a fixed field order."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(ts, dtype=np.int64).tobytes())
    for name in ("open", "high", "low", "close", "volume"):
        digest.update(np.ascontiguousarray(arrays[name], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _dataset_id(
    source: DataSource, symbol: str, timeframe: str, ts: np.ndarray, digest: str
) -> str:
    """Deterministic and human-readable: same bytes always yield the same id."""
    compact = symbol.replace("/", "").lower()
    start = int(ts[0]) if ts.size else 0
    return f"{source.value.lower()}_{compact}_{timeframe}_{start}_{digest[:8]}"


def assert_single_source(
    manifests: Sequence[DatasetManifest], *, allow_cross_venue: bool = False
) -> DataSource:
    """Guard against mixing incomparable data in one report.

    Called by every aggregation path, and it enforces two separate rules.

    **Synthetic never mixes with real.** Absolute, and ``allow_cross_venue``
    does not relax it. Averaging a simulator's results with a real exchange's
    produces a number that means nothing while looking authoritative — the most
    dangerous kind of number this system could emit.

    **Real venues do not silently blend either.** Fees, spreads, liquidity and
    leverage caps differ per exchange, so a Sharpe averaged over Binance and
    Kraken describes no venue anyone can actually trade. Set
    ``allow_cross_venue=True`` to opt in deliberately — which is exactly what a
    robustness comparison wants, since a policy that works on one venue's
    microstructure and nowhere else is overfit to that venue.

    Returns the single source, or the shared ``SYNTHETIC``/venue marker for a
    permitted cross-venue set.
    """
    if not manifests:
        raise ValueError("no datasets to aggregate")

    sources = {DataSource(m.source) for m in manifests}
    synthetic = {s for s in sources if s.is_synthetic}
    venues = sources - synthetic

    if synthetic and venues:
        raise ValueError(
            "refusing to aggregate synthetic data with real-market data "
            f"({sorted(str(v) for v in venues)}). Simulator results are not evidence "
            "about live markets and must be reported separately."
        )

    if len(venues) > 1 and not allow_cross_venue:
        raise ValueError(
            "refusing to aggregate across venues: "
            f"{sorted(str(v) for v in venues)}. Fees, spreads and leverage limits "
            "differ per exchange, so a blended figure describes no venue that exists. "
            "Pass allow_cross_venue=True to compare them deliberately."
        )

    return next(iter(sources)) if len(sources) == 1 else next(iter(venues))


def venues_of(manifests: Sequence[DatasetManifest]) -> list[str]:
    """Distinct real venues present, for labelling a cross-venue report."""
    return sorted({m.source for m in manifests if DataSource(m.source).is_real_market})
