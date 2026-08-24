"""Chronological splits: purged walk-forward, embargo, and a sealed holdout.

Random k-fold on a price series is the single most effective way to produce a
backtest that looks brilliant and loses money, so it is not available here. Every
split this module produces is forward-chaining: train on the past, validate on
the future, step forward, repeat.

Three separate protections, each against a different leak:

**Purge.** A training example near the boundary is not independent of the
validation window: its features look back, and a trade opened inside it can
still be open when validation begins. So the last ``purge_bars`` of every
training range are dropped. Purge must cover at least the feature lookback plus
the longest holding period you intend to allow — if it does not, the model
trains on bars whose outcome is a validation bar.

**Embargo.** After a validation window ends, the next ``embargo_bars`` are
correlated with it — a trade opened inside validation resolves just after it. In
forward-chaining those bars are chronologically legitimate training data, so
this is a weaker safeguard than it is in López de Prado's k-fold setting, and it
would be dishonest to imply otherwise. What it buys is comparability *across*
windows: without it, window k+1 trains on the resolution of window k's
validation trades, and the two windows' scores stop being independent
observations of the same policy. Since we report a median across windows and
seeds, that independence is the thing being measured.

**A sealed holdout.** The final slice of every dataset is ``TEST_LOCKED`` and no
walk-forward window can reach it. Development — model selection, hyperparameter
search, feature choice, the promotion gate — happens entirely on the windows.
Opening the holdout requires an explicit confirmation phrase and writes an audit
record, because the honest truth about a holdout is that it stops being one the
first time you look. The friction is the feature.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path

import numpy as np

from .datasets import Dataset

logger = logging.getLogger("research.splits")

SPLIT_VERSION = "v1"

#: Deliberately hard to type by accident, and deliberately explicit about cost.
HOLDOUT_CONFIRMATION = "I_ACCEPT_THIS_BURNS_THE_HOLDOUT"


class SplitPurpose(str, Enum):
    TRAIN = "TRAIN"
    VALIDATE = "VALIDATE"
    TEST_LOCKED = "TEST_LOCKED"


class HoldoutSealed(RuntimeError):
    """Raised when code reaches for the sealed test period without unlocking it."""


@dataclass(frozen=True)
class Range:
    """Half-open bar range ``[start, stop)``."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.stop < self.start:
            raise ValueError(f"range {self.start}..{self.stop} runs backwards")

    def __len__(self) -> int:
        return self.stop - self.start

    def __contains__(self, index: int) -> bool:
        return self.start <= index < self.stop

    def indices(self) -> np.ndarray:
        return np.arange(self.start, self.stop, dtype=np.int64)

    def overlaps(self, other: "Range") -> bool:
        return self.start < other.stop and other.start < self.stop

    def clip(self, limit: int) -> "Range":
        return Range(min(self.start, limit), min(self.stop, limit))

    def describe(self) -> str:
        return f"[{self.start:,}..{self.stop:,})"


@dataclass(frozen=True)
class SplitConfig:
    """Window geometry, in bars.

    Bars rather than days so the same config means the same thing on any
    timeframe; :func:`bars_for_days` converts when you would rather think in days.
    """

    train_bars: int
    validate_bars: int
    step_bars: int
    purge_bars: int
    embargo_bars: int
    warmup_bars: int = 0
    holdout_fraction: float = 0.15
    expanding: bool = False
    allow_overlapping_validation: bool = False
    version: str = SPLIT_VERSION

    def validate(self) -> None:
        for name in ("train_bars", "validate_bars", "step_bars"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("purge_bars", "embargo_bars", "warmup_bars"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 0.0 <= self.holdout_fraction < 0.5:
            raise ValueError("holdout_fraction must be in [0, 0.5)")
        if self.purge_bars >= self.train_bars:
            raise ValueError(
                f"purge_bars ({self.purge_bars}) would consume the whole "
                f"{self.train_bars}-bar training window"
            )
        if self.step_bars > self.validate_bars:
            # Stepping further than a validation window leaves bars that are never
            # validated by anything — a silent hole in the evaluation.
            raise ValueError(
                f"step_bars ({self.step_bars}) exceeds validate_bars "
                f"({self.validate_bars}), which would skip data entirely"
            )
        if self.step_bars < self.validate_bars and not self.allow_overlapping_validation:
            # Stepping less than a full window makes consecutive windows share
            # bars. That buys more windows out of a short dataset, but their
            # scores are then correlated — and since the promotion gate reads a
            # median across windows as if they were independent observations,
            # overlapping quietly inflates confidence. Allowed, but never by
            # default and never by accident.
            raise ValueError(
                f"step_bars ({self.step_bars}) is less than validate_bars "
                f"({self.validate_bars}), so validation windows would overlap and "
                "their scores would not be independent. Set "
                "allow_overlapping_validation=True if you accept that trade-off."
            )

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:8]


def bars_for_days(days: float, dataset: Dataset) -> int:
    return int(round(days * dataset.bars_per_day))


def default_config(dataset: Dataset, *, warmup_bars: int) -> SplitConfig:
    """The plan's geometry — train 180d, validate 45d, step 45d — in bars.

    ``purge_bars`` defaults to one day, which comfortably covers the longest
    feature lookback (60 bars) and any intraday holding period. ``embargo_bars``
    is ~1% of a validation window, following López de Prado's rule of thumb.
    """
    per_day = dataset.bars_per_day
    validate_bars = int(round(45 * per_day))
    return SplitConfig(
        train_bars=int(round(180 * per_day)),
        validate_bars=validate_bars,
        step_bars=validate_bars,
        purge_bars=int(round(per_day)),
        embargo_bars=max(1, int(round(0.01 * validate_bars))),
        warmup_bars=warmup_bars,
    )


@dataclass(frozen=True)
class Window:
    index: int
    train: Range          # contiguous span; embargo holes are removed on access
    validate: Range

    def describe(self) -> str:
        return (
            f"w{self.index:<2} train {self.train.describe():>20} ({len(self.train):>6,})"
            f"  validate {self.validate.describe():>20} ({len(self.validate):>5,})"
        )


@dataclass
class SplitPlan:
    """A frozen, reproducible set of windows plus a sealed holdout."""

    split_id: str
    dataset_id: str
    dataset_sha256: str
    dataset_rows: int
    config: SplitConfig
    windows: tuple[Window, ...]
    development: Range
    _holdout: Range
    unlocks: list[dict] = field(default_factory=list)

    # ---- the seal ----------------------------------------------------------

    @property
    def holdout_size(self) -> int:
        """How big the sealed region is. Knowing the size leaks nothing."""
        return len(self._holdout)

    @property
    def is_unlocked(self) -> bool:
        return bool(self.unlocks)

    def holdout(self) -> Range:
        """The sealed test range. Raises unless it has been deliberately unlocked."""
        if not self.is_unlocked:
            raise HoldoutSealed(
                f"the {self.holdout_size:,}-bar holdout for {self.dataset_id} is sealed. "
                "Development must use the walk-forward windows. Call unlock_holdout() "
                "with an explicit reason if you truly intend to spend it."
            )
        return self._holdout

    def unlock_holdout(
        self, *, actor: str, reason: str, confirmation: str, audit_path: str | Path | None = None
    ) -> Range:
        """Spend the holdout, on the record.

        Every unlock is appended rather than replacing the last one, so a plan
        that has been opened repeatedly says so. A holdout looked at five times
        is a validation set, and the record is what makes that visible instead
        of forgotten.
        """
        if confirmation != HOLDOUT_CONFIRMATION:
            raise HoldoutSealed(
                f"holdout unlock requires confirmation={HOLDOUT_CONFIRMATION!r}"
            )
        if not actor.strip() or not reason.strip():
            raise ValueError("holdout unlock requires a non-empty actor and reason")

        record = {
            "split_id": self.split_id,
            "dataset_id": self.dataset_id,
            "actor": actor.strip(),
            "reason": reason.strip(),
            "ts": int(time.time() * 1000),
            "prior_unlocks": len(self.unlocks),
        }
        self.unlocks.append(record)

        if len(self.unlocks) == 1:
            logger.warning(
                "HOLDOUT UNLOCKED for %s by %s: %s — results from it are one-shot",
                self.dataset_id, record["actor"], record["reason"],
            )
        else:
            logger.warning(
                "HOLDOUT UNLOCKED AGAIN (#%d) for %s by %s: %s — this is no longer an "
                "out-of-sample test and must not be reported as one",
                len(self.unlocks), self.dataset_id, record["actor"], record["reason"],
            )

        if audit_path is not None:
            path = Path(audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        return self._holdout

    # ---- window access -----------------------------------------------------

    def embargo_zones(self) -> tuple[Range, ...]:
        """Bars immediately following each validation window.

        These only intersect a *later* window's training span once two windows
        separate them: window ``k``'s training stops one purge before its own
        validation, which is before window ``k-1``'s validation even ends. So
        the embargo starts removing bars at window 2 and never affects windows
        0 and 1. That is expected, not a bug — but it does mean a two-window
        plan gets no benefit from the embargo at all.
        """
        if self.config.embargo_bars == 0:
            return ()
        return tuple(
            Range(w.validate.stop, min(w.validate.stop + self.config.embargo_bars,
                                       self.dataset_rows))
            for w in self.windows
        )

    def train_indices(self, window: Window) -> np.ndarray:
        """Training bars with embargoed regions removed.

        The purge is already baked into ``window.train``; the embargo has to be
        subtracted here because a later window's contiguous training span can
        swallow an earlier window's embargo zone.
        """
        idx = window.train.indices()
        for zone in self.embargo_zones():
            if zone.overlaps(window.train):
                idx = idx[(idx < zone.start) | (idx >= zone.stop)]
        return idx

    def validate_indices(self, window: Window) -> np.ndarray:
        return window.validate.indices()

    def __len__(self) -> int:
        return len(self.windows)

    def __iter__(self):
        return iter(self.windows)

    # ---- invariants --------------------------------------------------------

    def check(self) -> None:
        """Assert every property the rest of the harness relies on."""
        cfg = self.config
        if not self.windows:
            raise ValueError("split plan has no windows")

        for i, w in enumerate(self.windows):
            if w.index != i:
                raise ValueError(f"window indices are not sequential at {i}")
            if len(w.train) <= 0 or len(w.validate) <= 0:
                raise ValueError(f"window {i} has an empty range")
            if w.train.start < cfg.warmup_bars:
                raise ValueError(
                    f"window {i} trains from bar {w.train.start}, inside the "
                    f"{cfg.warmup_bars}-bar warm-up where features are not formed"
                )
            if w.train.stop > w.validate.start:
                raise ValueError(f"window {i} train overlaps validate")
            gap = w.validate.start - w.train.stop
            if gap < cfg.purge_bars:
                raise ValueError(
                    f"window {i} leaves a {gap}-bar gap, below the {cfg.purge_bars}-bar purge"
                )
            if w.validate.overlaps(self._holdout):
                raise ValueError(f"window {i} validates inside the sealed holdout")
            if w.train.overlaps(self._holdout):
                raise ValueError(f"window {i} trains inside the sealed holdout")
            if w.validate.stop > self.development.stop:
                raise ValueError(f"window {i} runs past the development range")

        for a, b in zip(self.windows, self.windows[1:]):
            if b.validate.start <= a.validate.start:
                raise ValueError("validation windows do not advance chronologically")
            if a.validate.overlaps(b.validate) and not cfg.allow_overlapping_validation:
                raise ValueError(
                    f"windows {a.index} and {b.index} validate on overlapping bars"
                )

        # Every development bar after the first validation must be validated by
        # at least one window, or evaluation has a hole in it.
        covered = np.zeros(self.dataset_rows, dtype=np.int32)
        for w in self.windows:
            covered[w.validate.start:w.validate.stop] += 1
        if covered.max() > 1 and not cfg.allow_overlapping_validation:
            raise ValueError("some bars are validated by more than one window")
        first, last = self.windows[0].validate.start, self.windows[-1].validate.stop
        if int(covered[first:last].min()) == 0:
            raise ValueError("gap in validation coverage between the first and last window")

        # And nothing anywhere may reach into the holdout.
        if int(covered[self._holdout.start:self._holdout.stop].sum()) != 0:
            raise ValueError("validation coverage extends into the sealed holdout")

    # ---- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "split_id": self.split_id,
            "dataset_id": self.dataset_id,
            "dataset_sha256": self.dataset_sha256,
            "dataset_rows": self.dataset_rows,
            "config": asdict(self.config),
            "windows": [
                {"index": w.index,
                 "train": [w.train.start, w.train.stop],
                 "validate": [w.validate.start, w.validate.stop]}
                for w in self.windows
            ],
            "development": [self.development.start, self.development.stop],
            "holdout": [self._holdout.start, self._holdout.stop],
            "unlocks": self.unlocks,
        }

    def save(self, root: str | Path) -> Path:
        directory = Path(root) / "splits"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.split_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, root: str | Path, split_id: str) -> "SplitPlan":
        path = Path(root) / "splits" / f"{split_id}.json"
        payload = json.loads(path.read_text())
        return cls(
            split_id=payload["split_id"],
            dataset_id=payload["dataset_id"],
            dataset_sha256=payload["dataset_sha256"],
            dataset_rows=payload["dataset_rows"],
            config=SplitConfig(**payload["config"]),
            windows=tuple(
                Window(index=w["index"], train=Range(*w["train"]), validate=Range(*w["validate"]))
                for w in payload["windows"]
            ),
            development=Range(*payload["development"]),
            _holdout=Range(*payload["holdout"]),
            unlocks=payload.get("unlocks", []),
        )

    def bind(self, dataset: Dataset) -> None:
        """Refuse to apply this plan to bars it was not built from.

        Index ranges are meaningless against different data, and a plan reused
        on a re-downloaded dataset with one extra bar would silently shift every
        window. The checksum makes that impossible rather than unlikely.
        """
        if dataset.manifest.sha256 != self.dataset_sha256:
            raise ValueError(
                f"split {self.split_id} was built for dataset {self.dataset_id} "
                f"(sha {self.dataset_sha256[:12]}), not {dataset.manifest.dataset_id} "
                f"(sha {dataset.manifest.sha256[:12]})"
            )

    def summary(self) -> str:
        lines = [
            f"split {self.split_id}  {self.dataset_id}",
            f"  {len(self.windows)} windows over {len(self.development):,} development bars "
            f"| holdout {self.holdout_size:,} bars SEALED"
            + (f" (UNLOCKED x{len(self.unlocks)})" if self.is_unlocked else ""),
            f"  purge {self.config.purge_bars} bars, embargo {self.config.embargo_bars} bars, "
            f"{'expanding' if self.config.expanding else 'rolling'} train"
            + (", OVERLAPPING validation (scores are correlated)"
               if self.config.allow_overlapping_validation else ""),
        ]
        lines.extend("  " + w.describe() for w in self.windows)
        return "\n".join(lines)


def make_walk_forward(
    dataset: Dataset, config: SplitConfig, *, split_id: str | None = None
) -> SplitPlan:
    """Build the window set, or explain exactly why the dataset is too short."""
    config.validate()
    rows = len(dataset)

    holdout_size = int(round(rows * config.holdout_fraction))
    development = Range(0, rows - holdout_size)
    holdout = Range(rows - holdout_size, rows)

    usable = len(development) - config.warmup_bars
    minimum = config.train_bars + config.purge_bars + config.validate_bars
    if usable < minimum:
        raise ValueError(
            f"dataset {dataset.manifest.dataset_id} gives {usable:,} usable development "
            f"bars ({rows:,} total − {holdout_size:,} holdout − {config.warmup_bars:,} "
            f"warm-up) but one window needs {minimum:,} "
            f"({config.train_bars:,} train + {config.purge_bars:,} purge + "
            f"{config.validate_bars:,} validate). Collect more history, shorten the "
            f"windows, or reduce holdout_fraction."
        )

    windows: list[Window] = []
    validate_start = config.warmup_bars + config.train_bars + config.purge_bars
    while validate_start + config.validate_bars <= development.stop:
        validate = Range(validate_start, validate_start + config.validate_bars)
        train_stop = validate_start - config.purge_bars
        train_start = (
            config.warmup_bars if config.expanding
            else max(config.warmup_bars, train_stop - config.train_bars)
        )
        windows.append(Window(index=len(windows), train=Range(train_start, train_stop),
                              validate=validate))
        validate_start += config.step_bars

    if not windows:
        raise ValueError("no complete walk-forward window fits in this dataset")

    plan = SplitPlan(
        split_id=split_id or _split_id(dataset, config),
        dataset_id=dataset.manifest.dataset_id,
        dataset_sha256=dataset.manifest.sha256,
        dataset_rows=rows,
        config=config,
        windows=tuple(windows),
        development=development,
        _holdout=holdout,
    )
    plan.check()
    logger.info(
        "%s: %d windows, %s train, %s validate, holdout %d bars sealed",
        plan.split_id, len(windows),
        "expanding" if config.expanding else f"{config.train_bars} bar",
        config.validate_bars, holdout_size,
    )
    return plan


def scale_config(config: SplitConfig, factor: float) -> SplitConfig:
    """Shrink or grow every span together — for short datasets and for tests."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    def s(v: int, floor: int = 1) -> int:
        return max(floor, int(round(v * factor)))
    return replace(
        config,
        train_bars=s(config.train_bars),
        validate_bars=s(config.validate_bars),
        step_bars=s(config.step_bars),
        purge_bars=s(config.purge_bars, floor=0),
        embargo_bars=s(config.embargo_bars, floor=0),
    )


def _split_id(dataset: Dataset, config: SplitConfig) -> str:
    return f"wf_{dataset.manifest.dataset_id[:24]}_{config.fingerprint()}"
