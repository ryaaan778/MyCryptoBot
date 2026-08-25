"""What the agent has already tried, and how it turned out.

This is the part that makes the loop a *learning* loop rather than five
independent guesses an hour. Every stance is written down when it is taken;
when the position it opened finally closes, the realised P&L is attached to it.
The next prompt is built from that record.

Three things get fed back, and each answers a different failure the model has
without them:

**Recent decisions with outcomes** — otherwise the model re-proposes an idea
that lost money last week, having no way to know it did.

**Calibration** — stated confidence against realised hit rate, bucketed. A model
that says 0.9 and is right 45% of the time cannot discover that from inside a
single call. Showing it the table is the cheapest correction available, and it
works: confidence claims tighten within a few dozen decisions.

**Fired invalidations** — the stance said "wrong if we lose 61,800"; we lost
61,800. That pairing is the highest-signal thing in the file, because the model
wrote the test itself and the market marked it.

Storage is JSONL, one file per bot, appended and never rewritten. That choice is
deliberate: the record of what an agent believed is an audit trail, and an audit
trail you can UPDATE is not one. It also means no schema migration and no lock
contention with the trading store.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from ..models import now_ms
from .schema import TradeStance

logger = logging.getLogger("jojo.llm.memory")

#: Confidence buckets for the calibration table.
_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 0.9), (0.9, 1.01)
)


@dataclass
class DecisionRecord:
    """One stance, plus what the market later did about it."""

    id: str
    bot_id: str
    symbol: str
    ts: int
    action: str
    confidence: float
    reason: str
    thesis: str
    invalidation: str
    horizon_minutes: int
    sources: list[str] = field(default_factory=list)
    price_at_decision: float = 0.0

    # Filled in later, when the position this opened closes.
    outcome_pnl: float | None = None
    outcome_ts: int | None = None
    outcome_note: str = ""

    @property
    def resolved(self) -> bool:
        return self.outcome_pnl is not None

    @property
    def won(self) -> bool | None:
        if self.outcome_pnl is None:
            return None
        return self.outcome_pnl > 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "DecisionRecord | None":
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            return None
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


class LLMMemory:
    """Append-only decision history for one bot, with feedback rendering."""

    def __init__(self, path: Path, *, window: int = 20) -> None:
        self.path = Path(path)
        self.window = max(1, int(window))
        self._records: list[DecisionRecord] = []
        self._load()

    # ---- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = DecisionRecord.from_json(line)
                if record is not None:
                    self._records.append(record)
        except OSError as exc:                      # pragma: no cover - disk-dependent
            logger.warning("could not read decision memory %s: %s", self.path, exc)

    def _append_line(self, payload: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except OSError as exc:                      # pragma: no cover - disk-dependent
            logger.warning("could not write decision memory %s: %s", self.path, exc)

    # ---- writing -----------------------------------------------------------

    def record(
        self, bot_id: str, symbol: str, stance: TradeStance, *, price: float = 0.0
    ) -> DecisionRecord:
        record = DecisionRecord(
            id=f"dec-{now_ms()}-{len(self._records)}",
            bot_id=bot_id, symbol=symbol, ts=now_ms(),
            action=stance.action.value, confidence=float(stance.confidence),
            reason=stance.reason, thesis=stance.thesis,
            invalidation=stance.invalidation,
            horizon_minutes=int(stance.horizon_minutes),
            sources=list(stance.sources), price_at_decision=float(price),
        )
        self._records.append(record)
        self._append_line(asdict(record))
        return record

    def resolve_latest(self, pnl: float, *, note: str = "") -> DecisionRecord | None:
        """Attach an outcome to the most recent unresolved directional decision.

        Appended as a new line rather than an edit — the file stays append-only,
        and the loader takes the last mention of an id as current.
        """
        for record in reversed(self._records):
            if record.resolved or record.action in ("HOLD",):
                continue
            record.outcome_pnl = float(pnl)
            record.outcome_ts = now_ms()
            record.outcome_note = note
            self._append_line(asdict(record))
            return record
        return None

    # ---- reading -----------------------------------------------------------

    @property
    def records(self) -> list[DecisionRecord]:
        """Latest state per decision id, oldest first."""
        latest: dict[str, DecisionRecord] = {}
        for record in self._records:
            latest[record.id] = record
        return sorted(latest.values(), key=lambda r: r.ts)

    def recent(self, limit: int | None = None) -> list[DecisionRecord]:
        return self.records[-(limit or self.window):]

    def calibration(self) -> list[tuple[str, int, float]]:
        """(bucket label, resolved count, hit rate) — the honesty table."""
        resolved = [r for r in self.records if r.resolved]
        table: list[tuple[str, int, float]] = []
        for low, high in _BUCKETS:
            bucket = [r for r in resolved if low <= r.confidence < high]
            if not bucket:
                continue
            wins = sum(1 for r in bucket if r.won)
            table.append((f"{low:.2f}-{min(high, 1.0):.2f}", len(bucket),
                          wins / len(bucket)))
        return table

    # ---- prompt rendering --------------------------------------------------

    def render(self) -> str:
        """The feedback block that goes into the next prompt."""
        recent = self.recent()
        if not recent:
            return "No prior decisions on this market. This is your first call."

        lines = ["YOUR RECENT DECISIONS ON THIS MARKET (oldest first)"]
        for record in recent:
            if record.resolved:
                verdict = f"{'WON' if record.won else 'LOST'} {record.outcome_pnl:+.2f}"
            else:
                verdict = "still open"
            lines.append(
                f"- {record.action} @ conf {record.confidence:.2f} — {record.reason} "
                f"→ {verdict}"
            )
            if record.resolved and not record.won and record.invalidation:
                lines.append(f"    you said it would be wrong if: {record.invalidation}")

        table = self.calibration()
        if table:
            lines.append("")
            lines.append("YOUR CALIBRATION SO FAR (stated confidence vs. actual hit rate)")
            for label, count, rate in table:
                flag = ""
                mid = sum(float(x) for x in label.split("-")) / 2
                if count >= 5 and rate < mid - 0.15:
                    flag = "  <- OVERCONFIDENT"
                elif count >= 5 and rate > mid + 0.15:
                    flag = "  <- UNDERCONFIDENT"
                lines.append(f"- said {label}: right {rate:.0%} of {count}{flag}")

        resolved = [r for r in self.records if r.resolved]
        if resolved:
            total = sum(r.outcome_pnl or 0.0 for r in resolved)
            wins = sum(1 for r in resolved if r.won)
            lines.append("")
            lines.append(
                f"LIFETIME: {len(resolved)} resolved, {wins} won "
                f"({wins / len(resolved):.0%}), net {total:+.2f}"
            )
        return "\n".join(lines)


def memory_path(data_dir: str | Path, bot_id: str) -> Path:
    safe = "".join(ch for ch in bot_id if ch.isalnum() or ch in "-_") or "bot"
    return Path(data_dir) / "llm" / f"{safe}.jsonl"
