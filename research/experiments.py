"""Experiment records: what was tried, on which bytes, and what came out.

A research system that cannot answer "which exact data produced this number"
is not producing evidence, it is producing anecdotes. Three properties make
that answerable here:

**Metrics cannot exist without a dataset.** ``policy_metrics.dataset_id`` is a
foreign key, foreign keys are enforced (SQLite needs telling, and this module
tells it), so there is no way to write a metric without naming the bytes it
came from — and therefore no way to write one whose data source is unknown.

**Aggregation across sources is refused, not warned about.** :meth:`compare`
routes every multi-dataset read through
:func:`research.datasets.assert_single_source`, so a query that would average a
simulator result with a live-market one raises instead of returning a number.

**Everything is append-only except status.** Experiments and policies keep their
history; a rejected experiment stays in the table with its rejection reason,
because the record of what did *not* work is what stops an agent proposing it
again next week.

Synchronous ``sqlite3`` rather than the trading server's async store: research
runs from a CLI, and an event loop would buy nothing. Both halves share one
database file and the DDL lives in ``backend.store`` so the dependency still
runs one way only.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

from backend.store import RESEARCH_SCHEMA

from .datasets import DataSource, DatasetManifest, assert_single_source
from .metrics import MetricSet

logger = logging.getLogger("research.experiments")


class ExperimentStatus(str, Enum):
    PROPOSED = "PROPOSED"
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    REJECTED = "REJECTED"
    CANDIDATE = "CANDIDATE"
    PAPER_TRADING = "PAPER_TRADING"
    PROMOTED = "PROMOTED"
    RETIRED = "RETIRED"


#: The only transitions the pipeline allows. Enforced so a policy cannot appear
#: in PROMOTED without having passed through VALIDATING and CANDIDATE first —
#: which is the whole point of having stages.
_ALLOWED: dict[ExperimentStatus, tuple[ExperimentStatus, ...]] = {
    ExperimentStatus.PROPOSED: (ExperimentStatus.TRAINING, ExperimentStatus.REJECTED),
    ExperimentStatus.TRAINING: (ExperimentStatus.VALIDATING, ExperimentStatus.REJECTED),
    ExperimentStatus.VALIDATING: (ExperimentStatus.CANDIDATE, ExperimentStatus.REJECTED),
    ExperimentStatus.CANDIDATE: (ExperimentStatus.PAPER_TRADING, ExperimentStatus.REJECTED),
    ExperimentStatus.PAPER_TRADING: (ExperimentStatus.PROMOTED, ExperimentStatus.REJECTED),
    ExperimentStatus.PROMOTED: (ExperimentStatus.RETIRED,),
    ExperimentStatus.REJECTED: (),
    ExperimentStatus.RETIRED: (),
}


class EvaluationStage(str, Enum):
    WALK_FORWARD = "WALK_FORWARD"
    HOLDOUT = "HOLDOUT"
    PAPER = "PAPER"
    SHADOW = "SHADOW"


class MemoryKind(str, Enum):
    HYPOTHESIS = "HYPOTHESIS"
    OUTCOME = "OUTCOME"
    LESSON = "LESSON"
    DEAD_END = "DEAD_END"


def _now() -> int:
    return int(time.time() * 1000)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class PolicyRecord:
    policy_id: str
    agent: str
    version: int
    kind: str
    status: str


class ExperimentStore:
    """Synchronous record store for the research pipeline."""

    def __init__(self, path: str | Path, *, create: bool = True) -> None:
        self.path = Path(path)
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        # SQLite ships with foreign keys OFF. Without this line the dataset_id
        # constraint below is decorative and a metric with no provenance writes
        # happily.
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(RESEARCH_SCHEMA)
        self._migrate()
        self.db.commit()

    def _migrate(self) -> None:
        """Bring an older database up to the current schema.

        ``CREATE TABLE IF NOT EXISTS`` cannot change a constraint on a table that
        already exists, so a database created before ``policies`` was keyed on
        the seed keeps rejecting the second seed of a version with an opaque
        IntegrityError. Rebuilding is safe because every column is carried over;
        the alternative is telling people to delete their research history.

        The rename runs under ``legacy_alter_table``. Without it SQLite helpfully
        rewrites every foreign key that pointed at ``policies`` to point at the
        temporary name instead — and once the temporary table is dropped, those
        tables reference something that no longer exists and every insert fails
        with "no such table". :meth:`_repair_dangling_references` undoes that
        damage where an earlier version already caused it.
        """
        self._repair_dangling_references()

        row = self.db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='policies'"
        ).fetchone()
        if row is None or "UNIQUE (agent, version, seed)" in (row["sql"] or ""):
            return

        logger.info("migrating `policies` to key on (agent, version, seed)")
        self.db.execute("PRAGMA foreign_keys = OFF")
        self.db.execute("PRAGMA legacy_alter_table = ON")
        try:
            self.db.execute("ALTER TABLE policies RENAME TO policies_legacy")
            self.db.executescript(RESEARCH_SCHEMA)
            columns = [r["name"] for r in self.db.execute("PRAGMA table_info(policies)")]
            legacy = {r["name"] for r in self.db.execute("PRAGMA table_info(policies_legacy)")}
            shared = ", ".join(c for c in columns if c in legacy)
            self.db.execute(
                f"INSERT INTO policies ({shared}) SELECT {shared} FROM policies_legacy")
            self.db.execute("DROP TABLE policies_legacy")
            self.db.commit()
        finally:
            self.db.execute("PRAGMA legacy_alter_table = OFF")
            self.db.execute("PRAGMA foreign_keys = ON")

    def _repair_dangling_references(self) -> None:
        """Rebuild tables whose foreign keys point at a table that no longer exists.

        An earlier migration renamed ``policies`` without ``legacy_alter_table``,
        which silently repointed ``policy_metrics`` and ``evaluations`` at the
        temporary name. Everything reads fine until the first insert, which then
        fails with an error naming a table nobody has ever heard of.
        """
        damaged = [
            r["name"] for r in self.db.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' "
                "AND sql LIKE '%policies_legacy%' AND name != 'policies_legacy'"
            )
        ]
        if not damaged:
            return

        logger.warning("repairing %s: foreign keys point at a dropped table", damaged)
        self.db.execute("PRAGMA foreign_keys = OFF")
        self.db.execute("PRAGMA legacy_alter_table = ON")
        try:
            for table in damaged:
                self.db.execute(f"ALTER TABLE {table} RENAME TO {table}_broken")
                self.db.executescript(RESEARCH_SCHEMA)
                columns = [r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")]
                old = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table}_broken)")}
                shared = ", ".join(c for c in columns if c in old)
                self.db.execute(
                    f"INSERT INTO {table} ({shared}) SELECT {shared} FROM {table}_broken")
                self.db.execute(f"DROP TABLE {table}_broken")
            self.db.commit()
        finally:
            self.db.execute("PRAGMA legacy_alter_table = OFF")
            self.db.execute("PRAGMA foreign_keys = ON")

    def __enter__(self) -> "ExperimentStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    # ---- datasets ----------------------------------------------------------

    def register_dataset(self, manifest: DatasetManifest) -> str:
        """Idempotent: a dataset id is a hash of its bytes, so re-registering is a no-op.

        A *changed* dataset under an existing id is a different matter — it
        means two different byte sequences claim the same identity, which would
        silently re-point every metric that cites it. That raises.
        """
        existing = self.db.execute(
            "SELECT sha256, rows FROM datasets WHERE dataset_id = ?", (manifest.dataset_id,)
        ).fetchone()
        if existing is not None:
            if existing["sha256"] != manifest.sha256:
                raise ValueError(
                    f"dataset {manifest.dataset_id} is already registered with a different "
                    f"checksum ({existing['sha256'][:12]} vs {manifest.sha256[:12]}); "
                    "metrics already cite the registered bytes"
                )
            return manifest.dataset_id

        self.db.execute(
            """INSERT INTO datasets (dataset_id, source, symbol, timeframe, start_ts, end_ts,
                                     rows, sha256, collector_version, seed, gaps, notes, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (manifest.dataset_id, manifest.source, manifest.symbol, manifest.timeframe,
             manifest.start_ts, manifest.end_ts, manifest.rows, manifest.sha256,
             manifest.collector_version, manifest.seed, manifest.gaps, manifest.notes,
             manifest.created_at),
        )
        self.db.commit()
        return manifest.dataset_id

    def dataset(self, dataset_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
        ).fetchone()
        return dict(row) if row else None

    def datasets(self, source: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM datasets"
        params: tuple = ()
        if source is not None:
            sql += " WHERE source = ?"
            params = (str(DataSource(source)),)
        return [dict(r) for r in self.db.execute(sql + " ORDER BY created_at DESC", params)]

    # ---- experiments -------------------------------------------------------

    def create_experiment(
        self, *, agent: str, hypothesis: str, spec: dict[str, Any],
        seed: int | None = None, parent: str | None = None,
        experiment_id: str | None = None,
    ) -> str:
        experiment_id = experiment_id or _new_id("exp")
        now = _now()
        self.db.execute(
            """INSERT INTO experiments (experiment_id, agent, hypothesis, spec_json, status,
                                        seed, parent_experiment_id, decision_reason,
                                        created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (experiment_id, agent, hypothesis, json.dumps(spec, sort_keys=True),
             ExperimentStatus.PROPOSED.value, seed, parent, None, now, now),
        )
        self.db.commit()
        return experiment_id

    def set_experiment_status(
        self, experiment_id: str, status: ExperimentStatus, reason: str = ""
    ) -> None:
        row = self.db.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no experiment {experiment_id!r}")
        current = ExperimentStatus(row["status"])
        if status is not current and status not in _ALLOWED[current]:
            raise ValueError(
                f"{experiment_id}: {current.value} -> {status.value} is not a legal transition "
                f"(allowed: {[s.value for s in _ALLOWED[current]] or 'none, this is terminal'})"
            )
        if status is ExperimentStatus.REJECTED and not reason.strip():
            # A rejection with no reason teaches the agent nothing and it will
            # propose the same thing again.
            raise ValueError("rejecting an experiment requires a reason")
        self.db.execute(
            "UPDATE experiments SET status = ?, decision_reason = ?, updated_at = ? "
            "WHERE experiment_id = ?",
            (status.value, reason or None, _now(), experiment_id),
        )
        self.db.commit()

    def experiment(self, experiment_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        return dict(row) if row else None

    def experiments(self, agent: str | None = None, status: str | None = None) -> list[dict]:
        clauses, params = [], []
        if agent:
            clauses.append("agent = ?"); params.append(agent)
        if status:
            clauses.append("status = ?"); params.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return [dict(r) for r in self.db.execute(
            f"SELECT * FROM experiments{where} ORDER BY created_at DESC", tuple(params))]

    # ---- policies ----------------------------------------------------------

    def register_policy(
        self, *, agent: str, kind: str, algo: str | None = None,
        experiment_id: str | None = None, hyperparams: dict | None = None,
        feature_set: Sequence[str] | None = None, feature_set_version: str | None = None,
        reward_version: str | None = None, seed: int | None = None,
        model_path: str | None = None, weights_path: str | None = None,
        status: ExperimentStatus = ExperimentStatus.CANDIDATE,
        version: int | None = None, policy_id: str | None = None,
    ) -> str:
        if version is None:
            row = self.db.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM policies WHERE agent = ?", (agent,)
            ).fetchone()
            version = int(row["v"]) + 1
        policy_id = policy_id or f"{agent}_v{version}"
        self.db.execute(
            """INSERT INTO policies (policy_id, agent, version, kind, algo, experiment_id,
                                     hyperparams_json, feature_set, feature_set_version,
                                     reward_version, seed, model_path, weights_path,
                                     status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (policy_id, agent, version, kind, algo, experiment_id,
             json.dumps(hyperparams or {}, sort_keys=True),
             ",".join(feature_set) if feature_set else None,
             feature_set_version, reward_version, seed, model_path, weights_path,
             status.value, _now()),
        )
        self.db.commit()
        return policy_id

    def policy(self, policy_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM policies WHERE policy_id = ?", (policy_id,)
        ).fetchone()
        return dict(row) if row else None

    def policies(self, agent: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM policies"
        params: tuple = ()
        if agent:
            sql += " WHERE agent = ?"
            params = (agent,)
        return [dict(r) for r in self.db.execute(sql + " ORDER BY agent, version DESC", params)]

    # ---- metrics -----------------------------------------------------------

    def record_metrics(
        self, *, policy_id: str, dataset_id: str, metrics: MetricSet,
        split_id: str | None = None, window: int | None = None,
        regime: str | None = None, seed: int | None = None,
    ) -> int:
        """Write one run's metrics in long format. Fails if the dataset is unknown."""
        now = _now()
        rows = [
            (policy_id, dataset_id, split_id, window, regime, seed,
             name, float(value), metrics.version, now)
            for name, value in sorted(metrics.values.items())
        ]
        try:
            self.db.executemany(
                """INSERT INTO policy_metrics (policy_id, dataset_id, split_id, window, regime,
                                               seed, metric, value, metrics_version, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"cannot record metrics for policy {policy_id!r} on dataset {dataset_id!r}: "
                "both must be registered first, so that every metric carries its provenance"
            ) from exc
        self.db.commit()
        return len(rows)

    def metrics_for(
        self, policy_id: str, *, metric: str | None = None, dataset_id: str | None = None,
        split_id: str | None = None, regime: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses, params = ["policy_id = ?"], [policy_id]
        for column, value in (("metric", metric), ("dataset_id", dataset_id),
                              ("split_id", split_id), ("regime", regime)):
            if value is not None:
                clauses.append(f"{column} = ?"); params.append(value)
        return [dict(r) for r in self.db.execute(
            f"SELECT * FROM policy_metrics WHERE {' AND '.join(clauses)} "
            "ORDER BY window, seed, metric", tuple(params))]

    def manifests_for_metrics(self, policy_ids: Iterable[str]) -> list[DatasetManifest]:
        ids = list(policy_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.db.execute(
            f"""SELECT DISTINCT d.* FROM datasets d
                JOIN policy_metrics m ON m.dataset_id = d.dataset_id
                WHERE m.policy_id IN ({placeholders})""",
            tuple(ids),
        ).fetchall()
        return [
            DatasetManifest(
                dataset_id=r["dataset_id"], source=r["source"], symbol=r["symbol"],
                timeframe=r["timeframe"], start_ts=r["start_ts"], end_ts=r["end_ts"],
                rows=r["rows"], sha256=r["sha256"],
                collector_version=r["collector_version"], created_at=r["created_at"],
                seed=r["seed"], gaps=r["gaps"] or 0, notes=r["notes"] or "",
            )
            for r in rows
        ]

    def compare(
        self, policy_ids: Sequence[str], metric: str, *,
        split_id: str | None = None, regime: str | None = None,
        allow_cross_venue: bool = False,
    ) -> dict[str, list[float]]:
        """Metric values per policy, but only if the underlying data is comparable.

        The segregation check runs *before* any number is returned. Ranking a
        policy scored on a simulator against one scored on Binance would produce
        a leaderboard that looks authoritative and means nothing, so the call
        raises rather than ranking them.
        """
        manifests = self.manifests_for_metrics(policy_ids)
        if manifests:
            assert_single_source(manifests, allow_cross_venue=allow_cross_venue)

        out: dict[str, list[float]] = {}
        for policy_id in policy_ids:
            rows = self.metrics_for(
                policy_id, metric=metric, split_id=split_id, regime=regime
            )
            out[policy_id] = [float(r["value"]) for r in rows]
        return out

    def leaderboard(
        self, metric: str, *, dataset_id: str | None = None, split_id: str | None = None,
        regime: str | None = None, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Median of ``metric`` per policy — median, because best-of-N is a lie.

        SQLite has no median, so the ordering statistic is computed here rather
        than delegated to AVG, which a single lucky window would drag around.
        """
        clauses, params = ["metric = ?"], [metric]
        for column, value in (("dataset_id", dataset_id), ("split_id", split_id),
                              ("regime", regime)):
            if value is not None:
                clauses.append(f"{column} = ?"); params.append(value)
        rows = self.db.execute(
            f"SELECT policy_id, value FROM policy_metrics WHERE {' AND '.join(clauses)}",
            tuple(params),
        ).fetchall()

        grouped: dict[str, list[float]] = {}
        for row in rows:
            grouped.setdefault(row["policy_id"], []).append(float(row["value"]))

        import numpy as np

        table = [
            {"policy_id": pid,
             "median": float(np.median(values)),
             "q25": float(np.quantile(values, 0.25)),
             "q75": float(np.quantile(values, 0.75)),
             "n": len(values)}
            for pid, values in grouped.items()
        ]
        table.sort(key=lambda r: -r["median"])
        return table[:limit]

    # ---- evaluations, memory, champions, allocations -----------------------

    def record_evaluation(
        self, *, policy_id: str, dataset_id: str, stage: EvaluationStage,
        passed: bool, gates: dict[str, Any], split_id: str | None = None,
        reason: str = "",
    ) -> str:
        evaluation_id = _new_id("eval")
        self.db.execute(
            """INSERT INTO evaluations (evaluation_id, policy_id, dataset_id, split_id,
                                        stage, passed, gate_json, reason, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (evaluation_id, policy_id, dataset_id, split_id, stage.value,
             1 if passed else 0, json.dumps(gates, sort_keys=True, default=str),
             reason or None, _now()),
        )
        self.db.commit()
        return evaluation_id

    def evaluations(self, policy_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM evaluations WHERE policy_id = ? ORDER BY created_at DESC",
            (policy_id,))]

    def remember(
        self, *, agent: str, kind: MemoryKind, content: dict[str, Any],
        experiment_id: str | None = None, policy_id: str | None = None,
    ) -> None:
        self.db.execute(
            """INSERT INTO agent_memory (agent, kind, content_json, experiment_id,
                                         policy_id, created_at)
               VALUES (?,?,?,?,?,?)""",
            (agent, kind.value, json.dumps(content, sort_keys=True, default=str),
             experiment_id, policy_id, _now()),
        )
        self.db.commit()

    def recall(
        self, agent: str, kind: MemoryKind | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM agent_memory WHERE agent = ?"
        params: list = [agent]
        if kind is not None:
            sql += " AND kind = ?"; params.append(kind.value)
        rows = self.db.execute(sql + " ORDER BY created_at DESC LIMIT ?",
                               tuple(params + [limit]))
        return [{**dict(r), "content": json.loads(r["content_json"])} for r in rows]

    def promote(self, *, agent: str, policy_id: str, reason: str) -> None:
        previous = self.champion(agent)
        self.db.execute(
            """INSERT INTO champion_history (agent, policy_id, action, reason,
                                             previous_policy_id, ts)
               VALUES (?,?,?,?,?,?)""",
            (agent, policy_id, "PROMOTED", reason,
             previous["policy_id"] if previous else None, _now()),
        )
        self.db.commit()
        logger.info("%s promoted %s (was %s): %s", agent, policy_id,
                    previous["policy_id"] if previous else "none", reason)

    def retire(self, *, agent: str, policy_id: str, reason: str) -> None:
        self.db.execute(
            """INSERT INTO champion_history (agent, policy_id, action, reason,
                                             previous_policy_id, ts)
               VALUES (?,?,?,?,?,?)""",
            (agent, policy_id, "RETIRED", reason, None, _now()),
        )
        self.db.commit()

    def champion(self, agent: str) -> dict[str, Any] | None:
        """The current champion, or None once it has been retired."""
        row = self.db.execute(
            "SELECT * FROM champion_history WHERE agent = ? ORDER BY ts DESC, id DESC LIMIT 1",
            (agent,),
        ).fetchone()
        if row is None or row["action"] == "RETIRED":
            return None
        return dict(row)

    def champion_history(self, agent: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM champion_history WHERE agent = ? ORDER BY ts DESC, id DESC",
            (agent,))]

    def record_allocation(
        self, *, agent: str, paper_capital: float, reason: str, policy_id: str | None = None
    ) -> None:
        self.db.execute(
            "INSERT INTO allocations (agent, policy_id, paper_capital, reason, ts) "
            "VALUES (?,?,?,?,?)",
            (agent, policy_id, float(paper_capital), reason, _now()),
        )
        self.db.commit()

    def allocations(self, agent: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM allocations"
        params: list = []
        if agent:
            sql += " WHERE agent = ?"; params.append(agent)
        return [dict(r) for r in self.db.execute(
            sql + " ORDER BY ts DESC LIMIT ?", tuple(params + [limit]))]

    # ---- summary -----------------------------------------------------------

    def counts(self) -> dict[str, int]:
        tables = ("datasets", "experiments", "policies", "policy_metrics",
                  "evaluations", "agent_memory", "champion_history", "allocations")
        return {
            table: int(self.db.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"])
            for table in tables
        }
