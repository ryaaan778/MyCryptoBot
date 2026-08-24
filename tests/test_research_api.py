"""The research surface on the trading API.

Research is a *side channel*. The rules it has to obey are that it never delays
or displaces trading data, never takes the trading API down when it is missing
or malformed, and never presents simulator output as evidence about a live
market.
"""

from __future__ import annotations

import ast
import pathlib
import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.models import AgentResearch, BotStatus, ResearchState
from backend.store import RESEARCH_SCHEMA


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as opened:
        yield opened


# ---- the dependency rule ---------------------------------------------------


def test_the_trading_server_does_not_import_research():
    """The trading side reads the research tables; it never imports the package."""
    for path in pathlib.Path("backend").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("research") for a in node.names), path
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("research"), path


def test_research_states_exist_and_are_distinct_from_trading_states():
    research = {BotStatus.RESEARCHING, BotStatus.TRAINING, BotStatus.VALIDATING}
    trading = {BotStatus.TRADING, BotStatus.ANALYZING, BotStatus.IDLE,
               BotStatus.PAUSED, BotStatus.OFFLINE, BotStatus.HALTED}
    assert not (research & trading)
    assert len(BotStatus) == 9


# ---- the endpoint ----------------------------------------------------------


def test_a_fresh_install_reports_no_research_rather_than_empty_panels(client):
    payload = client.get("/api/research").json()
    assert payload["available"] is False
    assert payload["experiments"] == 0
    assert payload["synthetic_only"] is True


def test_the_endpoint_survives_a_missing_research_database(client):
    """Research is optional. Its absence must not break the trading API."""
    assert client.get("/api/research").status_code == 200
    assert client.get("/api/state").status_code == 200


def test_a_corrupt_research_database_does_not_take_down_trading(settings, tmp_path):
    """A malformed side channel is a degraded panel, not an outage."""
    broken = pathlib.Path(settings.data_dir) / "jojo.db"
    broken.parent.mkdir(parents=True, exist_ok=True)

    with TestClient(create_app(settings)) as client:
        connection = sqlite3.connect(broken)
        connection.execute("DROP TABLE IF EXISTS experiments")
        connection.execute("CREATE TABLE experiments (nonsense TEXT)")
        connection.commit()
        connection.close()

        assert client.get("/api/research").status_code == 200
        assert client.get("/api/state").status_code == 200
        assert client.get("/api/health").json()["status"]


def test_recorded_research_is_reported(settings):
    database = pathlib.Path(settings.data_dir) / "jojo.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript(RESEARCH_SCHEMA)
    connection.execute(
        "INSERT INTO datasets VALUES ('d1','SYNTHETIC','BTC/USDT','5m',0,1,10,'x','1',NULL,0,'',0)")
    connection.execute(
        "INSERT INTO experiments VALUES ('e1','KIRA','fade the breakout','{}','TRAINING',"
        "0,NULL,NULL,0,5)")
    connection.execute(
        "INSERT INTO policies VALUES ('KIRA_v1','KIRA',1,'learned','PPO',NULL,'{}',NULL,"
        "NULL,NULL,0,NULL,NULL,'CANDIDATE',0)")
    connection.execute("INSERT INTO allocations VALUES (1,'KIRA','KIRA_v1',2500.0,'rank 1',0)")
    connection.commit()
    connection.close()

    with TestClient(create_app(settings)) as client:
        payload = client.get("/api/research").json()

    assert payload["available"] is True
    assert payload["datasets"] == 1 and payload["experiments"] == 1
    assert payload["paper_capital_deployed"] == 2500.0
    kira = next(a for a in payload["agents"] if a["agent"] == "KIRA")
    assert kira["current_status"] == "TRAINING"
    assert kira["hypothesis"] == "fade the breakout"
    assert kira["paper_allocation"] == 2500.0


def test_a_promoted_champion_is_reported_with_its_version(settings):
    database = pathlib.Path(settings.data_dir) / "jojo.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript(RESEARCH_SCHEMA)
    connection.execute(
        "INSERT INTO policies VALUES ('JOSEPH_v3','JOSEPH',3,'learned','PPO',NULL,'{}',NULL,"
        "NULL,NULL,0,NULL,NULL,'PROMOTED',0)")
    connection.execute(
        "INSERT INTO champion_history VALUES (1,'JOSEPH','JOSEPH_v3','PROMOTED','beat it',NULL,1)")
    connection.commit()
    connection.close()

    with TestClient(create_app(settings)) as client:
        payload = client.get("/api/research").json()
    joseph = next(a for a in payload["agents"] if a["agent"] == "JOSEPH")
    assert joseph["champion_policy"] == "JOSEPH_v3"
    assert joseph["champion_version"] == 3


def test_a_retired_champion_is_reported_as_having_none(settings):
    database = pathlib.Path(settings.data_dir) / "jojo.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript(RESEARCH_SCHEMA)
    connection.execute(
        "INSERT INTO policies VALUES ('KIRA_v1','KIRA',1,'learned','PPO',NULL,'{}',NULL,"
        "NULL,NULL,0,NULL,NULL,'RETIRED',0)")
    connection.execute(
        "INSERT INTO champion_history VALUES (1,'KIRA','KIRA_v1','PROMOTED','x',NULL,1)")
    connection.execute(
        "INSERT INTO champion_history VALUES (2,'KIRA','KIRA_v1','RETIRED','decayed',NULL,2)")
    connection.commit()
    connection.close()

    with TestClient(create_app(settings)) as client:
        payload = client.get("/api/research").json()
    kira = next(a for a in payload["agents"] if a["agent"] == "KIRA")
    assert kira["champion_policy"] is None


# ---- the honesty flag ------------------------------------------------------


def test_synthetic_only_is_true_for_simulator_data(settings):
    database = pathlib.Path(settings.data_dir) / "jojo.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript(RESEARCH_SCHEMA)
    connection.execute(
        "INSERT INTO datasets VALUES ('d1','SYNTHETIC','BTC/USDT','5m',0,1,10,'x','1',NULL,0,'',0)")
    connection.commit()
    connection.close()

    with TestClient(create_app(settings)) as client:
        payload = client.get("/api/research").json()
    assert payload["synthetic_only"] is True
    assert payload["sources"] == ["SYNTHETIC"]


def test_real_venue_data_clears_the_synthetic_flag(settings):
    database = pathlib.Path(settings.data_dir) / "jojo.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript(RESEARCH_SCHEMA)
    connection.execute(
        "INSERT INTO datasets VALUES ('d1','SYNTHETIC','BTC/USDT','5m',0,1,10,'x','1',NULL,0,'',0)")
    connection.execute(
        "INSERT INTO datasets VALUES ('d2','binance','BTC/USDT','5m',0,1,10,'y','1',NULL,0,'',0)")
    connection.commit()
    connection.close()

    with TestClient(create_app(settings)) as client:
        payload = client.get("/api/research").json()
    assert payload["synthetic_only"] is False
    assert set(payload["sources"]) == {"SYNTHETIC", "binance"}


def test_the_default_state_is_labelled_synthetic():
    """The safe default is the cautious one: assume simulator until told otherwise."""
    assert ResearchState().synthetic_only is True
    assert ResearchState().available is False


# ---- the websocket ---------------------------------------------------------


def test_research_arrives_as_its_own_frame_after_the_snapshot(client):
    """Trading data goes first. Research must never delay the snapshot."""
    with client.websocket_connect("/ws") as socket:
        first = socket.receive_json()
        assert first["type"] == "snapshot"
        second = socket.receive_json()
        assert second["type"] == "research"
        assert "available" in second["data"]


def test_research_can_be_requested_on_demand(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()   # snapshot
        socket.receive_json()   # research
        socket.send_json({"type": "research.request"})
        frame = socket.receive_json()
        assert frame["type"] == "research"


def test_agent_research_knows_when_it_is_busy():
    for status in ("PROPOSED", "TRAINING", "VALIDATING"):
        assert AgentResearch(agent="A", current_status=status).is_busy
    for status in ("REJECTED", "CANDIDATE", "PROMOTED", None):
        assert not AgentResearch(agent="A", current_status=status).is_busy
