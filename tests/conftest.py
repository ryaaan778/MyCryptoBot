from __future__ import annotations

import pytest

from backend.config import Settings, load_settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    cfg = load_settings(overrides={
        "provider": "simulated",
        "initial_balance": 10_000.0,
        "data_dir": str(tmp_path / "data"),
    })
    return cfg
