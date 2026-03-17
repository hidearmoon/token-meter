"""Integration tests for TokenTracker and the public init() API."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import token_meter
from token_meter.config import TokenMeterConfig
from token_meter.core import TokenTracker
from token_meter.storage.sqlite import SQLiteStorage


@pytest.fixture(autouse=True)
def reset_global_tracker():
    """Ensure each test starts with a clean global tracker."""
    token_meter.disable()
    token_meter._tracker = None
    yield
    token_meter.disable()
    token_meter._tracker = None


@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "test.db")


class TestInitAPI:
    def test_init_returns_tracker(self, tmp_db_path):
        tracker = token_meter.init(db_path=tmp_db_path)
        assert tracker is not None
        assert tracker.is_active

    def test_init_twice_returns_same_tracker(self, tmp_db_path):
        t1 = token_meter.init(db_path=tmp_db_path)
        t2 = token_meter.init(db_path=tmp_db_path)
        assert t1 is t2

    def test_disable_stops_tracker(self, tmp_db_path):
        tracker = token_meter.init(db_path=tmp_db_path)
        assert tracker.is_active
        token_meter.disable()
        assert not tracker.is_active

    def test_get_tracker_before_init(self):
        assert token_meter.get_tracker() is None

    def test_get_tracker_after_init(self, tmp_db_path):
        token_meter.init(db_path=tmp_db_path)
        t = token_meter.get_tracker()
        assert t is not None
        assert t.is_active

    def test_get_tracker_after_disable(self, tmp_db_path):
        token_meter.init(db_path=tmp_db_path)
        token_meter.disable()
        t = token_meter.get_tracker()
        assert t is not None
        assert not t.is_active

    def test_custom_project(self, tmp_db_path):
        tracker = token_meter.init(project="my-service", db_path=tmp_db_path)
        assert tracker._config.project == "my-service"

    def test_custom_providers_subset(self, tmp_db_path):
        tracker = token_meter.init(
            db_path=tmp_db_path, providers=["openai"]
        )
        assert tracker._config.providers == ["openai"]

    def test_invalid_provider_raises(self, tmp_db_path):
        with pytest.raises(ValueError, match="Unknown provider"):
            token_meter.init(db_path=tmp_db_path, providers=["invalid-llm"])

    def test_db_file_created(self, tmp_path):
        db_path = str(tmp_path / "new_dir" / "usage.db")
        token_meter.init(db_path=db_path)
        assert Path(db_path).exists()


class TestTokenTrackerLifecycle:
    def _make_tracker(self, tmp_path) -> tuple[TokenTracker, SQLiteStorage]:
        config = TokenMeterConfig(
            project="test",
            db_path=tmp_path / "test.db",
        )
        storage = SQLiteStorage(config.db_path)
        return TokenTracker(config, storage), storage

    def test_start_and_stop(self, tmp_path):
        tracker, storage = self._make_tracker(tmp_path)
        tracker.start()
        assert tracker.is_active
        tracker.stop()
        assert not tracker.is_active
        storage.close()

    def test_double_start_is_idempotent(self, tmp_path):
        tracker, storage = self._make_tracker(tmp_path)
        tracker.start()
        patchers_count = len(tracker._patchers)
        tracker.start()
        assert len(tracker._patchers) == patchers_count
        tracker.stop()
        storage.close()

    def test_stop_without_start_is_noop(self, tmp_path):
        tracker, storage = self._make_tracker(tmp_path)
        tracker.stop()  # should not raise
        storage.close()

    def test_query_delegates_to_storage(self, tmp_path):
        tracker, storage = self._make_tracker(tmp_path)
        results = tracker.query()
        assert results == []
        storage.close()

    def test_aggregate_delegates_to_storage(self, tmp_path):
        tracker, storage = self._make_tracker(tmp_path)
        stats = tracker.aggregate()
        assert stats["call_count"] == 0
        storage.close()

    def test_close_stops_tracker(self, tmp_path):
        tracker, _ = self._make_tracker(tmp_path)
        tracker.start()
        tracker.close()
        assert not tracker.is_active


class TestConfig:
    def test_default_config(self):
        cfg = TokenMeterConfig()
        assert cfg.project == "default"
        assert "openai" in cfg.providers
        assert "anthropic" in cfg.providers
        assert "google" in cfg.providers

    def test_custom_providers(self):
        cfg = TokenMeterConfig(providers=["openai"])
        assert cfg.providers == ["openai"]

    def test_invalid_provider(self):
        with pytest.raises(ValueError):
            TokenMeterConfig(providers=["nonexistent"])

    def test_db_path_expanded(self):
        cfg = TokenMeterConfig(db_path="~/mydata/usage.db")
        assert "~" not in str(cfg.db_path)

    def test_from_kwargs_env_db_path(self, monkeypatch, tmp_path):
        db = str(tmp_path / "env.db")
        monkeypatch.setenv("TOKEN_METER_DB_PATH", db)
        cfg = TokenMeterConfig.from_kwargs()
        assert str(cfg.db_path) == str(Path(db).resolve())

    def test_from_kwargs_env_project(self, monkeypatch):
        monkeypatch.setenv("TOKEN_METER_PROJECT", "env-project")
        cfg = TokenMeterConfig.from_kwargs()
        assert cfg.project == "env-project"

    def test_from_kwargs_kwarg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("TOKEN_METER_PROJECT", "env-project")
        cfg = TokenMeterConfig.from_kwargs(project="kwarg-project")
        assert cfg.project == "kwarg-project"
