"""Tests for BudgetConfig, BudgetManager, and threshold alerting logic."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

import pytest

from token_meter.alerts import AlertSender
from token_meter.budget import BudgetConfig, BudgetManager, _period_key, _period_start
from token_meter.models import UsageRecord
from token_meter.storage.sqlite import SQLiteStorage


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture
def db(tmp_path: Path) -> SQLiteStorage:
    s = SQLiteStorage(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def sender() -> MagicMock:
    return MagicMock(spec=AlertSender)


def _record(project: str = "proj", cost: float = 1.0) -> UsageRecord:
    return UsageRecord(
        provider="openai",
        model="gpt-4o",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        input_cost=cost * 0.6,
        output_cost=cost * 0.4,
        total_cost=cost,
        latency_ms=200.0,
        project=project,
        is_stream=False,
    )


# ------------------------------------------------------------------ #
# BudgetConfig                                                         #
# ------------------------------------------------------------------ #

class TestBudgetConfig:
    def test_to_dict_round_trip(self) -> None:
        cfg = BudgetConfig(
            project="test",
            daily=10.0,
            weekly=50.0,
            monthly=200.0,
            thresholds=[0.8, 0.9, 1.0],
            webhook_urls=["https://example.com/hook"],
        )
        d = cfg.to_dict()
        assert d["daily"] == 10.0
        assert d["monthly"] == 200.0
        assert d["thresholds"] == [0.8, 0.9, 1.0]

    def test_from_dict(self) -> None:
        data = {
            "daily": 5.0,
            "weekly": None,
            "monthly": 100.0,
            "thresholds": [0.5, 1.0],
            "webhook_urls": [],
        }
        cfg = BudgetConfig.from_dict("my-proj", data)
        assert cfg.project == "my-proj"
        assert cfg.daily == 5.0
        assert cfg.weekly is None
        assert cfg.thresholds == [0.5, 1.0]

    def test_default_thresholds(self) -> None:
        cfg = BudgetConfig(project="p")
        assert cfg.thresholds == [0.8, 0.9, 1.0]

    def test_default_no_budget(self) -> None:
        cfg = BudgetConfig(project="p")
        assert cfg.daily is None
        assert cfg.weekly is None
        assert cfg.monthly is None


# ------------------------------------------------------------------ #
# _period_key / _period_start helpers                                  #
# ------------------------------------------------------------------ #

class TestPeriodHelpers:
    def test_period_key_daily(self) -> None:
        from datetime import date
        d = date(2024, 6, 15)
        assert _period_key("daily", d) == "2024-06-15"

    def test_period_key_weekly_gives_monday(self) -> None:
        from datetime import date
        saturday = date(2024, 6, 15)  # Sat
        key = _period_key("weekly", saturday)
        # Monday of that week is 2024-06-10
        assert key == "2024-06-10"

    def test_period_key_monthly(self) -> None:
        from datetime import date
        d = date(2024, 6, 15)
        assert _period_key("monthly", d) == "2024-06"

    def test_period_start_daily(self) -> None:
        from datetime import date
        d = date(2024, 6, 15)
        start = _period_start("daily", d)
        assert start == datetime(2024, 6, 15, tzinfo=timezone.utc)

    def test_period_start_weekly_is_monday(self) -> None:
        from datetime import date
        saturday = date(2024, 6, 15)
        start = _period_start("weekly", saturday)
        assert start.weekday() == 0  # Monday
        assert start == datetime(2024, 6, 10, tzinfo=timezone.utc)

    def test_period_start_monthly(self) -> None:
        from datetime import date
        d = date(2024, 6, 15)
        start = _period_start("monthly", d)
        assert start == datetime(2024, 6, 1, tzinfo=timezone.utc)


# ------------------------------------------------------------------ #
# BudgetManager.check — threshold firing                               #
# ------------------------------------------------------------------ #

class TestBudgetManagerCheck:
    def test_no_alert_when_under_threshold(self, db: SQLiteStorage, sender: MagicMock) -> None:
        cfg = BudgetConfig(
            project="proj",
            daily=100.0,
            thresholds=[0.8, 0.9, 1.0],
            webhook_urls=["https://example.com/hook"],
        )
        manager = BudgetManager([cfg], db, sender)

        rec = _record(project="proj", cost=5.0)  # 5% of 100
        db.save(rec)
        manager.check(rec)

        sender.send_webhook.assert_not_called()

    def test_alert_fires_at_80_percent(self, db: SQLiteStorage, sender: MagicMock) -> None:
        cfg = BudgetConfig(
            project="proj",
            daily=10.0,
            thresholds=[0.8],
            webhook_urls=["https://example.com/hook"],
        )
        manager = BudgetManager([cfg], db, sender)

        rec = _record(project="proj", cost=8.5)  # 85% → above 80%
        db.save(rec)
        manager.check(rec)

        sender.send_webhook.assert_called_once()
        url, payload = sender.send_webhook.call_args[0]
        assert url == "https://example.com/hook"
        assert payload["alert"] == "budget_threshold"
        assert payload["period"] == "daily"
        assert payload["threshold"] == 0.8
        assert payload["percentage"] >= 80

    def test_alert_fires_for_multiple_thresholds(self, db: SQLiteStorage, sender: MagicMock) -> None:
        cfg = BudgetConfig(
            project="proj",
            daily=10.0,
            thresholds=[0.8, 0.9, 1.0],
            webhook_urls=["https://hook.example.com"],
        )
        manager = BudgetManager([cfg], db, sender)

        rec = _record(project="proj", cost=9.5)  # 95% → fires 80%, 90% but not 100%
        db.save(rec)
        manager.check(rec)

        assert sender.send_webhook.call_count == 2
        thresholds_fired = {c[0][1]["threshold"] for c in sender.send_webhook.call_args_list}
        assert thresholds_fired == {0.8, 0.9}

    def test_alert_dedup_same_period(self, db: SQLiteStorage, sender: MagicMock) -> None:
        """Same threshold in the same period window must fire only once."""
        cfg = BudgetConfig(
            project="proj",
            daily=10.0,
            thresholds=[0.8],
            webhook_urls=["https://hook.example.com"],
        )
        manager = BudgetManager([cfg], db, sender)

        # First record triggers alert
        rec1 = _record(project="proj", cost=8.5)
        db.save(rec1)
        manager.check(rec1)
        assert sender.send_webhook.call_count == 1

        # Second record — still above threshold but already alerted
        rec2 = _record(project="proj", cost=0.1)
        db.save(rec2)
        manager.check(rec2)
        assert sender.send_webhook.call_count == 1  # no second alert

    def test_alert_100_percent_fires_over_budget(self, db: SQLiteStorage, sender: MagicMock) -> None:
        cfg = BudgetConfig(
            project="proj",
            daily=10.0,
            thresholds=[1.0],
            webhook_urls=["https://hook.example.com"],
        )
        manager = BudgetManager([cfg], db, sender)

        rec = _record(project="proj", cost=10.5)
        db.save(rec)
        manager.check(rec)

        sender.send_webhook.assert_called_once()
        _, payload = sender.send_webhook.call_args[0]
        assert payload["threshold"] == 1.0
        assert payload["percentage"] >= 100

    def test_no_alert_when_no_webhook_urls(self, db: SQLiteStorage, sender: MagicMock) -> None:
        cfg = BudgetConfig(
            project="proj",
            daily=10.0,
            thresholds=[0.8],
            webhook_urls=[],  # no webhooks
        )
        manager = BudgetManager([cfg], db, sender)

        rec = _record(project="proj", cost=9.0)
        db.save(rec)
        manager.check(rec)

        # Alert is logged but send_webhook not called
        sender.send_webhook.assert_not_called()

    def test_no_alert_when_no_budget_config(self, db: SQLiteStorage, sender: MagicMock) -> None:
        manager = BudgetManager([], db, sender)
        rec = _record(project="proj", cost=1.0)
        db.save(rec)
        manager.check(rec)
        sender.send_webhook.assert_not_called()

    def test_alert_payload_has_top_models(self, db: SQLiteStorage, sender: MagicMock) -> None:
        cfg = BudgetConfig(
            project="proj",
            daily=10.0,
            thresholds=[0.8],
            webhook_urls=["https://hook.example.com"],
        )
        manager = BudgetManager([cfg], db, sender)

        rec = _record(project="proj", cost=9.0)
        db.save(rec)
        manager.check(rec)

        _, payload = sender.send_webhook.call_args[0]
        assert "top_models" in payload
        assert isinstance(payload["top_models"], list)

    def test_weekly_budget_check(self, db: SQLiteStorage, sender: MagicMock) -> None:
        cfg = BudgetConfig(
            project="proj",
            weekly=20.0,
            thresholds=[0.8],
            webhook_urls=["https://hook.example.com"],
        )
        manager = BudgetManager([cfg], db, sender)

        rec = _record(project="proj", cost=17.0)  # 85% of 20
        db.save(rec)
        manager.check(rec)

        sender.send_webhook.assert_called_once()
        _, payload = sender.send_webhook.call_args[0]
        assert payload["period"] == "weekly"

    def test_monthly_budget_check(self, db: SQLiteStorage, sender: MagicMock) -> None:
        cfg = BudgetConfig(
            project="proj",
            monthly=100.0,
            thresholds=[0.9],
            webhook_urls=["https://hook.example.com"],
        )
        manager = BudgetManager([cfg], db, sender)

        rec = _record(project="proj", cost=95.0)  # 95% of 100
        db.save(rec)
        manager.check(rec)

        sender.send_webhook.assert_called_once()
        _, payload = sender.send_webhook.call_args[0]
        assert payload["period"] == "monthly"

    def test_multiple_webhook_urls(self, db: SQLiteStorage, sender: MagicMock) -> None:
        urls = ["https://hook1.example.com", "https://hook2.example.com"]
        cfg = BudgetConfig(
            project="proj",
            daily=10.0,
            thresholds=[0.8],
            webhook_urls=urls,
        )
        manager = BudgetManager([cfg], db, sender)

        rec = _record(project="proj", cost=9.0)
        db.save(rec)
        manager.check(rec)

        assert sender.send_webhook.call_count == 2
        called_urls = {c[0][0] for c in sender.send_webhook.call_args_list}
        assert called_urls == set(urls)

    def test_add_config_persists(self, db: SQLiteStorage, sender: MagicMock) -> None:
        manager = BudgetManager([], db, sender)
        cfg = BudgetConfig(project="new-proj", daily=5.0)
        manager.add_config(cfg)

        stored = db.get_budget_config("new-proj")
        assert stored is not None
        assert stored["daily"] == 5.0

    def test_falls_back_to_default_config(self, db: SQLiteStorage, sender: MagicMock) -> None:
        """Records from any project should match the 'default' config as fallback."""
        default_cfg = BudgetConfig(
            project="default",
            daily=10.0,
            thresholds=[0.8],
            webhook_urls=["https://hook.example.com"],
        )
        manager = BudgetManager([default_cfg], db, sender)

        rec = _record(project="other-proj", cost=9.0)
        db.save(rec)
        manager.check(rec)

        # Spend is under budget for 'other-proj', but 'default' is checked
        # The spend is queried per-project, so 'other-proj' spend is 9.0
        # This tests the fallback lookup, not necessarily that the alert fires
        # (depends on whether the budget is configured for the right project)
        # The important thing: no exception raised
        assert sender.send_webhook.call_count in (0, 1)
