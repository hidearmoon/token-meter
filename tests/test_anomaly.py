"""Tests for AnomalyDetector — Z-score based cost spike detection."""
from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from token_meter.alerts import AlertSender
from token_meter.anomaly import AnomalyDetector, _MIN_HISTORY_DAYS, _WINDOW_DAYS
from token_meter.models import UsageRecord
from token_meter.storage.sqlite import SQLiteStorage


# ------------------------------------------------------------------ #
# Fixtures / helpers                                                   #
# ------------------------------------------------------------------ #

@pytest.fixture
def db(tmp_path: Path) -> SQLiteStorage:
    s = SQLiteStorage(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def sender() -> MagicMock:
    return MagicMock(spec=AlertSender)


def _record(
    project: str = "proj",
    model: str = "gpt-4o",
    cost: float = 1.0,
    ts: datetime | None = None,
) -> UsageRecord:
    if ts is None:
        ts = datetime.now(timezone.utc)
    return UsageRecord(
        provider="openai",
        model=model,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        input_cost=cost * 0.6,
        output_cost=cost * 0.4,
        total_cost=cost,
        latency_ms=200.0,
        project=project,
        is_stream=False,
        timestamp=ts,
    )


def _seed_history(
    db: SQLiteStorage,
    project: str,
    model: str,
    costs: list[float],
    end_date: date,
) -> None:
    """Insert one record per day going back from end_date (inclusive)."""
    for i, cost in enumerate(reversed(costs)):
        day = end_date - timedelta(days=i)
        ts = datetime(day.year, day.month, day.day, 12, 0, 0, tzinfo=timezone.utc)
        db.save(_record(project=project, model=model, cost=cost, ts=ts))


# ------------------------------------------------------------------ #
# Insufficient history — skip detection                                #
# ------------------------------------------------------------------ #

class TestInsufficientHistory:
    def test_fewer_than_min_history_days_skips(self, db: SQLiteStorage, sender: MagicMock) -> None:
        detector = AnomalyDetector(db, sender)
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()

        # Only 3 days of history (need _MIN_HISTORY_DAYS = 7)
        _seed_history(db, "proj", "gpt-4o", [1.0, 1.0, 1.0], yesterday - timedelta(days=1))

        results = detector.check_yesterday(project="proj")
        assert results == []
        sender.send_webhook.assert_not_called()

    def test_exactly_min_history_days_runs(self, db: SQLiteStorage, sender: MagicMock) -> None:
        """With exactly _MIN_HISTORY_DAYS history we should run (not necessarily find anomaly)."""
        detector = AnomalyDetector(db, sender)
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)

        # Seed _MIN_HISTORY_DAYS identical daily costs in history window (before target day)
        normal_costs = [1.0] * _MIN_HISTORY_DAYS
        _seed_history(db, "proj", "gpt-4o", normal_costs, yesterday - timedelta(days=1))

        # All costs are identical → stdev = 0 → detection skips (no division by zero)
        results = detector.check_yesterday(project="proj")
        assert isinstance(results, list)  # just no crash


# ------------------------------------------------------------------ #
# Normal data — no anomaly                                             #
# ------------------------------------------------------------------ #

class TestNoAnomaly:
    def test_stable_costs_no_anomaly(self, db: SQLiteStorage, sender: MagicMock) -> None:
        detector = AnomalyDetector(db, sender, z_threshold=2.0)
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)

        # 20 days of stable $1.00/day history
        normal_costs = [1.0] * 20
        _seed_history(db, "proj", "gpt-4o", normal_costs, yesterday - timedelta(days=1))

        # Yesterday's cost is also $1.00 — well within normal range
        ts = datetime(yesterday.year, yesterday.month, yesterday.day, 12, tzinfo=timezone.utc)
        db.save(_record(project="proj", model="gpt-4o", cost=1.0, ts=ts))

        results = detector.check_yesterday(project="proj")
        assert results == []
        sender.send_webhook.assert_not_called()

    def test_zero_std_skipped(self, db: SQLiteStorage, sender: MagicMock) -> None:
        """When all historical costs are identical, std=0 → skip detection."""
        detector = AnomalyDetector(db, sender, z_threshold=2.0)
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)

        # 10 days all same cost → std = 0
        _seed_history(db, "proj", "gpt-4o", [2.0] * 10, yesterday - timedelta(days=1))
        ts = datetime(yesterday.year, yesterday.month, yesterday.day, 12, tzinfo=timezone.utc)
        db.save(_record(project="proj", model="gpt-4o", cost=100.0, ts=ts))

        results = detector.check_yesterday(project="proj")
        assert results == []


# ------------------------------------------------------------------ #
# Anomaly detection                                                    #
# ------------------------------------------------------------------ #

class TestAnomalyDetected:
    def test_spike_detected_and_saved(self, db: SQLiteStorage, sender: MagicMock) -> None:
        detector = AnomalyDetector(db, sender, z_threshold=2.0, webhook_urls=[])
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)

        # Build 20-day history with mean ~$1 and low std
        normal_costs = [1.0 + (i % 3) * 0.05 for i in range(20)]  # ~$1.00–$1.10
        _seed_history(db, "proj", "gpt-4o", normal_costs, yesterday - timedelta(days=1))

        mean = statistics.mean(normal_costs)
        std = statistics.stdev(normal_costs)
        spike = mean + 5 * std  # clearly anomalous

        ts = datetime(yesterday.year, yesterday.month, yesterday.day, 12, tzinfo=timezone.utc)
        db.save(_record(project="proj", model="gpt-4o", cost=spike, ts=ts))

        results = detector.check_yesterday(project="proj")
        assert len(results) == 1
        a = results[0]
        assert a["project"] == "proj"
        assert a["model"] == "gpt-4o"
        assert a["z_score"] >= 2.0

        # Must be persisted
        stored = db.get_anomalies(project="proj")
        assert len(stored) == 1
        assert stored[0]["z_score"] == a["z_score"]

    def test_spike_sends_webhook(self, db: SQLiteStorage, sender: MagicMock) -> None:
        urls = ["https://hook.example.com"]
        detector = AnomalyDetector(db, sender, z_threshold=2.0, webhook_urls=urls)
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)

        normal_costs = [1.0 + (i % 3) * 0.05 for i in range(20)]
        _seed_history(db, "proj", "gpt-4o", normal_costs, yesterday - timedelta(days=1))

        mean = statistics.mean(normal_costs)
        std = statistics.stdev(normal_costs)
        spike = mean + 5 * std

        ts = datetime(yesterday.year, yesterday.month, yesterday.day, 12, tzinfo=timezone.utc)
        db.save(_record(project="proj", model="gpt-4o", cost=spike, ts=ts))

        detector.check_yesterday(project="proj")

        sender.send_webhook.assert_called_once()
        url_arg, payload = sender.send_webhook.call_args[0]
        assert url_arg == urls[0]
        assert payload["alert"] == "cost_anomaly"
        assert "z_score" in payload
        assert payload["project"] == "proj"

    def test_anomaly_payload_fields(self, db: SQLiteStorage, sender: MagicMock) -> None:
        urls = ["https://hook.example.com"]
        detector = AnomalyDetector(db, sender, z_threshold=2.0, webhook_urls=urls)
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)

        normal_costs = [1.0 + (i % 3) * 0.05 for i in range(20)]
        _seed_history(db, "proj", "gpt-4o", normal_costs, yesterday - timedelta(days=1))
        mean = statistics.mean(normal_costs)
        std = statistics.stdev(normal_costs)
        spike = mean + 5 * std

        ts = datetime(yesterday.year, yesterday.month, yesterday.day, 12, tzinfo=timezone.utc)
        db.save(_record(project="proj", model="gpt-4o", cost=spike, ts=ts))

        detector.check_yesterday(project="proj")

        _, payload = sender.send_webhook.call_args[0]
        required_fields = {"alert", "project", "model", "date", "daily_cost",
                           "rolling_avg", "rolling_std", "z_score", "timestamp"}
        assert required_fields.issubset(payload.keys())

    def test_custom_z_threshold(self, db: SQLiteStorage, sender: MagicMock) -> None:
        """With z_threshold=5.0 a moderate spike should not fire."""
        detector = AnomalyDetector(db, sender, z_threshold=5.0, webhook_urls=["https://h.example.com"])
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)

        normal_costs = [1.0 + (i % 3) * 0.05 for i in range(20)]
        _seed_history(db, "proj", "gpt-4o", normal_costs, yesterday - timedelta(days=1))
        mean = statistics.mean(normal_costs)
        std = statistics.stdev(normal_costs)

        # Moderate spike: z ≈ 3, below threshold of 5
        moderate_spike = mean + 3 * std
        ts = datetime(yesterday.year, yesterday.month, yesterday.day, 12, tzinfo=timezone.utc)
        db.save(_record(project="proj", model="gpt-4o", cost=moderate_spike, ts=ts))

        results = detector.check_yesterday(project="proj")
        assert results == []
        sender.send_webhook.assert_not_called()

    def test_check_date_arbitrary(self, db: SQLiteStorage, sender: MagicMock) -> None:
        """check_date should work for an arbitrary historical date."""
        detector = AnomalyDetector(db, sender, z_threshold=2.0, webhook_urls=[])
        target = date(2024, 1, 20)

        normal_costs = [1.0 + (i % 3) * 0.05 for i in range(20)]
        history_end = target - timedelta(days=1)
        _seed_history(db, "proj", "gpt-4o", normal_costs, history_end)

        mean = statistics.mean(normal_costs)
        std = statistics.stdev(normal_costs)
        spike = mean + 5 * std
        ts = datetime(target.year, target.month, target.day, 12, tzinfo=timezone.utc)
        db.save(_record(project="proj", model="gpt-4o", cost=spike, ts=ts))

        results = detector.check_date(target, project="proj")
        assert len(results) == 1
        assert results[0]["date"] == target.isoformat()

    def test_multiple_models_detected_independently(self, db: SQLiteStorage, sender: MagicMock) -> None:
        detector = AnomalyDetector(db, sender, z_threshold=2.0, webhook_urls=[])
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)

        for model in ("gpt-4o", "gpt-3.5-turbo"):
            normal_costs = [1.0 + (i % 3) * 0.05 for i in range(20)]
            _seed_history(db, "proj", model, normal_costs, yesterday - timedelta(days=1))
            mean = statistics.mean(normal_costs)
            std = statistics.stdev(normal_costs)
            spike = mean + 5 * std
            ts = datetime(yesterday.year, yesterday.month, yesterday.day, 12, tzinfo=timezone.utc)
            db.save(_record(project="proj", model=model, cost=spike, ts=ts))

        results = detector.check_yesterday(project="proj")
        assert len(results) == 2
        detected_models = {r["model"] for r in results}
        assert detected_models == {"gpt-4o", "gpt-3.5-turbo"}

    def test_no_anomaly_when_no_target_data(self, db: SQLiteStorage, sender: MagicMock) -> None:
        """If there's no data for the target date, skip gracefully."""
        detector = AnomalyDetector(db, sender, z_threshold=2.0, webhook_urls=[])
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)

        normal_costs = [1.0] * 20
        _seed_history(db, "proj", "gpt-4o", normal_costs, yesterday - timedelta(days=2))
        # No record for yesterday

        results = detector.check_yesterday(project="proj")
        assert results == []


# ------------------------------------------------------------------ #
# check_on_first_write hook                                            #
# ------------------------------------------------------------------ #

class TestCheckOnFirstWriteHook:
    def test_runs_only_once_per_day(self, db: SQLiteStorage, sender: MagicMock) -> None:
        """Subsequent writes on the same day should not re-run detection."""
        detector = AnomalyDetector(db, sender, z_threshold=2.0)
        run_count = [0]

        original_check_yesterday = detector.check_yesterday

        def counting_check(**kwargs):
            run_count[0] += 1
            return []

        detector.check_yesterday = counting_check  # type: ignore[method-assign]

        today = datetime.now(timezone.utc).date()

        rec1 = _record()
        rec2 = _record()
        rec3 = _record()

        db.save(rec1)
        detector.check_on_first_write(rec1)
        db.save(rec2)
        detector.check_on_first_write(rec2)
        db.save(rec3)
        detector.check_on_first_write(rec3)

        assert run_count[0] == 1, "Detection should run only once per day"

    def test_swallows_exceptions(self, db: SQLiteStorage, sender: MagicMock) -> None:
        """Exceptions in anomaly detection must not propagate to caller."""
        detector = AnomalyDetector(db, sender)

        def exploding_check(**kwargs):
            raise RuntimeError("boom")

        detector.check_yesterday = exploding_check  # type: ignore[method-assign]
        rec = _record()
        db.save(rec)
        # Should not raise
        detector.check_on_first_write(rec)

    def test_reruns_on_new_day(self, db: SQLiteStorage, sender: MagicMock) -> None:
        """After a day boundary, detection should run again."""
        detector = AnomalyDetector(db, sender, z_threshold=2.0)
        run_count = [0]

        def counting_check(**kwargs):
            run_count[0] += 1
            return []

        detector.check_yesterday = counting_check  # type: ignore[method-assign]

        rec = _record()
        db.save(rec)

        # Simulate first write today
        detector._last_check_date = date(2024, 1, 1)  # force "old" date
        detector.check_on_first_write(rec)
        assert run_count[0] == 1

        # Simulate another write on the same "new" day — should not re-run
        detector.check_on_first_write(rec)
        assert run_count[0] == 1

        # Simulate a new day
        detector._last_check_date = date(2024, 1, 2)  # different day
        detector.check_on_first_write(rec)
        assert run_count[0] == 2
