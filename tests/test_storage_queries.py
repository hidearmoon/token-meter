"""Tests for the CLI-oriented query methods added to SQLiteStorage.

Covers: get_records, get_total, get_projects, get_models, get_summary.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from token_meter.models import UsageRecord
from token_meter.storage.sqlite import SQLiteStorage


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture()
def db(tmp_path: Path) -> SQLiteStorage:
    storage = SQLiteStorage(tmp_path / "test.db")
    yield storage
    storage.close()


def _rec(
    provider: str = "openai",
    model: str = "gpt-4o",
    project: str = "default",
    total_cost: float = 0.01,
    input_tokens: int = 100,
    output_tokens: int = 50,
    latency_ms: float = 200.0,
    ts: datetime | None = None,
) -> UsageRecord:
    if ts is None:
        ts = datetime.now(timezone.utc)
    total_tokens = input_tokens + output_tokens
    return UsageRecord(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_cost=total_cost * 0.6,
        output_cost=total_cost * 0.4,
        total_cost=total_cost,
        latency_ms=latency_ms,
        project=project,
        is_stream=False,
        timestamp=ts,
    )


# ------------------------------------------------------------------ #
# get_records                                                          #
# ------------------------------------------------------------------ #

class TestGetRecords:
    def test_returns_empty_list_on_no_data(self, db: SQLiteStorage) -> None:
        assert db.get_records() == []

    def test_returns_records_newest_first(self, db: SQLiteStorage) -> None:
        older = _rec(ts=datetime(2024, 1, 1, tzinfo=timezone.utc))
        newer = _rec(ts=datetime(2024, 6, 1, tzinfo=timezone.utc))
        db.save(older)
        db.save(newer)
        records = db.get_records()
        assert records[0].timestamp >= records[1].timestamp

    def test_limit(self, db: SQLiteStorage) -> None:
        for _ in range(10):
            db.save(_rec())
        assert len(db.get_records(limit=3)) == 3

    def test_default_limit_is_20(self, db: SQLiteStorage) -> None:
        for _ in range(25):
            db.save(_rec())
        assert len(db.get_records()) == 20

    def test_filter_provider(self, db: SQLiteStorage) -> None:
        db.save(_rec(provider="openai"))
        db.save(_rec(provider="anthropic"))
        results = db.get_records(provider="openai")
        assert all(r.provider == "openai" for r in results)

    def test_filter_model(self, db: SQLiteStorage) -> None:
        db.save(_rec(model="gpt-4o"))
        db.save(_rec(model="gpt-4o-mini"))
        results = db.get_records(model="gpt-4o")
        assert all(r.model == "gpt-4o" for r in results)

    def test_filter_project(self, db: SQLiteStorage) -> None:
        db.save(_rec(project="proj-a"))
        db.save(_rec(project="proj-b"))
        results = db.get_records(project="proj-a")
        assert all(r.project == "proj-a" for r in results)

    def test_filter_date_range(self, db: SQLiteStorage) -> None:
        jan = _rec(ts=datetime(2024, 1, 15, tzinfo=timezone.utc))
        jun = _rec(ts=datetime(2024, 6, 15, tzinfo=timezone.utc))
        db.save(jan)
        db.save(jun)
        start = datetime(2024, 6, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 30, tzinfo=timezone.utc)
        results = db.get_records(start=start, end=end)
        assert len(results) == 1
        assert results[0].timestamp.month == 6


# ------------------------------------------------------------------ #
# get_total                                                            #
# ------------------------------------------------------------------ #

class TestGetTotal:
    def test_empty_returns_zero(self, db: SQLiteStorage) -> None:
        result = db.get_total()
        assert result["call_count"] == 0
        assert result["total_cost"] == 0.0

    def test_sums_correctly(self, db: SQLiteStorage) -> None:
        db.save(_rec(total_cost=0.01))
        db.save(_rec(total_cost=0.02))
        result = db.get_total()
        assert result["call_count"] == 2
        assert abs(result["total_cost"] - 0.03) < 1e-9

    def test_project_filter(self, db: SQLiteStorage) -> None:
        db.save(_rec(project="a", total_cost=0.05))
        db.save(_rec(project="b", total_cost=0.10))
        result = db.get_total(project="a")
        assert result["call_count"] == 1
        assert abs(result["total_cost"] - 0.05) < 1e-9

    def test_date_range_filter(self, db: SQLiteStorage) -> None:
        db.save(_rec(ts=datetime(2024, 1, 1, tzinfo=timezone.utc), total_cost=1.0))
        db.save(_rec(ts=datetime(2024, 12, 31, tzinfo=timezone.utc), total_cost=2.0))
        start = datetime(2024, 12, 1, tzinfo=timezone.utc)
        result = db.get_total(start=start)
        assert result["call_count"] == 1
        assert abs(result["total_cost"] - 2.0) < 1e-9


# ------------------------------------------------------------------ #
# get_projects                                                         #
# ------------------------------------------------------------------ #

class TestGetProjects:
    def test_empty(self, db: SQLiteStorage) -> None:
        assert db.get_projects() == []

    def test_returns_all_projects(self, db: SQLiteStorage) -> None:
        db.save(_rec(project="alpha"))
        db.save(_rec(project="beta"))
        projects = db.get_projects()
        names = {p["project"] for p in projects}
        assert "alpha" in names
        assert "beta" in names

    def test_aggregation_correctness(self, db: SQLiteStorage) -> None:
        db.save(_rec(project="x", total_cost=0.10, input_tokens=100, output_tokens=50))
        db.save(_rec(project="x", total_cost=0.20, input_tokens=200, output_tokens=100))
        projects = db.get_projects()
        assert len(projects) == 1
        p = projects[0]
        assert p["project"] == "x"
        assert p["call_count"] == 2
        assert abs(p["total_cost"] - 0.30) < 1e-9
        assert p["total_tokens"] == (150 + 300)

    def test_ordered_by_cost_desc(self, db: SQLiteStorage) -> None:
        db.save(_rec(project="cheap", total_cost=0.001))
        db.save(_rec(project="expensive", total_cost=1.0))
        projects = db.get_projects()
        assert projects[0]["project"] == "expensive"

    def test_has_first_and_last_call(self, db: SQLiteStorage) -> None:
        db.save(_rec(project="p"))
        projects = db.get_projects()
        assert projects[0]["first_call"] is not None
        assert projects[0]["last_call"] is not None


# ------------------------------------------------------------------ #
# get_models                                                           #
# ------------------------------------------------------------------ #

class TestGetModels:
    def test_empty(self, db: SQLiteStorage) -> None:
        assert db.get_models() == []

    def test_returns_all_models(self, db: SQLiteStorage) -> None:
        db.save(_rec(provider="openai", model="gpt-4o"))
        db.save(_rec(provider="anthropic", model="claude-sonnet-4"))
        models = db.get_models()
        keys = {(m["provider"], m["model"]) for m in models}
        assert ("openai", "gpt-4o") in keys
        assert ("anthropic", "claude-sonnet-4") in keys

    def test_ordered_by_cost_desc(self, db: SQLiteStorage) -> None:
        db.save(_rec(model="cheap-model", total_cost=0.001))
        db.save(_rec(model="pricey-model", total_cost=5.0))
        models = db.get_models()
        assert models[0]["model"] == "pricey-model"

    def test_aggregation_per_model(self, db: SQLiteStorage) -> None:
        db.save(_rec(model="gpt-4o", total_cost=0.01, input_tokens=100, output_tokens=50))
        db.save(_rec(model="gpt-4o", total_cost=0.02, input_tokens=200, output_tokens=100))
        models = db.get_models()
        assert len(models) == 1
        m = models[0]
        assert m["call_count"] == 2
        assert abs(m["total_cost"] - 0.03) < 1e-9
        assert m["total_input_tokens"] == 300
        assert m["total_output_tokens"] == 150

    def test_avg_latency_populated(self, db: SQLiteStorage) -> None:
        db.save(_rec(latency_ms=100.0))
        db.save(_rec(latency_ms=300.0))
        models = db.get_models()
        assert abs(models[0]["avg_latency_ms"] - 200.0) < 1.0


# ------------------------------------------------------------------ #
# get_summary                                                          #
# ------------------------------------------------------------------ #

class TestGetSummary:
    def test_empty(self, db: SQLiteStorage) -> None:
        assert db.get_summary() == []

    def test_group_by_model(self, db: SQLiteStorage) -> None:
        db.save(_rec(provider="openai", model="gpt-4o"))
        db.save(_rec(provider="anthropic", model="claude-sonnet-4"))
        rows = db.get_summary(group_by="model")
        assert len(rows) == 2
        providers = {r["provider"] for r in rows}
        assert "openai" in providers

    def test_group_by_provider(self, db: SQLiteStorage) -> None:
        db.save(_rec(provider="openai", model="gpt-4o"))
        db.save(_rec(provider="openai", model="gpt-4o-mini"))
        db.save(_rec(provider="anthropic", model="claude-sonnet-4"))
        rows = db.get_summary(group_by="provider")
        providers = [r["provider"] for r in rows]
        assert "openai" in providers
        assert "anthropic" in providers
        # openai should appear once (grouped)
        assert providers.count("openai") == 1

    def test_group_by_project(self, db: SQLiteStorage) -> None:
        db.save(_rec(project="alpha"))
        db.save(_rec(project="beta"))
        rows = db.get_summary(group_by="project")
        groups = {r["group"] for r in rows}
        assert "alpha" in groups
        assert "beta" in groups

    def test_group_by_day(self, db: SQLiteStorage) -> None:
        db.save(_rec(ts=datetime(2024, 3, 1, tzinfo=timezone.utc)))
        db.save(_rec(ts=datetime(2024, 3, 2, tzinfo=timezone.utc)))
        rows = db.get_summary(group_by="day")
        assert len(rows) == 2
        assert rows[0]["group"] is not None

    def test_group_by_week(self, db: SQLiteStorage) -> None:
        db.save(_rec(ts=datetime(2024, 1, 1, tzinfo=timezone.utc)))
        db.save(_rec(ts=datetime(2024, 1, 8, tzinfo=timezone.utc)))
        rows = db.get_summary(group_by="week")
        assert len(rows) == 2

    def test_group_by_month(self, db: SQLiteStorage) -> None:
        db.save(_rec(ts=datetime(2024, 1, 15, tzinfo=timezone.utc)))
        db.save(_rec(ts=datetime(2024, 2, 15, tzinfo=timezone.utc)))
        rows = db.get_summary(group_by="month")
        assert len(rows) == 2

    def test_date_range_filter(self, db: SQLiteStorage) -> None:
        db.save(_rec(ts=datetime(2024, 1, 15, tzinfo=timezone.utc), total_cost=1.0))
        db.save(_rec(ts=datetime(2024, 6, 15, tzinfo=timezone.utc), total_cost=2.0))
        start = datetime(2024, 6, 1, tzinfo=timezone.utc)
        rows = db.get_summary(start=start, group_by="model")
        assert len(rows) == 1
        assert abs(rows[0]["total_cost"] - 2.0) < 1e-9

    def test_project_filter(self, db: SQLiteStorage) -> None:
        db.save(_rec(project="a", total_cost=0.1))
        db.save(_rec(project="b", total_cost=0.5))
        rows = db.get_summary(project="a", group_by="model")
        assert all(row["total_cost"] < 0.5 for row in rows)

    def test_invalid_group_by_falls_back_to_model(self, db: SQLiteStorage) -> None:
        db.save(_rec())
        rows = db.get_summary(group_by="invalid_dimension")
        assert len(rows) == 1
        assert rows[0]["provider"] is not None

    def test_ordered_by_cost_desc(self, db: SQLiteStorage) -> None:
        db.save(_rec(model="cheap", provider="openai", total_cost=0.001))
        db.save(_rec(model="expensive", provider="openai", total_cost=9.99))
        rows = db.get_summary(group_by="model")
        assert rows[0]["total_cost"] >= rows[1]["total_cost"]

    def test_row_contains_expected_keys(self, db: SQLiteStorage) -> None:
        db.save(_rec())
        rows = db.get_summary(group_by="model")
        assert len(rows) == 1
        expected = {
            "provider", "model", "group",
            "call_count", "total_input_tokens", "total_output_tokens",
            "total_tokens", "total_cost", "avg_latency_ms",
        }
        assert expected == set(rows[0].keys())
