"""Tests for SQLite storage backend."""
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
from token_meter.models import UsageRecord
from token_meter.storage.sqlite import SQLiteStorage


def _make_record(
    provider="openai",
    model="gpt-4o",
    input_tokens=100,
    output_tokens=50,
    project="default",
    total_cost=0.001,
    latency_ms=200.0,
    timestamp: datetime | None = None,
    metadata=None,
) -> UsageRecord:
    ts = timestamp or datetime.now(timezone.utc)
    return UsageRecord(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_cost=total_cost * 0.3,
        output_cost=total_cost * 0.7,
        total_cost=total_cost,
        latency_ms=latency_ms,
        project=project,
        timestamp=ts,
        metadata=metadata,
    )


@pytest.fixture
def db(tmp_path):
    storage = SQLiteStorage(tmp_path / "test.db")
    yield storage
    storage.close()


class TestSQLiteStorage:
    def test_save_and_retrieve(self, db):
        rec = _make_record()
        db.save(rec)
        results = db.query()
        assert len(results) == 1
        assert results[0].id == rec.id

    def test_save_multiple(self, db):
        for i in range(5):
            db.save(_make_record(input_tokens=100 * (i + 1)))
        results = db.query()
        assert len(results) == 5

    def test_query_by_project(self, db):
        db.save(_make_record(project="proj-a"))
        db.save(_make_record(project="proj-b"))
        db.save(_make_record(project="proj-a"))
        results = db.query(project="proj-a")
        assert len(results) == 2
        assert all(r.project == "proj-a" for r in results)

    def test_query_by_provider(self, db):
        db.save(_make_record(provider="openai"))
        db.save(_make_record(provider="anthropic"))
        results = db.query(provider="openai")
        assert len(results) == 1
        assert results[0].provider == "openai"

    def test_query_by_model(self, db):
        db.save(_make_record(model="gpt-4o"))
        db.save(_make_record(model="claude-sonnet-4"))
        results = db.query(model="gpt-4o")
        assert len(results) == 1
        assert results[0].model == "gpt-4o"

    def test_query_by_time_range(self, db):
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=2)
        future = now + timedelta(hours=2)
        db.save(_make_record(timestamp=past))
        db.save(_make_record(timestamp=now))
        db.save(_make_record(timestamp=future))
        results = db.query(start=now - timedelta(minutes=1), end=now + timedelta(minutes=1))
        assert len(results) == 1

    def test_query_limit(self, db):
        for _ in range(10):
            db.save(_make_record())
        results = db.query(limit=3)
        assert len(results) == 3

    def test_query_newest_first(self, db):
        now = datetime.now(timezone.utc)
        db.save(_make_record(timestamp=now - timedelta(minutes=2)))
        db.save(_make_record(timestamp=now - timedelta(minutes=1)))
        db.save(_make_record(timestamp=now))
        results = db.query()
        assert results[0].timestamp >= results[1].timestamp >= results[2].timestamp

    def test_aggregate_empty(self, db):
        stats = db.aggregate()
        assert stats["call_count"] == 0
        assert stats["total_cost"] == 0.0

    def test_aggregate_with_records(self, db):
        db.save(_make_record(input_tokens=100, output_tokens=50, total_cost=0.001))
        db.save(_make_record(input_tokens=200, output_tokens=100, total_cost=0.002))
        stats = db.aggregate()
        assert stats["call_count"] == 2
        assert stats["total_tokens"] == 450
        assert stats["total_cost"] == pytest.approx(0.003, rel=1e-5)

    def test_aggregate_by_project(self, db):
        db.save(_make_record(project="proj-a", total_cost=0.005))
        db.save(_make_record(project="proj-b", total_cost=0.010))
        stats = db.aggregate(project="proj-a")
        assert stats["call_count"] == 1
        assert stats["total_cost"] == pytest.approx(0.005, rel=1e-5)

    def test_aggregate_by_model(self, db):
        db.save(_make_record(provider="openai", model="gpt-4o", total_cost=0.01))
        db.save(_make_record(provider="openai", model="gpt-4o", total_cost=0.02))
        db.save(_make_record(provider="anthropic", model="claude-sonnet-4", total_cost=0.005))
        rows = db.aggregate_by_model()
        assert len(rows) == 2
        # gpt-4o has highest cost → first
        assert rows[0]["model"] == "gpt-4o"
        assert rows[0]["call_count"] == 2
        assert rows[0]["total_cost"] == pytest.approx(0.03, rel=1e-5)

    def test_record_roundtrip_via_storage(self, db):
        original = _make_record(
            provider="anthropic",
            model="claude-opus-4",
            metadata={"session": "xyz"},
        )
        original.is_stream = True
        db.save(original)
        results = db.query()
        assert len(results) == 1
        r = results[0]
        assert r.provider == "anthropic"
        assert r.model == "claude-opus-4"
        assert r.is_stream is True
        assert r.metadata == {"session": "xyz"}

    def test_db_file_created(self, tmp_path):
        db_path = tmp_path / "sub" / "deep" / "test.db"
        storage = SQLiteStorage(db_path)
        storage.save(_make_record())
        assert db_path.exists()
        storage.close()

    def test_concurrent_writes_thread_safe(self, db):
        """Multiple threads must not corrupt the database."""
        import threading

        errors = []

        def write_records():
            try:
                for _ in range(20):
                    db.save(_make_record())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_records) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        stats = db.aggregate()
        assert stats["call_count"] == 100

    def test_query_with_naive_start_datetime(self, db):
        """_ts() must handle naive datetimes (assumes UTC)."""
        from datetime import datetime
        db.save(_make_record())
        # naive datetime (no tzinfo)
        naive_start = datetime.utcnow() - timedelta(hours=1)
        naive_end = datetime.utcnow() + timedelta(hours=1)
        results = db.query(start=naive_start, end=naive_end)
        assert len(results) == 1

    def test_aggregate_by_model_with_time_range(self, db):
        now = datetime.now(timezone.utc)
        db.save(_make_record(provider="openai", model="gpt-4o", total_cost=0.01,
                             timestamp=now))
        db.save(_make_record(provider="anthropic", model="claude-sonnet-4", total_cost=0.005,
                             timestamp=now - timedelta(days=10)))
        # Only query last 1 day → only openai record
        rows = db.aggregate_by_model(start=now - timedelta(days=1))
        assert len(rows) == 1
        assert rows[0]["model"] == "gpt-4o"

    def test_aggregate_by_model_with_project_filter(self, db):
        db.save(_make_record(project="proj-a", model="gpt-4o", total_cost=0.01))
        db.save(_make_record(project="proj-b", model="gemini-2.5-flash", total_cost=0.005))
        rows = db.aggregate_by_model(project="proj-a")
        assert len(rows) == 1
        assert rows[0]["model"] == "gpt-4o"

    def test_aggregate_by_provider(self, db):
        db.save(_make_record(provider="openai", total_cost=0.02))
        db.save(_make_record(provider="anthropic", total_cost=0.01))
        stats = db.aggregate(provider="openai")
        assert stats["call_count"] == 1
        assert stats["total_cost"] == pytest.approx(0.02, rel=1e-5)

    def test_wal_mode_enabled(self, tmp_path):
        """Database should use WAL journal mode."""
        import sqlite3
        db_path = tmp_path / "wal_test.db"
        storage = SQLiteStorage(db_path)
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("PRAGMA journal_mode;").fetchone()
        conn.close()
        storage.close()
        assert row[0] == "wal"
