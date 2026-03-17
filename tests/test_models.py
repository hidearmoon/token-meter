"""Tests for UsageRecord data model."""
import json
from datetime import datetime, timezone

import pytest
from token_meter.models import UsageRecord


def make_record(**kwargs) -> UsageRecord:
    defaults = dict(
        provider="openai",
        model="gpt-4o",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        input_cost=0.00025,
        output_cost=0.0005,
        total_cost=0.00075,
        latency_ms=250.0,
    )
    defaults.update(kwargs)
    return UsageRecord(**defaults)


class TestUsageRecord:
    def test_default_id_is_uuid(self):
        r = make_record()
        import re
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            r.id,
            re.I,
        )

    def test_default_timestamp_is_utc(self):
        r = make_record()
        assert r.timestamp.tzinfo is not None

    def test_default_project(self):
        r = make_record()
        assert r.project == "default"

    def test_custom_project(self):
        r = make_record(project="my-app")
        assert r.project == "my-app"

    def test_is_stream_default_false(self):
        r = make_record()
        assert r.is_stream is False

    def test_metadata_optional(self):
        r = make_record()
        assert r.metadata is None

    def test_to_dict_keys(self):
        r = make_record()
        d = r.to_dict()
        expected_keys = {
            "id", "timestamp", "provider", "model",
            "input_tokens", "output_tokens", "total_tokens",
            "input_cost", "output_cost", "total_cost",
            "latency_ms", "project", "is_stream", "metadata",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        r = make_record(model="claude-sonnet-4", project="test")
        d = r.to_dict()
        assert d["model"] == "claude-sonnet-4"
        assert d["project"] == "test"
        assert d["provider"] == "openai"

    def test_to_row_length(self):
        r = make_record()
        row = r.to_row()
        assert len(row) == 14

    def test_roundtrip_row(self):
        ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        original = make_record(
            timestamp=ts,
            metadata={"session": "abc123"},
            is_stream=True,
        )
        row = original.to_row()
        restored = UsageRecord.from_row(row)

        assert restored.id == original.id
        assert restored.provider == original.provider
        assert restored.model == original.model
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.total_tokens == original.total_tokens
        assert restored.input_cost == pytest.approx(original.input_cost, rel=1e-6)
        assert restored.output_cost == pytest.approx(original.output_cost, rel=1e-6)
        assert restored.total_cost == pytest.approx(original.total_cost, rel=1e-6)
        assert restored.latency_ms == pytest.approx(original.latency_ms, rel=1e-4)
        assert restored.project == original.project
        assert restored.is_stream == original.is_stream
        assert restored.metadata == original.metadata

    def test_metadata_json_serialisation(self):
        meta = {"key": "value", "count": 42, "nested": {"x": True}}
        r = make_record(metadata=meta)
        row = r.to_row()
        # metadata is the last element (index 13)
        assert json.loads(row[13]) == meta
        restored = UsageRecord.from_row(row)
        assert restored.metadata == meta

    def test_no_metadata_row_is_none(self):
        r = make_record()
        row = r.to_row()
        assert row[13] is None

    def test_cost_rounded_to_6_decimals(self):
        r = make_record(input_cost=0.123456789, output_cost=0.987654321, total_cost=1.111111110)
        row = r.to_row()
        # input_cost is index 7
        assert row[7] == round(0.123456789, 6)
