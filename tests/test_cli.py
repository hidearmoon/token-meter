"""Tests for the TokenMeter CLI (click commands).

Uses click.testing.CliRunner so no real DB or network is required.
Each test uses a pytest tmp_path fixture to get a fresh SQLite file.
"""
from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from token_meter.cli import cli, _parse_date
from token_meter.models import UsageRecord
from token_meter.storage.sqlite import SQLiteStorage


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


def _seed_db(db_path: Path, n: int = 5) -> None:
    """Insert n sample records into the DB."""
    storage = SQLiteStorage(db_path)
    try:
        for i in range(n):
            record = UsageRecord(
                provider="openai" if i % 2 == 0 else "anthropic",
                model="gpt-4o" if i % 2 == 0 else "claude-sonnet-4",
                input_tokens=100 + i * 10,
                output_tokens=50 + i * 5,
                total_tokens=150 + i * 15,
                input_cost=0.001 * (i + 1),
                output_cost=0.0005 * (i + 1),
                total_cost=0.0015 * (i + 1),
                latency_ms=100.0 + i * 20,
                project="test-project",
                is_stream=False,
            )
            storage.save(record)
    finally:
        storage.close()


def _run(db_path: Path, *args) -> "click.testing.Result":
    runner = CliRunner()
    return runner.invoke(cli, ["--db-path", str(db_path)] + list(args), catch_exceptions=False)


# ------------------------------------------------------------------ #
# _parse_date                                                          #
# ------------------------------------------------------------------ #

class TestParseDate:
    def test_today(self) -> None:
        today = datetime.now(timezone.utc).date()
        result = _parse_date("today")
        assert result.year == today.year
        assert result.month == today.month
        assert result.day == today.day
        assert result.tzinfo is not None

    def test_today_end_of_period(self) -> None:
        result = _parse_date("today", end_of_period=True)
        assert result.hour == 23
        assert result.minute == 59

    def test_this_week_start(self) -> None:
        result = _parse_date("this-week")
        assert result.weekday() == 0  # Monday

    def test_this_week_end(self) -> None:
        result = _parse_date("this-week", end_of_period=True)
        assert result.weekday() == 6  # Sunday

    def test_this_month_start(self) -> None:
        today = datetime.now(timezone.utc).date()
        result = _parse_date("this-month")
        assert result.day == 1
        assert result.month == today.month

    def test_this_month_end_of_period(self) -> None:
        result = _parse_date("this-month", end_of_period=True)
        assert result.month != 0

    def test_iso_date(self) -> None:
        result = _parse_date("2024-06-15")
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15
        assert result.tzinfo is not None

    def test_iso_date_end_of_period(self) -> None:
        result = _parse_date("2024-06-15", end_of_period=True)
        assert result.hour == 23
        assert result.second == 59

    def test_invalid_date_raises(self) -> None:
        import click
        with pytest.raises(click.BadParameter):
            _parse_date("not-a-date")


# ------------------------------------------------------------------ #
# dashboard                                                            #
# ------------------------------------------------------------------ #

class TestDashboard:
    def test_empty_db(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(db_path, "dashboard")
        assert result.exit_code == 0
        assert "No data" in result.output

    def test_with_data(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path)
        result = _run(db_path, "dashboard")
        assert result.exit_code == 0
        # Should show panels and model table
        assert "Today" in result.output
        assert "This Week" in result.output
        assert "This Month" in result.output

    def test_output_speed(self, tmp_path: Path) -> None:
        """Dashboard should respond quickly even with data."""
        import time
        db_path = _make_db(tmp_path)
        _seed_db(db_path, n=50)
        start = time.perf_counter()
        result = _run(db_path, "dashboard")
        elapsed = (time.perf_counter() - start) * 1000
        assert result.exit_code == 0
        assert elapsed < 2000  # generous threshold for CI; real target is 100ms


# ------------------------------------------------------------------ #
# summary                                                              #
# ------------------------------------------------------------------ #

class TestSummary:
    def test_empty_db(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(db_path, "summary")
        assert result.exit_code == 0
        assert "No data" in result.output

    def test_group_by_model(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path)
        result = _run(db_path, "summary", "--group-by", "model")
        assert result.exit_code == 0
        assert "gpt-4o" in result.output or "claude" in result.output

    def test_group_by_provider(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path)
        result = _run(db_path, "summary", "--group-by", "provider")
        assert result.exit_code == 0
        assert "openai" in result.output

    def test_group_by_project(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path)
        result = _run(db_path, "summary", "--group-by", "project")
        assert result.exit_code == 0
        # Rich may truncate long strings; check for a reliable prefix
        assert "test-pro" in result.output

    def test_group_by_day(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path)
        result = _run(db_path, "summary", "--group-by", "day")
        assert result.exit_code == 0

    def test_group_by_month(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path)
        result = _run(db_path, "summary", "--group-by", "month")
        assert result.exit_code == 0

    def test_group_by_week(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path)
        result = _run(db_path, "summary", "--group-by", "week")
        assert result.exit_code == 0

    def test_project_filter(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path)
        result = _run(db_path, "summary", "--project", "nonexistent")
        assert result.exit_code == 0
        assert "No data" in result.output

    def test_start_end_filter(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path)
        result = _run(db_path, "summary", "--start", "today", "--end", "today")
        assert result.exit_code == 0

    def test_date_range_no_data(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path)
        result = _run(db_path, "summary", "--start", "2000-01-01", "--end", "2000-01-31")
        assert result.exit_code == 0
        assert "No data" in result.output


# ------------------------------------------------------------------ #
# history                                                              #
# ------------------------------------------------------------------ #

class TestHistory:
    def test_empty_db(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(db_path, "history")
        assert result.exit_code == 0
        assert "No records" in result.output

    def test_shows_records(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path)
        result = _run(db_path, "history")
        assert result.exit_code == 0
        assert "gpt-4o" in result.output or "claude" in result.output

    def test_limit(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path, n=10)
        result = _run(db_path, "history", "--limit", "3")
        assert result.exit_code == 0
        # Table should mention 3 records
        assert "3 records" in result.output

    def test_provider_filter(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path, n=4)
        result = _run(db_path, "history", "--provider", "openai")
        assert result.exit_code == 0
        # anthropic records should not appear
        assert "claude" not in result.output

    def test_model_filter(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path, n=4)
        result = _run(db_path, "history", "--model", "gpt-4o")
        assert result.exit_code == 0
        assert "gpt-4o" in result.output

    def test_project_filter(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path)
        result = _run(db_path, "history", "--project", "nonexistent")
        assert result.exit_code == 0
        assert "No records" in result.output

    def test_start_date_shortcut(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path)
        result = _run(db_path, "history", "--start", "this-month")
        assert result.exit_code == 0


# ------------------------------------------------------------------ #
# export                                                               #
# ------------------------------------------------------------------ #

class TestExport:
    def test_empty_db(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(db_path, "export")
        assert result.exit_code == 0
        assert "No records" in result.output

    def test_csv_stdout(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path, n=3)
        result = _run(db_path, "export", "--format", "csv")
        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        rows = list(reader)
        assert len(rows) == 3
        assert "model" in rows[0]
        assert "total_cost" in rows[0]

    def test_json_stdout(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path, n=3)
        result = _run(db_path, "export", "--format", "json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 3
        assert "model" in data[0]
        assert "total_cost" in data[0]

    def test_csv_to_file(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path, n=2)
        out_file = tmp_path / "report.csv"
        result = _run(db_path, "export", "--format", "csv", "--output", str(out_file))
        assert result.exit_code == 0
        assert out_file.exists()
        content = out_file.read_text()
        reader = csv.DictReader(io.StringIO(content))
        assert len(list(reader)) == 2

    def test_json_to_file(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path, n=2)
        out_file = tmp_path / "report.json"
        result = _run(db_path, "export", "--format", "json", "--output", str(out_file))
        assert result.exit_code == 0
        data = json.loads(out_file.read_text())
        assert len(data) == 2

    def test_csv_date_filter(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path, n=5)
        result = _run(db_path, "export", "--start", "2000-01-01", "--end", "2000-01-31", "--format", "csv")
        assert result.exit_code == 0
        # future-dated filter returns no records
        assert "No records" in result.output

    def test_project_filter(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path, n=3)
        result = _run(db_path, "export", "--format", "json", "--project", "test-project")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert all(r["project"] == "test-project" for r in data)


# ------------------------------------------------------------------ #
# projects                                                             #
# ------------------------------------------------------------------ #

class TestProjects:
    def test_empty_db(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(db_path, "projects")
        assert result.exit_code == 0
        assert "No projects" in result.output

    def test_shows_projects(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path)
        result = _run(db_path, "projects")
        assert result.exit_code == 0
        assert "test-project" in result.output

    def test_shows_cost(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path, n=3)
        result = _run(db_path, "projects")
        assert result.exit_code == 0
        # At least one cost entry
        assert "$" in result.output


# ------------------------------------------------------------------ #
# models                                                               #
# ------------------------------------------------------------------ #

class TestModels:
    def test_empty_db(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(db_path, "models")
        assert result.exit_code == 0
        assert "No model" in result.output

    def test_shows_models(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path)
        result = _run(db_path, "models")
        assert result.exit_code == 0
        assert "gpt-4o" in result.output or "claude" in result.output

    def test_shows_all_used_models(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _seed_db(db_path, n=4)
        result = _run(db_path, "models")
        assert result.exit_code == 0
        assert "openai" in result.output
        # Rich may truncate "anthropic" → "anthro…" in narrow terminals
        assert "anthro" in result.output


# ------------------------------------------------------------------ #
# config                                                               #
# ------------------------------------------------------------------ #

class TestConfig:
    def test_show_config_no_file(self, tmp_path: Path, monkeypatch) -> None:
        # Point config file to a temp location so we don't pollute real config
        monkeypatch.setattr(
            "token_meter.cli._CLI_CONFIG_PATH",
            tmp_path / "config.json",
        )
        result = _run(_make_db(tmp_path), "config")
        assert result.exit_code == 0
        assert "db-path" in result.output
        assert "project" in result.output

    def test_set_project(self, tmp_path: Path, monkeypatch) -> None:
        cfg_path = tmp_path / "config.json"
        monkeypatch.setattr("token_meter.cli._CLI_CONFIG_PATH", cfg_path)
        monkeypatch.setattr("token_meter.cli._load_cli_config", lambda: (
            json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        ))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--db-path", str(_make_db(tmp_path)), "config", "--set", "project=my-proj"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Config saved" in result.output

    def test_set_invalid_key(self, tmp_path: Path, monkeypatch) -> None:
        cfg_path = tmp_path / "config.json"
        monkeypatch.setattr("token_meter.cli._CLI_CONFIG_PATH", cfg_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--db-path", str(_make_db(tmp_path)), "config", "--set", "unknown-key=value"],
        )
        assert result.exit_code != 0

    def test_set_invalid_format(self, tmp_path: Path, monkeypatch) -> None:
        cfg_path = tmp_path / "config.json"
        monkeypatch.setattr("token_meter.cli._CLI_CONFIG_PATH", cfg_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--db-path", str(_make_db(tmp_path)), "config", "--set", "no-equals-sign"],
        )
        assert result.exit_code != 0


# ------------------------------------------------------------------ #
# budget                                                               #
# ------------------------------------------------------------------ #

class TestBudgetCLI:
    def test_budget_set_daily(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(db_path, "budget", "set", "--project", "my-app", "--daily", "10")
        assert result.exit_code == 0
        assert "Budget saved" in result.output

    def test_budget_set_weekly_monthly(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(db_path, "budget", "set", "--weekly", "50", "--monthly", "200")
        assert result.exit_code == 0
        assert "Budget saved" in result.output

    def test_budget_set_with_thresholds(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(
            db_path, "budget", "set",
            "--daily", "10",
            "--threshold", "0.8",
            "--threshold", "1.0",
        )
        assert result.exit_code == 0

    def test_budget_set_persists(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _run(db_path, "budget", "set", "--project", "proj-a", "--daily", "25")
        from token_meter.storage.sqlite import SQLiteStorage
        db = SQLiteStorage(db_path)
        try:
            cfg = db.get_budget_config("proj-a")
            assert cfg is not None
            assert cfg["daily"] == 25.0
        finally:
            db.close()

    def test_budget_set_updates_existing(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _run(db_path, "budget", "set", "--project", "p", "--daily", "10")
        _run(db_path, "budget", "set", "--project", "p", "--daily", "20")
        from token_meter.storage.sqlite import SQLiteStorage
        db = SQLiteStorage(db_path)
        try:
            cfg = db.get_budget_config("p")
            assert cfg["daily"] == 20.0
        finally:
            db.close()

    def test_budget_status_no_budgets(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(db_path, "budget", "status")
        assert result.exit_code == 0
        assert "No budgets" in result.output

    def test_budget_status_shows_configured(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _run(db_path, "budget", "set", "--project", "my-app", "--daily", "10")
        result = _run(db_path, "budget", "status")
        assert result.exit_code == 0
        assert "my-app" in result.output
        assert "daily" in result.output

    def test_budget_status_project_filter(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _run(db_path, "budget", "set", "--project", "proj-a", "--daily", "10")
        _run(db_path, "budget", "set", "--project", "proj-b", "--daily", "20")
        result = _run(db_path, "budget", "status", "--project", "proj-a")
        assert result.exit_code == 0
        assert "proj-a" in result.output
        assert "proj-b" not in result.output

    def test_budget_status_shows_spend_percentage(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _run(db_path, "budget", "set", "--project", "test-project", "--daily", "10")
        _seed_db(db_path)  # adds spend
        result = _run(db_path, "budget", "status", "--project", "test-project")
        assert result.exit_code == 0
        assert "%" in result.output


# ------------------------------------------------------------------ #
# alert                                                                #
# ------------------------------------------------------------------ #

class TestAlertCLI:
    def test_alert_add(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(
            db_path, "alert", "add",
            "--webhook", "https://hooks.slack.com/test",
            "--project", "my-app",
        )
        assert result.exit_code == 0
        assert "Webhook added" in result.output

    def test_alert_add_persists(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        url = "https://hooks.slack.com/test"
        _run(db_path, "alert", "add", "--webhook", url, "--project", "p")
        from token_meter.storage.sqlite import SQLiteStorage
        db = SQLiteStorage(db_path)
        try:
            cfg = db.get_budget_config("p")
            assert cfg is not None
            assert url in cfg["webhook_urls"]
        finally:
            db.close()

    def test_alert_add_dedup(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        url = "https://hooks.slack.com/test"
        _run(db_path, "alert", "add", "--webhook", url)
        result = _run(db_path, "alert", "add", "--webhook", url)
        assert result.exit_code == 0
        assert "already registered" in result.output

    def test_alert_list_empty(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(db_path, "alert", "list")
        assert result.exit_code == 0
        assert "No webhooks" in result.output

    def test_alert_list_shows_urls(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        url = "https://my-server.com/alert"
        _run(db_path, "alert", "add", "--webhook", url, "--project", "p")
        result = _run(db_path, "alert", "list")
        assert result.exit_code == 0
        assert url in result.output

    def test_alert_list_project_filter(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        _run(db_path, "alert", "add", "--webhook", "https://a.example.com", "--project", "proj-a")
        _run(db_path, "alert", "add", "--webhook", "https://b.example.com", "--project", "proj-b")
        result = _run(db_path, "alert", "list", "--project", "proj-a")
        assert result.exit_code == 0
        assert "proj-a" in result.output
        assert "proj-b" not in result.output


# ------------------------------------------------------------------ #
# anomalies                                                            #
# ------------------------------------------------------------------ #

class TestAnomaliesCLI:
    def test_anomalies_list_empty(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(db_path, "anomalies")
        assert result.exit_code == 0
        assert "No anomalies" in result.output

    def test_anomalies_list_with_data(self, tmp_path: Path) -> None:
        from token_meter.storage.sqlite import SQLiteStorage
        db_path = _make_db(tmp_path)
        db = SQLiteStorage(db_path)
        try:
            db.save_anomaly({
                "project": "my-app",
                "model": "gpt-4o",
                "date": "2024-01-15",
                "daily_cost": 45.20,
                "rolling_avg": 12.30,
                "rolling_std": 5.50,
                "z_score": 5.98,
            })
        finally:
            db.close()

        result = _run(db_path, "anomalies")
        assert result.exit_code == 0
        assert "my-app" in result.output
        assert "gpt-4o" in result.output

    def test_anomalies_days_option(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(db_path, "anomalies", "--days", "7")
        assert result.exit_code == 0

    def test_anomalies_project_filter(self, tmp_path: Path) -> None:
        from token_meter.storage.sqlite import SQLiteStorage
        db_path = _make_db(tmp_path)
        db = SQLiteStorage(db_path)
        try:
            for proj in ("app-a", "app-b"):
                db.save_anomaly({
                    "project": proj,
                    "model": "gpt-4o",
                    "date": "2024-01-15",
                    "daily_cost": 50.0,
                    "rolling_avg": 10.0,
                    "rolling_std": 2.0,
                    "z_score": 20.0,
                })
        finally:
            db.close()

        result = _run(db_path, "anomalies", "--project", "app-a")
        assert result.exit_code == 0
        assert "app-a" in result.output
        assert "app-b" not in result.output

    def test_anomalies_check_no_data(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(db_path, "anomalies", "check")
        assert result.exit_code == 0
        assert "No anomalies" in result.output

    def test_anomalies_check_custom_z_score(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(db_path, "anomalies", "check", "--z-score", "3.0")
        assert result.exit_code == 0

    def test_anomalies_check_project_filter(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = _run(db_path, "anomalies", "check", "--project", "nonexistent")
        assert result.exit_code == 0
        assert "No anomalies" in result.output
