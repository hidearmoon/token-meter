"""TokenMeter CLI — `tokenmeter` / `tm` commands.

Entry points (pyproject.toml):
    tokenmeter = "token_meter.cli:cli"
    tm         = "token_meter.cli:cli"
"""
from __future__ import annotations

import csv
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from .config import DEFAULT_DB_PATH
from .display import (
    console,
    fmt_cost,
    print_dashboard,
    print_history_table,
    print_models_table,
    print_projects_table,
    print_summary_table,
)
from .storage.sqlite import SQLiteStorage

# ------------------------------------------------------------------ #
# Config file helpers                                                  #
# ------------------------------------------------------------------ #

_CLI_CONFIG_PATH = Path.home() / ".token-meter" / "config.json"


def _load_cli_config() -> Dict[str, Any]:
    if _CLI_CONFIG_PATH.exists():
        try:
            with open(_CLI_CONFIG_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cli_config(cfg: Dict[str, Any]) -> None:
    _CLI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CLI_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def _resolve_db_path(explicit: Optional[str]) -> Path:
    """Resolve DB path from explicit flag → config file → default."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    cfg = _load_cli_config()
    if "db-path" in cfg:
        return Path(cfg["db-path"]).expanduser().resolve()
    return DEFAULT_DB_PATH


# ------------------------------------------------------------------ #
# Date parsing                                                         #
# ------------------------------------------------------------------ #

def _parse_date(value: str, end_of_period: bool = False) -> datetime:
    """Parse a date string or shortcut into a UTC datetime.

    Shortcuts: today | this-week | this-month
    Standard:  YYYY-MM-DD  or ISO-8601 datetime
    """
    today = datetime.now(timezone.utc).date()

    if value == "today":
        if end_of_period:
            return datetime(today.year, today.month, today.day, 23, 59, 59, tzinfo=timezone.utc)
        return datetime(today.year, today.month, today.day, tzinfo=timezone.utc)

    if value == "this-week":
        monday = today - timedelta(days=today.weekday())
        if end_of_period:
            sunday = monday + timedelta(days=6)
            return datetime(sunday.year, sunday.month, sunday.day, 23, 59, 59, tzinfo=timezone.utc)
        return datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)

    if value == "this-month":
        if end_of_period:
            # last moment of current month
            if today.month == 12:
                first_next = datetime(today.year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                first_next = datetime(today.year, today.month + 1, 1, tzinfo=timezone.utc)
            return first_next - timedelta(seconds=1)
        return datetime(today.year, today.month, 1, tzinfo=timezone.utc)

    # Standard ISO date/datetime
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise click.BadParameter(
            f"Invalid date: {value!r}. "
            "Use YYYY-MM-DD or today / this-week / this-month"
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
        # If only date supplied (no time), treat as end-of-day for end params
        if end_of_period and "T" not in value and " " not in value:
            dt = dt.replace(hour=23, minute=59, second=59)
    return dt


# ------------------------------------------------------------------ #
# CLI group                                                            #
# ------------------------------------------------------------------ #

@click.group()
@click.option(
    "--db-path",
    envvar="TOKEN_METER_DB_PATH",
    default=None,
    metavar="PATH",
    help="SQLite database path (overrides config and default).",
)
@click.pass_context
def cli(ctx: click.Context, db_path: Optional[str]) -> None:
    """TokenMeter — LLM cost & usage observability CLI."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path


# ------------------------------------------------------------------ #
# dashboard                                                            #
# ------------------------------------------------------------------ #

@cli.command()
@click.pass_context
def dashboard(ctx: click.Context) -> None:
    """Real-time overview: today / this-week / this-month costs and top models."""
    db = SQLiteStorage(_resolve_db_path(ctx.obj.get("db_path")))
    try:
        now = datetime.now(timezone.utc)
        today_date = now.date()

        today_start = datetime(today_date.year, today_date.month, today_date.day, tzinfo=timezone.utc)
        week_start = today_start - timedelta(days=today_date.weekday())
        month_start = datetime(today_date.year, today_date.month, 1, tzinfo=timezone.utc)

        today_stats = db.aggregate(start=today_start)
        week_stats = db.aggregate(start=week_start)
        month_stats = db.aggregate(start=month_start)
        top_models = db.aggregate_by_model(start=month_start)
        top_projects = db.get_projects()

        print_dashboard(today_stats, week_stats, month_stats, top_models, top_projects)
    finally:
        db.close()


# ------------------------------------------------------------------ #
# summary                                                              #
# ------------------------------------------------------------------ #

@cli.command()
@click.option("--start", default=None, help="Start date (YYYY-MM-DD or today/this-week/this-month).")
@click.option("--end", default=None, help="End date (YYYY-MM-DD or today/this-week/this-month).")
@click.option(
    "--group-by",
    default="model",
    show_default=True,
    type=click.Choice(["model", "provider", "project", "day", "week", "month"], case_sensitive=False),
    help="Aggregation dimension.",
)
@click.option("--project", default=None, help="Filter by project name.")
@click.pass_context
def summary(
    ctx: click.Context,
    start: Optional[str],
    end: Optional[str],
    group_by: str,
    project: Optional[str],
) -> None:
    """Aggregated cost & token stats for a time range."""
    start_dt = _parse_date(start) if start else None
    end_dt = _parse_date(end, end_of_period=True) if end else None

    db = SQLiteStorage(_resolve_db_path(ctx.obj.get("db_path")))
    try:
        rows = db.get_summary(start=start_dt, end=end_dt, group_by=group_by, project=project)
        print_summary_table(rows, group_by)
    finally:
        db.close()


# ------------------------------------------------------------------ #
# history                                                              #
# ------------------------------------------------------------------ #

@cli.command()
@click.option("--limit", default=20, show_default=True, type=int, help="Number of records to show.")
@click.option("--provider", default=None, help="Filter by provider (openai/anthropic/google).")
@click.option("--model", default=None, help="Filter by model name.")
@click.option("--project", default=None, help="Filter by project name.")
@click.option("--start", default=None, help="Start date filter.")
@click.option("--end", default=None, help="End date filter.")
@click.pass_context
def history(
    ctx: click.Context,
    limit: int,
    provider: Optional[str],
    model: Optional[str],
    project: Optional[str],
    start: Optional[str],
    end: Optional[str],
) -> None:
    """Most recent LLM API call records."""
    start_dt = _parse_date(start) if start else None
    end_dt = _parse_date(end, end_of_period=True) if end else None

    db = SQLiteStorage(_resolve_db_path(ctx.obj.get("db_path")))
    try:
        records = db.get_records(
            start=start_dt,
            end=end_dt,
            limit=limit,
            provider=provider,
            model=model,
            project=project,
        )
        print_history_table(records)
    finally:
        db.close()


# ------------------------------------------------------------------ #
# export                                                               #
# ------------------------------------------------------------------ #

@cli.command()
@click.option(
    "--format",
    "fmt",
    default="csv",
    show_default=True,
    type=click.Choice(["csv", "json"], case_sensitive=False),
    help="Output format.",
)
@click.option("--start", default=None, help="Start date filter.")
@click.option("--end", default=None, help="End date filter.")
@click.option("--project", default=None, help="Filter by project name.")
@click.option("--output", default=None, metavar="FILE", help="Output file (default: stdout).")
@click.pass_context
def export(
    ctx: click.Context,
    fmt: str,
    start: Optional[str],
    end: Optional[str],
    project: Optional[str],
    output: Optional[str],
) -> None:
    """Export call records to CSV or JSON."""
    start_dt = _parse_date(start) if start else None
    end_dt = _parse_date(end, end_of_period=True) if end else None

    db = SQLiteStorage(_resolve_db_path(ctx.obj.get("db_path")))
    try:
        records = db.get_records(
            start=start_dt,
            end=end_dt,
            limit=100_000,
            project=project,
        )
    finally:
        db.close()

    if not records:
        console.print("[dim]No records to export.[/dim]")
        return

    dicts = [r.to_dict() for r in records]

    if fmt == "json":
        content = json.dumps(dicts, indent=2, default=str)
    else:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(dicts[0].keys()))
        writer.writeheader()
        writer.writerows(dicts)
        content = buf.getvalue()

    if output:
        Path(output).write_text(content, encoding="utf-8")
        console.print(f"[green]Exported {len(records)} records → {output}[/green]")
    else:
        sys.stdout.write(content)


# ------------------------------------------------------------------ #
# projects                                                             #
# ------------------------------------------------------------------ #

@cli.command()
@click.pass_context
def projects(ctx: click.Context) -> None:
    """List all projects and their total usage."""
    db = SQLiteStorage(_resolve_db_path(ctx.obj.get("db_path")))
    try:
        rows = db.get_projects()
        print_projects_table(rows)
    finally:
        db.close()


# ------------------------------------------------------------------ #
# models                                                               #
# ------------------------------------------------------------------ #

@cli.command()
@click.pass_context
def models(ctx: click.Context) -> None:
    """List all models used, ranked by total cost."""
    db = SQLiteStorage(_resolve_db_path(ctx.obj.get("db_path")))
    try:
        rows = db.get_models()
        print_models_table(rows)
    finally:
        db.close()


# ------------------------------------------------------------------ #
# config                                                               #
# ------------------------------------------------------------------ #

@cli.command()
@click.option(
    "--set",
    "set_pairs",
    multiple=True,
    metavar="KEY=VALUE",
    help="Set a config value. Supported keys: project, db-path.",
)
def config(set_pairs: tuple) -> None:
    """View or update CLI configuration.

    \b
    Examples:
      tm config
      tm config --set project=my-app
      tm config --set db-path=~/work/usage.db
    """
    cfg = _load_cli_config()

    if set_pairs:
        for pair in set_pairs:
            if "=" not in pair:
                raise click.BadParameter(f"Expected KEY=VALUE, got: {pair!r}", param_hint="--set")
            key, _, value = pair.partition("=")
            key = key.strip()
            _ALLOWED_KEYS = {"project", "db-path"}
            if key not in _ALLOWED_KEYS:
                raise click.BadParameter(
                    f"Unknown config key {key!r}. Allowed: {sorted(_ALLOWED_KEYS)}",
                    param_hint="--set",
                )
            cfg[key] = value.strip()
        _save_cli_config(cfg)
        console.print(f"[green]Config saved → {_CLI_CONFIG_PATH}[/green]")

    # Display current config
    from rich.table import Table
    from rich import box as rbox

    t = Table(box=rbox.SIMPLE_HEAVY, show_header=True, header_style="bold magenta", title="[bold]Config[/bold]")
    t.add_column("Key")
    t.add_column("Value")
    t.add_row("db-path", str(_resolve_db_path(None)))
    t.add_row("project", cfg.get("project", "default"))
    t.add_row("config-file", str(_CLI_CONFIG_PATH))
    console.print(t)
