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

from rich import box as rbox
from rich.table import Table

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
    t = Table(box=rbox.SIMPLE_HEAVY, show_header=True, header_style="bold magenta", title="[bold]Config[/bold]")
    t.add_column("Key")
    t.add_column("Value")
    t.add_row("db-path", str(_resolve_db_path(None)))
    t.add_row("project", cfg.get("project", "default"))
    t.add_row("config-file", str(_CLI_CONFIG_PATH))
    console.print(t)


# ------------------------------------------------------------------ #
# budget                                                               #
# ------------------------------------------------------------------ #

@cli.group()
def budget() -> None:
    """Manage per-project spending budgets."""


@budget.command("set")
@click.option("--project", default="default", show_default=True, help="Project name.")
@click.option("--daily", default=None, type=float, help="Daily budget limit in USD.")
@click.option("--weekly", default=None, type=float, help="Weekly budget limit in USD.")
@click.option("--monthly", default=None, type=float, help="Monthly budget limit in USD.")
@click.option(
    "--threshold",
    "thresholds",
    multiple=True,
    type=float,
    help="Alert threshold fraction (e.g. 0.8). May be repeated. Default: 0.8 0.9 1.0",
)
@click.pass_context
def budget_set(
    ctx: click.Context,
    project: str,
    daily: Optional[float],
    weekly: Optional[float],
    monthly: Optional[float],
    thresholds: tuple,
) -> None:
    """Set budget limits for a project.

    \b
    Examples:
      tm budget set --daily 10 --weekly 50 --monthly 200
      tm budget set --project my-app --daily 5 --threshold 0.8 --threshold 1.0
    """
    db = SQLiteStorage(_resolve_db_path(ctx.obj.get("db_path") if ctx.obj else None))
    try:
        existing = db.get_budget_config(project) or {}
        if daily is not None:
            existing["daily"] = daily
        if weekly is not None:
            existing["weekly"] = weekly
        if monthly is not None:
            existing["monthly"] = monthly
        if thresholds:
            existing["thresholds"] = sorted(set(thresholds))
        elif "thresholds" not in existing:
            existing["thresholds"] = [0.8, 0.9, 1.0]
        if "webhook_urls" not in existing:
            existing["webhook_urls"] = []
        db.set_budget_config(project, existing)
        console.print(f"[green]Budget saved for project '{project}'[/green]")
        _print_budget_row(project, existing)
    finally:
        db.close()


@budget.command("status")
@click.option("--project", default=None, help="Filter by project name.")
@click.pass_context
def budget_status(ctx: click.Context, project: Optional[str]) -> None:
    """Show current budget usage for all (or one) project(s)."""
    db = SQLiteStorage(_resolve_db_path(ctx.obj.get("db_path") if ctx.obj else None))
    try:
        configs = db.get_all_budget_configs()
        if project:
            configs = [c for c in configs if c.get("project") == project]
        if not configs:
            console.print("[dim]No budgets configured. Use `tm budget set` to add one.[/dim]")
            return

        today = datetime.now(timezone.utc).date()

        def period_start(period: str) -> datetime:
            if period == "daily":
                return datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
            if period == "weekly":
                monday = today - timedelta(days=today.weekday())
                return datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)
            return datetime(today.year, today.month, 1, tzinfo=timezone.utc)

        t = Table(
            box=rbox.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            title="[bold]Budget Status[/bold]",
        )
        t.add_column("Project")
        t.add_column("Period")
        t.add_column("Limit")
        t.add_column("Spent")
        t.add_column("Remaining")
        t.add_column("Usage %", justify="right")

        for cfg in configs:
            proj = cfg.get("project", "default")
            for period in ("daily", "weekly", "monthly"):
                limit = cfg.get(period)
                if limit is None:
                    continue
                start = period_start(period)
                spent = db.get_period_spend(proj, start)
                pct = (spent / limit * 100) if limit > 0 else 0
                remaining = max(0.0, limit - spent)
                pct_color = "red" if pct >= 100 else ("yellow" if pct >= 80 else "green")
                t.add_row(
                    proj,
                    period,
                    f"${limit:.2f}",
                    fmt_cost(spent),
                    fmt_cost(remaining),
                    f"[{pct_color}]{pct:.1f}%[/{pct_color}]",
                )
        console.print(t)
    finally:
        db.close()


def _print_budget_row(project: str, cfg: Dict[str, Any]) -> None:
    t = Table(box=rbox.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
    t.add_column("Project")
    t.add_column("Daily")
    t.add_column("Weekly")
    t.add_column("Monthly")
    t.add_column("Thresholds")
    t.add_column("Webhooks")
    t.add_row(
        project,
        f"${cfg['daily']:.2f}" if cfg.get("daily") is not None else "[dim]—[/dim]",
        f"${cfg['weekly']:.2f}" if cfg.get("weekly") is not None else "[dim]—[/dim]",
        f"${cfg['monthly']:.2f}" if cfg.get("monthly") is not None else "[dim]—[/dim]",
        ", ".join(f"{th:.0%}" for th in cfg.get("thresholds", [])),
        str(len(cfg.get("webhook_urls", []))),
    )
    console.print(t)


# ------------------------------------------------------------------ #
# alert                                                                #
# ------------------------------------------------------------------ #

@cli.group()
def alert() -> None:
    """Manage alert destinations (webhooks)."""


@alert.command("add")
@click.option("--webhook", "url", required=True, help="Webhook URL to POST alerts to.")
@click.option("--project", default="default", show_default=True, help="Project name.")
@click.pass_context
def alert_add(ctx: click.Context, url: str, project: str) -> None:
    """Register a webhook URL for budget / anomaly alerts.

    \b
    Examples:
      tm alert add --webhook https://hooks.slack.com/services/...
      tm alert add --project my-app --webhook https://my-server.com/alert
    """
    db = SQLiteStorage(_resolve_db_path(ctx.obj.get("db_path") if ctx.obj else None))
    try:
        existing = db.get_budget_config(project) or {}
        urls: List[str] = existing.get("webhook_urls", [])
        if url in urls:
            console.print(f"[yellow]Webhook already registered for '{project}'[/yellow]")
            return
        urls.append(url)
        existing["webhook_urls"] = urls
        if "thresholds" not in existing:
            existing["thresholds"] = [0.8, 0.9, 1.0]
        db.set_budget_config(project, existing)
        console.print(f"[green]Webhook added for project '{project}'[/green]")
        console.print(f"  [dim]{url}[/dim]")
    finally:
        db.close()


@alert.command("list")
@click.option("--project", default=None, help="Filter by project name.")
@click.pass_context
def alert_list(ctx: click.Context, project: Optional[str]) -> None:
    """List all registered webhook URLs."""
    db = SQLiteStorage(_resolve_db_path(ctx.obj.get("db_path") if ctx.obj else None))
    try:
        configs = db.get_all_budget_configs()
        if project:
            configs = [c for c in configs if c.get("project") == project]

        t = Table(
            box=rbox.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            title="[bold]Registered Webhooks[/bold]",
        )
        t.add_column("Project")
        t.add_column("Webhook URL")

        for cfg in configs:
            proj = cfg.get("project", "default")
            for wurl in cfg.get("webhook_urls", []):
                t.add_row(proj, wurl)

        if t.row_count == 0:
            console.print("[dim]No webhooks registered. Use `tm alert add --webhook URL`.[/dim]")
        else:
            console.print(t)
    finally:
        db.close()


# ------------------------------------------------------------------ #
# anomalies                                                            #
# ------------------------------------------------------------------ #

@cli.group(invoke_without_command=True)
@click.option("--days", default=30, show_default=True, type=int, help="Look-back window in days.")
@click.option("--project", default=None, help="Filter by project name.")
@click.pass_context
def anomalies(ctx: click.Context, days: int, project: Optional[str]) -> None:
    """View detected cost anomalies.

    Run without a sub-command to list recent anomalies.
    """
    ctx.ensure_object(dict)
    ctx.obj["anomaly_days"] = days
    ctx.obj["anomaly_project"] = project

    if ctx.invoked_subcommand is not None:
        return

    db_path = ctx.obj.get("db_path")
    db = SQLiteStorage(_resolve_db_path(db_path))
    try:
        rows = db.get_anomalies(project=project, days=days)
        _print_anomalies_table(rows)
    finally:
        db.close()


@anomalies.command("check")
@click.option("--project", default=None, help="Limit check to one project.")
@click.option(
    "--z-score",
    "z_threshold",
    default=2.0,
    show_default=True,
    type=float,
    help="Z-score threshold for anomaly detection.",
)
@click.pass_context
def anomalies_check(
    ctx: click.Context, project: Optional[str], z_threshold: float
) -> None:
    """Manually run anomaly detection for yesterday's data.

    \b
    Examples:
      tm anomalies check
      tm anomalies check --project my-app --z-score 2.5
    """
    ctx.ensure_object(dict)
    db_path = ctx.obj.get("db_path")
    db = SQLiteStorage(_resolve_db_path(db_path))
    try:
        from .alerts import AlertSender
        from .anomaly import AnomalyDetector

        webhook_urls: List[str] = []
        for cfg in db.get_all_budget_configs():
            webhook_urls.extend(cfg.get("webhook_urls", []))
        webhook_urls = list(dict.fromkeys(webhook_urls))  # deduplicate, preserve order

        sender = AlertSender()
        detector = AnomalyDetector(
            db, sender, webhook_urls=webhook_urls, z_threshold=z_threshold
        )
        console.print(
            f"[dim]Running anomaly detection for yesterday "
            f"(z-score threshold: {z_threshold})…[/dim]"
        )
        found = detector.check_yesterday(project=project, z_threshold=z_threshold)
        if found:
            n = len(found)
            console.print(f"[yellow]Detected {n} anomal{'y' if n == 1 else 'ies'}:[/yellow]")
            _print_anomalies_table(found)
        else:
            console.print("[green]No anomalies detected.[/green]")
    finally:
        db.close()


def _print_anomalies_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        console.print("[dim]No anomalies found.[/dim]")
        return

    t = Table(
        box=rbox.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title="[bold]Cost Anomalies[/bold]",
    )
    t.add_column("Date")
    t.add_column("Project")
    t.add_column("Model")
    t.add_column("Daily Cost", justify="right")
    t.add_column("Avg (30d)", justify="right")
    t.add_column("Std Dev", justify="right")
    t.add_column("Z-Score", justify="right")

    for row in rows:
        z = row.get("z_score", 0.0)
        z_color = "red" if z >= 3.0 else "yellow"
        t.add_row(
            str(row.get("date", "")),
            str(row.get("project", "")),
            str(row.get("model", "") or "—"),
            fmt_cost(row.get("daily_cost", 0)),
            fmt_cost(row.get("rolling_avg", 0)),
            fmt_cost(row.get("rolling_std", 0)),
            f"[{z_color}]{z:.2f}[/{z_color}]",
        )
    console.print(t)
