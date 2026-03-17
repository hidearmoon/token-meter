"""Rich-based display utilities for the TokenMeter CLI."""
from __future__ import annotations

from typing import Any, Dict, List

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


# ------------------------------------------------------------------ #
# Formatting helpers                                                   #
# ------------------------------------------------------------------ #

def fmt_cost(value: float) -> str:
    """Format a USD cost value.

    - Zero  → "$0.0000"
    - < $0.001 → "<$0.001"
    - otherwise → "$X.XXXX" (4 decimal places)
    """
    if value == 0.0:
        return "$0.0000"
    if value < 0.001:
        return "<$0.001"
    return f"${value:.4f}"


def cost_style(value: float) -> str:
    """Rich style based on cost magnitude."""
    if value < 0.01:
        return "green"
    if value <= 1.0:
        return "yellow"
    return "red"


def styled_cost(value: float) -> Text:
    """Return a Rich Text object with cost colour."""
    return Text(fmt_cost(value), style=cost_style(value))


def fmt_tokens(n: int) -> str:
    return f"{n:,}"


def fmt_latency(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _markup_cost(value: float) -> str:
    style = cost_style(value)
    return f"[{style}]{fmt_cost(value)}[/{style}]"


# ------------------------------------------------------------------ #
# Dashboard panels                                                     #
# ------------------------------------------------------------------ #

def _period_panel(title: str, stats: Dict[str, Any]) -> Panel:
    if stats["call_count"] == 0:
        body = "[dim]No data[/dim]"
    else:
        avg = stats["total_cost"] / stats["call_count"]
        body = (
            f"[bold]Cost:[/bold]          {_markup_cost(stats['total_cost'])}\n"
            f"[bold]Calls:[/bold]         {stats['call_count']}\n"
            f"[bold]Tokens:[/bold]        {fmt_tokens(stats['total_tokens'])}\n"
            f"[bold]Avg/Call:[/bold]      {_markup_cost(avg)}\n"
            f"[bold]Avg Latency:[/bold]   {fmt_latency(stats['avg_latency_ms'])}"
        )
    return Panel(body, title=f"[bold cyan]{title}[/bold cyan]", box=box.ROUNDED, padding=(1, 2))


def print_dashboard(
    today_stats: Dict[str, Any],
    week_stats: Dict[str, Any],
    month_stats: Dict[str, Any],
    top_models: List[Dict[str, Any]],
    top_projects: List[Dict[str, Any]],
) -> None:
    """Print the full dashboard: 3 period panels + top models table."""
    console.print()
    console.print(
        Columns(
            [
                _period_panel("Today", today_stats),
                _period_panel("This Week", week_stats),
                _period_panel("This Month", month_stats),
            ],
            equal=True,
        )
    )

    # Top models
    if top_models:
        console.print()
        t = Table(
            title="[bold]Top Models[/bold]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
        )
        t.add_column("Provider", style="dim")
        t.add_column("Model")
        t.add_column("Calls", justify="right")
        t.add_column("Tokens", justify="right")
        t.add_column("Cost", justify="right")
        for row in top_models[:5]:
            t.add_row(
                row["provider"] or "",
                row["model"] or "",
                str(row["call_count"]),
                fmt_tokens(row["total_tokens"]),
                styled_cost(row["total_cost"]),
            )
        console.print(t)

    # Top projects (only when >1 project exists)
    if len(top_projects) > 1:
        console.print()
        p = Table(
            title="[bold]Top Projects[/bold]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
        )
        p.add_column("Project")
        p.add_column("Calls", justify="right")
        p.add_column("Cost", justify="right")
        for row in top_projects[:5]:
            p.add_row(
                row["project"],
                str(row["call_count"]),
                styled_cost(row["total_cost"]),
            )
        console.print(p)


# ------------------------------------------------------------------ #
# Summary table                                                        #
# ------------------------------------------------------------------ #

def print_summary_table(rows: List[Dict[str, Any]], group_by: str) -> None:
    if not rows:
        console.print("[dim]No data for the selected range.[/dim]")
        return

    t = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title=f"[bold]Summary — grouped by {group_by}[/bold]",
    )

    if group_by == "model":
        t.add_column("Provider", style="dim")
        t.add_column("Model")
    elif group_by == "provider":
        t.add_column("Provider")
    elif group_by == "project":
        t.add_column("Project")
    else:
        t.add_column("Period")

    t.add_column("Calls", justify="right")
    t.add_column("Input Tok", justify="right")
    t.add_column("Output Tok", justify="right")
    t.add_column("Total Tok", justify="right")
    t.add_column("Cost", justify="right")
    t.add_column("Avg Latency", justify="right")

    for row in rows:
        if group_by == "model":
            key_cols = [row.get("provider") or "", row.get("model") or ""]
        elif group_by == "provider":
            key_cols = [row.get("provider") or ""]
        elif group_by == "project":
            key_cols = [row.get("group") or ""]
        else:
            key_cols = [row.get("group") or ""]

        t.add_row(
            *key_cols,
            str(row["call_count"]),
            fmt_tokens(row["total_input_tokens"]),
            fmt_tokens(row["total_output_tokens"]),
            fmt_tokens(row["total_tokens"]),
            styled_cost(row["total_cost"]),
            fmt_latency(row["avg_latency_ms"]),
        )

    console.print(t)


# ------------------------------------------------------------------ #
# History table                                                        #
# ------------------------------------------------------------------ #

def print_history_table(records: List[Any]) -> None:
    if not records:
        console.print("[dim]No records found.[/dim]")
        return

    t = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title=f"[bold]Call History ({len(records)} records)[/bold]",
    )
    t.add_column("Time", style="dim")
    t.add_column("Provider", style="dim")
    t.add_column("Model")
    t.add_column("In Tok", justify="right")
    t.add_column("Out Tok", justify="right")
    t.add_column("Cost", justify="right")
    t.add_column("Latency", justify="right")
    t.add_column("Project", style="dim")

    for r in records:
        ts = r.timestamp
        if hasattr(ts, "strftime"):
            ts_str = ts.strftime("%m-%d %H:%M:%S")
        else:
            ts_str = str(ts)[:19]

        t.add_row(
            ts_str,
            r.provider,
            r.model,
            fmt_tokens(r.input_tokens),
            fmt_tokens(r.output_tokens),
            styled_cost(r.total_cost),
            fmt_latency(r.latency_ms),
            r.project,
        )

    console.print(t)


# ------------------------------------------------------------------ #
# Projects table                                                       #
# ------------------------------------------------------------------ #

def print_projects_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        console.print("[dim]No projects found.[/dim]")
        return

    t = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title="[bold]Projects[/bold]",
    )
    t.add_column("Project")
    t.add_column("Calls", justify="right")
    t.add_column("Tokens", justify="right")
    t.add_column("Cost", justify="right")
    t.add_column("Avg Latency", justify="right")
    t.add_column("Last Call", style="dim")

    for row in rows:
        t.add_row(
            row["project"],
            str(row["call_count"]),
            fmt_tokens(row["total_tokens"]),
            styled_cost(row["total_cost"]),
            fmt_latency(row["avg_latency_ms"]),
            (row["last_call"] or "")[:19],
        )

    console.print(t)


# ------------------------------------------------------------------ #
# Models table                                                         #
# ------------------------------------------------------------------ #

def print_models_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        console.print("[dim]No model usage found.[/dim]")
        return

    t = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title="[bold]Model Usage (ranked by cost)[/bold]",
    )
    t.add_column("Provider", style="dim")
    t.add_column("Model")
    t.add_column("Calls", justify="right")
    t.add_column("In Tok", justify="right")
    t.add_column("Out Tok", justify="right")
    t.add_column("Total Tok", justify="right")
    t.add_column("Cost", justify="right")
    t.add_column("Avg Latency", justify="right")

    for row in rows:
        t.add_row(
            row["provider"],
            row["model"],
            str(row["call_count"]),
            fmt_tokens(row["total_input_tokens"]),
            fmt_tokens(row["total_output_tokens"]),
            fmt_tokens(row["total_tokens"]),
            styled_cost(row["total_cost"]),
            fmt_latency(row["avg_latency_ms"]),
        )

    console.print(t)
