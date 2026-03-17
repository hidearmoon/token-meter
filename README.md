# TokenMeter

**Track every token, control every dollar.**

Lightweight LLM API cost & usage observability — monkey-patch OpenAI / Anthropic / Google in one line, store everything locally in SQLite, get budget alerts and anomaly detection out of the box.

[![CI](https://github.com/hidearmoon/token-meter/actions/workflows/ci.yml/badge.svg)](https://github.com/hidearmoon/token-meter/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/hidearmoon/token-meter/branch/main/graph/badge.svg)](https://codecov.io/gh/hidearmoon/token-meter)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) | [中文](README_zh.md)

---

- 🔌 **One-line setup** — `import token_meter; token_meter.init()` and you're tracking
- 🏠 **Local-first** — all data stays in a local SQLite file, no third-party SaaS
- 📊 **Full visibility** — real-time dashboard, budget alerts, anomaly detection, CSV/JSON export

---

## Table of Contents

- [Quick Start](#quick-start)
- [Why TokenMeter](#why-tokenmeter)
- [Installation](#installation)
- [Supported Providers](#supported-providers)
- [SDK API](#sdk-api)
- [CLI Reference](#cli-reference)
- [Budget Alerts](#budget-alerts)
- [Anomaly Detection](#anomaly-detection)
- [Custom Pricing](#custom-pricing)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

```bash
pip install "token-meter[openai,cli]"
```

```python
import token_meter
token_meter.init(project="my-app")  # ← add this one line

# Your existing code, unchanged:
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)

# See your costs instantly:
stats = token_meter.get_tracker().aggregate()
print(f"Cost: ${stats['total_cost']:.4f}  |  Tokens: {stats['total_tokens']:,}")
# Cost: $0.0035  |  Tokens: 1,234
```

```bash
# Live dashboard in your terminal:
tm dashboard
```

---

## Why TokenMeter

| Feature | **TokenMeter** | Helicone | LangSmith | LiteLLM Proxy | token-cost-guard |
|---------|:--------------:|:--------:|:---------:|:-------------:|:----------------:|
| Monkey-patch (zero code change) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Local SQLite (no SaaS) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Budget alerts | ✅ | ✅ | ❌ | ✅ | ❌ |
| Anomaly detection | ✅ | ✅ | ❌ | ❌ | ❌ |
| Open source | ✅ | ✅ | ✅ | ✅ | ✅ |
| Zero core dependencies | ✅ | ❌ | ❌ | ❌ | ❌ |
| CLI dashboard | ✅ | ❌ | ❌ | ✅ | ❌ |
| Streaming support | ✅ | ✅ | ✅ | ✅ | ❌ |

**Key differentiators:**
- The only tool that combines monkey-patching + local storage + budget alerts + anomaly detection in one zero-dependency package
- Helicone/LangSmith require routing all traffic through their servers; TokenMeter never touches the network
- LiteLLM requires running a separate proxy process; TokenMeter is a pure in-process library

---

## Installation

```bash
# Core only (no provider SDKs):
pip install token-meter

# With specific provider:
pip install "token-meter[openai]"       # OpenAI SDK included
pip install "token-meter[anthropic]"    # Anthropic SDK included
pip install "token-meter[google]"       # Google GenAI SDK included
pip install "token-meter[all]"          # All providers + CLI (rich + click)

# CLI only (if SDKs already installed):
pip install "token-meter[cli]"
```

---

## Supported Providers

| Provider | SDK requirement | Sync | Async | Streaming |
|----------|----------------|:----:|:-----:|:---------:|
| OpenAI | `openai >= 1.0` | ✅ | ✅ | ✅ |
| Anthropic | `anthropic >= 0.20` | ✅ | ✅ | ✅ |
| Google Gemini | `google-genai >= 1.0` | ✅ | ✅ | ✅ |

**Streaming implementation:**
- **OpenAI** — injects `stream_options={"include_usage": True}` so the final chunk carries token counts
- **Anthropic** — accumulates `input_tokens` from `message_start` and `output_tokens` from `message_delta` events
- **Google** — reads `usage_metadata` from the final chunk

---

## SDK API

### `token_meter.init()`

```python
import token_meter

tracker = token_meter.init(
    project="my-app",               # logical project name for multi-project isolation
    db_path="~/data/usage.db",      # custom SQLite path (default: ~/.token-meter/usage.db)
    providers=["openai", "anthropic"],  # which SDKs to patch (default: all installed)
    budgets={                        # optional budget limits in USD
        "daily": 5.00,
        "weekly": 25.00,
        "monthly": 80.00,
    },
    alerts=[                         # optional webhook destinations
        {"type": "webhook", "url": "https://hooks.slack.com/..."},
    ],
    alert_thresholds=[0.8, 0.9, 1.0],  # trigger at 80%, 90%, 100% of budget
)
```

### Querying data

```python
tracker = token_meter.get_tracker()

# Aggregate totals
stats = tracker.aggregate()
# {
#   "call_count": 42,
#   "total_input_tokens": 125000,
#   "total_output_tokens": 38000,
#   "total_tokens": 163000,
#   "total_cost": 0.5124,
#   "avg_latency_ms": 812.4,
#   "first_call": "2025-01-15T09:00:00+00:00",
#   "last_call":  "2025-01-15T17:42:00+00:00"
# }

# Per-model breakdown
rows = tracker.aggregate_by_model()
# [
#   {"provider": "openai", "model": "gpt-4o", "call_count": 30, "total_cost": 0.48},
#   {"provider": "anthropic", "model": "claude-sonnet-4", "call_count": 12, "total_cost": 0.032},
# ]

# Raw call records (with filters)
records = tracker.query(
    project="my-app",
    provider="openai",
    model="gpt-4o",
    limit=50,
)

# Stop tracking and restore original SDK methods
token_meter.disable()
```

---

## CLI Reference

Install with `pip install "token-meter[cli]"`. Both `tokenmeter` and the short alias `tm` work.

### `tm dashboard`

Real-time overview of today / this-week / this-month spending and top models.

```
╭─────────────────────── TokenMeter Dashboard ───────────────────────╮
│ Today: $1.24  │  This Week: $8.71  │  This Month: $31.05           │
│ Calls: 312    │  Tokens: 2.1M      │  Avg latency: 843ms           │
╰────────────────────────────────────────────────────────────────────╯
 Top Models (today)
 gpt-4o          182 calls   $0.98
 claude-sonnet-4  98 calls   $0.21
 gpt-4o-mini      32 calls   $0.05
```

### `tm summary`

Aggregated cost & token stats for a custom time range.

```bash
tm summary --start this-week --group-by model
tm summary --start 2025-01-01 --end 2025-01-31 --project my-app
```

Options: `--start`, `--end` (YYYY-MM-DD or `today`/`this-week`/`this-month`), `--group-by` (model|provider|project|day|week|month), `--project`

### `tm history`

Most recent LLM call records.

```bash
tm history --limit 50 --provider openai --model gpt-4o
```

Options: `--limit` (default 20), `--provider`, `--model`, `--project`, `--start`, `--end`

### `tm export`

Export records to CSV or JSON.

```bash
tm export --format json --start this-month --output costs.json
tm export --format csv --project my-app > report.csv
```

Options: `--format` (csv|json), `--start`, `--end`, `--project`, `--output`

### `tm projects`

List all projects and their total usage.

```bash
tm projects
```

### `tm models`

All models ranked by total spend.

```bash
tm models
```

### `tm budget set` / `tm budget status`

```bash
tm budget set --project my-app --daily 5.00 --weekly 25.00 --monthly 80.00
tm budget status
tm budget status --project my-app
```

### `tm alert add` / `tm alert list`

```bash
tm alert add --webhook https://hooks.slack.com/services/... --project my-app
tm alert list
```

### `tm anomalies` / `tm anomalies check`

```bash
tm anomalies --days 30
tm anomalies check --z-score 2.5 --project my-app
```

### `tm config`

```bash
tm config --set project=my-app --set db-path=~/data/usage.db
```

### Global options

```bash
tm --db-path /custom/path/usage.db dashboard   # override DB path for any command
# or set TOKEN_METER_DB_PATH env var
```

---

## Budget Alerts

Set per-project spend limits. TokenMeter fires webhook notifications at configurable threshold fractions (80%, 90%, 100% by default).

### Via SDK

```python
token_meter.init(
    project="production",
    budgets={"daily": 10.00, "monthly": 200.00},
    alerts=[{"type": "webhook", "url": "https://hooks.slack.com/services/..."}],
    alert_thresholds=[0.8, 0.9, 1.0],
)
```

### Via CLI

```bash
tm budget set --project production --daily 10.00 --monthly 200.00
tm alert add --webhook https://hooks.slack.com/services/... --project production
```

### Webhook payload

```json
{
  "alert": "budget_threshold",
  "project": "production",
  "period": "daily",
  "threshold": 0.8,
  "current_spend": 8.14,
  "budget_limit": 10.00,
  "percentage": 81.4,
  "timestamp": "2025-01-15T14:23:00+00:00",
  "top_models": [
    {"model": "gpt-4o", "cost": 6.20},
    {"model": "claude-sonnet-4", "cost": 1.94}
  ]
}
```

Webhooks are fire-and-forget (daemon thread, 5 s timeout). A failed delivery is logged as a warning and never raises an exception in your application.

---

## Anomaly Detection

TokenMeter uses a **Z-score** on a 30-day rolling window of daily costs to flag unusual spending spikes — no external ML libraries, pure stdlib `statistics`.

**Algorithm:**
1. Collect the past 30 days of daily cost per `(project, model)` pair
2. Compute mean and standard deviation
3. If `(today_cost − mean) / std ≥ threshold` (default 2.0), fire an alert
4. Requires at least 7 days of history before activating

Detection runs automatically on the first API call of each new day. You can also trigger it manually:

```bash
tm anomalies check --z-score 2.0 --project my-app
```

### Anomaly webhook payload

```json
{
  "alert": "cost_anomaly",
  "project": "my-app",
  "model": "gpt-4o",
  "date": "2025-01-15",
  "daily_cost": 45.20,
  "rolling_avg": 12.30,
  "rolling_std": 3.80,
  "z_score": 8.66,
  "timestamp": "2025-01-15T09:00:00+00:00"
}
```

---

## Custom Pricing

```python
from token_meter.pricing import add_custom_pricing

add_custom_pricing("my-finetune-v1", input_per_1m=5.00, output_per_1m=20.00)
```

Built-in pricing table (USD / 1M tokens):

| Model | Input | Output |
|-------|------:|-------:|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4.1 | $2.00 | $8.00 |
| claude-opus-4 | $15.00 | $75.00 |
| claude-sonnet-4 | $3.00 | $15.00 |
| claude-haiku-3.5 | $0.80 | $4.00 |
| gemini-2.5-pro | $1.25 | $10.00 |
| gemini-2.5-flash | $0.30 | $2.50 |

Model names are fuzzy-matched (e.g. `gpt-4o-2024-11-20` → `gpt-4o`). Unknown models are recorded with cost `$0.00` and a warning is emitted.

---

## Configuration

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TOKEN_METER_DB_PATH` | SQLite database path | `~/.token-meter/usage.db` |
| `TOKEN_METER_PROJECT` | Default project name | `default` |

### `token_meter.init()` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | `str` | `"default"` | Logical project name |
| `db_path` | `str \| None` | `None` | Custom SQLite path |
| `providers` | `list[str] \| None` | `None` | Providers to patch (`["openai","anthropic","google"]`) |
| `budgets` | `dict \| None` | `None` | Keys: `daily`, `weekly`, `monthly` (USD float) |
| `alerts` | `list[dict] \| None` | `None` | Alert destinations: `{"type": "webhook", "url": "..."}` |
| `alert_thresholds` | `list[float] \| None` | `[0.8, 0.9, 1.0]` | Fractions of budget that trigger notifications |

---

## Architecture

```
token-meter/
├── src/token_meter/
│   ├── __init__.py          # Public API: init(), disable(), get_tracker()
│   ├── core.py              # TokenTracker — manages patcher lifecycle
│   ├── patchers/
│   │   ├── base.py          # BasePatcher abstract class
│   │   ├── openai.py        # OpenAI monkey-patch (sync + async + streaming)
│   │   ├── anthropic.py     # Anthropic monkey-patch (sync + async + streaming)
│   │   └── google.py        # Google GenAI monkey-patch
│   ├── storage/
│   │   ├── base.py          # BaseStorage abstract class
│   │   └── sqlite.py        # SQLite WAL-mode, thread-safe, 5-index query optimization
│   ├── pricing.py           # Built-in pricing + fuzzy-match engine (exact → regex → prefix)
│   ├── models.py            # UsageRecord dataclass
│   ├── budget.py            # BudgetConfig + BudgetManager
│   ├── anomaly.py           # AnomalyDetector (Z-score, stdlib only)
│   ├── alerts.py            # AlertSender (fire-and-forget webhook, urllib only)
│   └── cli.py               # Click CLI (tokenmeter / tm)
└── tests/                   # 179 tests, 86% coverage
```

**Design principles:**
- Zero core dependencies — stdlib only for anomaly detection, webhooks, and storage
- True monkey-patching — replaces SDK methods at the class level, intercepts at the response boundary, returns the unmodified response
- WAL-mode SQLite — safe for concurrent writes from multi-threaded async code

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/hidearmoon/token-meter
cd token-meter
pip install -e ".[dev]"
pytest --cov=token_meter       # 179 tests, 86% coverage
ruff check src/ tests/
```

---

## License

MIT © [OpenForge AI](https://github.com/hidearmoon)
