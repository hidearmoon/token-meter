# TokenMeter

**追踪每一个 Token，掌控每一分钱。**

轻量级 LLM API 成本与用量可观测 SDK + CLI。一行代码无侵入接入 OpenAI / Anthropic / Google，数据本地存储于 SQLite，开箱即得预算告警和异常检测。

[![CI](https://github.com/hidearmoon/token-meter/actions/workflows/ci.yml/badge.svg)](https://github.com/hidearmoon/token-meter/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/token-meter.svg)](https://pypi.org/project/token-meter/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen.svg)](#)

[English](README.md) | [中文](README_zh.md)

---

- 🔌 **一行接入** — `import token_meter; token_meter.init()` 立即开始追踪
- 🏠 **本地优先** — 所有数据存储在本地 SQLite，不依赖任何 SaaS
- 📊 **全面可视** — 实时仪表盘、预算告警、异常检测、CSV/JSON 导出

---

## 目录

- [快速上手](#快速上手)
- [为什么选择-tokenmeter](#为什么选择-tokenmeter)
- [安装](#安装)
- [支持的供应商](#支持的供应商)
- [SDK-API](#sdk-api)
- [CLI-命令参考](#cli-命令参考)
- [预算告警](#预算告警)
- [异常检测](#异常检测)
- [自定义定价](#自定义定价)
- [配置项](#配置项)
- [架构说明](#架构说明)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

---

## 快速上手

```bash
pip install "token-meter[openai,cli]"
```

```python
import token_meter
token_meter.init(project="my-app")  # ← 只需添加这一行

# 以下是你已有的代码，无需任何改动：
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "你好"}],
)

# 立即查看费用：
stats = token_meter.get_tracker().aggregate()
print(f"费用：${stats['total_cost']:.4f}  |  Token 数：{stats['total_tokens']:,}")
# 费用：$0.0035  |  Token 数：1,234
```

```bash
# 在终端查看实时仪表盘：
tm dashboard
```

---

## 为什么选择 TokenMeter

| 功能 | **TokenMeter** | Helicone | LangSmith | LiteLLM Proxy | token-cost-guard |
|------|:--------------:|:--------:|:---------:|:-------------:|:----------------:|
| Monkey-patch（无需改代码） | ✅ | ❌ | ❌ | ❌ | ❌ |
| 本地 SQLite（不依赖 SaaS） | ✅ | ❌ | ❌ | ✅ | ❌ |
| 预算告警 | ✅ | ✅ | ❌ | ✅ | ❌ |
| 异常检测 | ✅ | ✅ | ❌ | ❌ | ❌ |
| 开源 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 核心零依赖 | ✅ | ❌ | ❌ | ❌ | ❌ |
| CLI 仪表盘 | ✅ | ❌ | ❌ | ✅ | ❌ |
| 流式响应支持 | ✅ | ✅ | ✅ | ✅ | ❌ |

**核心差异化：**
- 唯一将 monkey-patch + 本地存储 + 预算告警 + 异常检测集于一身的零依赖工具包
- Helicone / LangSmith 需要将流量路由到其服务器；TokenMeter 完全不经过网络
- LiteLLM 需要运行独立的代理进程；TokenMeter 是纯进程内库

---

## 安装

```bash
# 仅核心包（不含 provider SDK）：
pip install token-meter

# 按需选择 provider：
pip install "token-meter[openai]"       # 包含 OpenAI SDK
pip install "token-meter[anthropic]"    # 包含 Anthropic SDK
pip install "token-meter[google]"       # 包含 Google GenAI SDK
pip install "token-meter[all]"          # 所有 provider + CLI（rich + click）

# 仅 CLI（SDK 已自行安装时）：
pip install "token-meter[cli]"
```

---

## 支持的供应商

| 供应商 | SDK 要求 | 同步 | 异步 | 流式 |
|--------|---------|:----:|:----:|:----:|
| OpenAI | `openai >= 1.0` | ✅ | ✅ | ✅ |
| Anthropic | `anthropic >= 0.20` | ✅ | ✅ | ✅ |
| Google Gemini | `google-genai >= 1.0` | ✅ | ✅ | ✅ |

**流式响应实现方式：**
- **OpenAI** — 注入 `stream_options={"include_usage": True}`，从最终数据块获取 token 计数
- **Anthropic** — 从 `message_start` 事件累积 `input_tokens`，从 `message_delta` 累积 `output_tokens`
- **Google** — 从最终数据块读取 `usage_metadata`

---

## SDK API

### `token_meter.init()`

```python
import token_meter

tracker = token_meter.init(
    project="my-app",                    # 项目名，用于多项目隔离
    db_path="~/data/usage.db",           # 自定义 SQLite 路径（默认：~/.token-meter/usage.db）
    providers=["openai", "anthropic"],   # 要 patch 的 SDK（默认：全部已安装的）
    budgets={                            # 可选预算限额（美元）
        "daily": 5.00,
        "weekly": 25.00,
        "monthly": 80.00,
    },
    alerts=[                             # 可选 webhook 告警目标
        {"type": "webhook", "url": "https://hooks.slack.com/..."},
    ],
    alert_thresholds=[0.8, 0.9, 1.0],   # 到达预算的 80%/90%/100% 时触发告警
)
```

### 查询数据

```python
tracker = token_meter.get_tracker()

# 聚合统计
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

# 按模型分组统计
rows = tracker.aggregate_by_model()
# [
#   {"provider": "openai", "model": "gpt-4o", "call_count": 30, "total_cost": 0.48},
#   {"provider": "anthropic", "model": "claude-sonnet-4", "call_count": 12, "total_cost": 0.032},
# ]

# 原始调用记录（支持过滤）
records = tracker.query(
    project="my-app",
    provider="openai",
    model="gpt-4o",
    limit=50,
)

# 停止追踪并恢复原始 SDK 方法
token_meter.disable()
```

---

## CLI 命令参考

使用 `pip install "token-meter[cli]"` 安装。支持 `tokenmeter` 和缩写 `tm` 两种调用方式。

### `tm dashboard`

实时显示今日 / 本周 / 本月的消费概览和热门模型。

```
╭─────────────────────── TokenMeter Dashboard ───────────────────────╮
│ 今日：$1.24  │  本周：$8.71  │  本月：$31.05                        │
│ 调用次数：312  │  Token：2.1M  │  平均延迟：843ms                    │
╰────────────────────────────────────────────────────────────────────╯
 热门模型（今日）
 gpt-4o            182 次   $0.98
 claude-sonnet-4    98 次   $0.21
 gpt-4o-mini        32 次   $0.05
```

### `tm summary`

自定义时间范围的聚合统计。

```bash
tm summary --start this-week --group-by model
tm summary --start 2025-01-01 --end 2025-01-31 --project my-app
```

选项：`--start`、`--end`（YYYY-MM-DD 或 `today`/`this-week`/`this-month`）、`--group-by`（model|provider|project|day|week|month）、`--project`

### `tm history`

最近的 LLM 调用记录。

```bash
tm history --limit 50 --provider openai --model gpt-4o
```

选项：`--limit`（默认 20）、`--provider`、`--model`、`--project`、`--start`、`--end`

### `tm export`

导出为 CSV 或 JSON。

```bash
tm export --format json --start this-month --output costs.json
tm export --format csv --project my-app > report.csv
```

选项：`--format`（csv|json）、`--start`、`--end`、`--project`、`--output`

### `tm projects` / `tm models`

```bash
tm projects    # 列出所有项目及总用量
tm models      # 列出所有模型，按总费用排序
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

### 全局选项

```bash
tm --db-path /custom/path/usage.db dashboard   # 为任意命令指定数据库路径
# 或设置环境变量 TOKEN_METER_DB_PATH
```

---

## 预算告警

为每个项目设置消费限额。TokenMeter 在达到配置的阈值比例时（默认 80%、90%、100%）通过 webhook 发送通知。

### 通过 SDK 配置

```python
token_meter.init(
    project="production",
    budgets={"daily": 10.00, "monthly": 200.00},
    alerts=[{"type": "webhook", "url": "https://hooks.slack.com/services/..."}],
    alert_thresholds=[0.8, 0.9, 1.0],
)
```

### 通过 CLI 配置

```bash
tm budget set --project production --daily 10.00 --monthly 200.00
tm alert add --webhook https://hooks.slack.com/services/... --project production
```

### Webhook 消息体

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

Webhook 为非阻塞发送（守护线程，5 秒超时）。发送失败仅记录警告，不会抛出异常影响你的应用。

---

## 异常检测

TokenMeter 对过去 30 天的每日费用计算 **Z-score**，自动标记消费突增——纯标准库实现，无需任何 ML 外部依赖。

**算法：**
1. 按 `(project, model)` 维度收集过去 30 天的每日费用
2. 计算均值和标准差
3. 若 `(今日费用 − 均值) / 标准差 ≥ 阈值`（默认 2.0），触发告警
4. 至少需要 7 天历史数据才会开始检测

每天第一次 API 调用时自动执行检测，也可手动触发：

```bash
tm anomalies check --z-score 2.0 --project my-app
```

### 异常告警消息体

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

## 自定义定价

```python
from token_meter.pricing import add_custom_pricing

add_custom_pricing("my-finetune-v1", input_per_1m=5.00, output_per_1m=20.00)
```

内置定价表（美元 / 100万 token）：

| 模型 | 输入 | 输出 |
|------|-----:|-----:|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4.1 | $2.00 | $8.00 |
| claude-opus-4 | $15.00 | $75.00 |
| claude-sonnet-4 | $3.00 | $15.00 |
| claude-haiku-3.5 | $0.80 | $4.00 |
| gemini-2.5-pro | $1.25 | $10.00 |
| gemini-2.5-flash | $0.30 | $2.50 |

支持模型名模糊匹配（如 `gpt-4o-2024-11-20` 自动映射到 `gpt-4o`）。未知模型记录费用为 $0.00 并打印警告。

---

## 配置项

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `TOKEN_METER_DB_PATH` | SQLite 数据库路径 | `~/.token-meter/usage.db` |
| `TOKEN_METER_PROJECT` | 默认项目名 | `default` |

### `token_meter.init()` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `project` | `str` | `"default"` | 逻辑项目名 |
| `db_path` | `str \| None` | `None` | 自定义 SQLite 路径 |
| `providers` | `list[str] \| None` | `None` | 要 patch 的供应商（`["openai","anthropic","google"]`） |
| `budgets` | `dict \| None` | `None` | 键：`daily`、`weekly`、`monthly`（USD 浮点数） |
| `alerts` | `list[dict] \| None` | `None` | 告警目标：`{"type": "webhook", "url": "..."}` |
| `alert_thresholds` | `list[float] \| None` | `[0.8, 0.9, 1.0]` | 触发通知的预算比例 |

---

## 架构说明

```
token-meter/
├── src/token_meter/
│   ├── __init__.py          # 公开 API: init(), disable(), get_tracker()
│   ├── core.py              # TokenTracker — 管理 patcher 生命周期
│   ├── patchers/
│   │   ├── base.py          # BasePatcher 抽象基类
│   │   ├── openai.py        # OpenAI monkey-patch（同步+异步+流式）
│   │   ├── anthropic.py     # Anthropic monkey-patch（同步+异步+流式）
│   │   └── google.py        # Google GenAI monkey-patch
│   ├── storage/
│   │   ├── base.py          # BaseStorage 抽象基类
│   │   └── sqlite.py        # SQLite WAL 模式，线程安全，5 索引查询优化
│   ├── pricing.py           # 内置定价 + 模糊匹配引擎（精确→正则→前缀）
│   ├── models.py            # UsageRecord 数据模型
│   ├── budget.py            # BudgetConfig + BudgetManager
│   ├── anomaly.py           # AnomalyDetector（Z-score，仅标准库）
│   ├── alerts.py            # AlertSender（非阻塞 webhook，仅 urllib）
│   └── cli.py               # Click CLI（tokenmeter / tm）
└── tests/                   # 179 个测试，86% 覆盖率
```

**设计原则：**
- 核心零依赖——异常检测、webhook、存储均仅使用标准库
- 真正的 monkey-patch——在类级别替换 SDK 方法，在响应边界拦截，返回未修改的原始响应
- WAL 模式 SQLite——对多线程异步代码的并发写入安全

---

## 参与贡献

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

```bash
git clone https://github.com/hidearmoon/token-meter
cd token-meter
pip install -e ".[dev]"
pytest --cov=token_meter       # 179 个测试，86% 覆盖率
ruff check src/ tests/
```

---

## 许可证

MIT © [OpenForge AI](https://github.com/hidearmoon)
