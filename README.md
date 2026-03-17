# token-meter 💰

> Zero-code LLM API cost & usage observability — one line to track every OpenAI / Anthropic / Google call, stored locally in SQLite.

[![CI](https://github.com/hidearmoon/token-meter/actions/workflows/ci.yml/badge.svg)](https://github.com/hidearmoon/token-meter/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/token-meter.svg)](https://badge.fury.io/py/token-meter)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen.svg)](#)

[English](#english) | [中文](#中文)

---

## English

### What is token-meter?

**token-meter** is a lightweight Python SDK that automatically tracks every LLM API call your code makes — token usage, cost, latency, and model breakdown — with **zero code changes** required. Data lives in a local SQLite database; nothing is ever sent to a third party.

```python
import token_meter
token_meter.init()          # ← add this one line

# Your existing code, unchanged:
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)

stats = token_meter.get_tracker().aggregate()
print(f"Total cost today: ${stats['total_cost']:.4f}")
# Total cost today: $0.0035
```

### Why token-meter?

| Tool | Monkey-patch | Local SQLite | Budget alerts | Anomaly detection |
|------|:-----------:|:------------:|:-------------:|:-----------------:|
| **token-meter** | ✅ | ✅ | ✅ | ✅ |
| token-cost-guard | ❌ (wrapper only) | ❌ | ❌ | ❌ |
| OpenLLMetry | ✅ | ❌ (needs OTel backend) | ❌ | ❌ |
| tokentop | ❌ (reads provider API) | ❌ | ❌ | ❌ |
| Helicone / LangSmith | ❌ (SaaS only) | ❌ | ✅ | ✅ |

### Installation

```bash
pip install token-meter

# With specific provider SDKs:
pip install "token-meter[openai]"       # + openai>=1.0
pip install "token-meter[anthropic]"    # + anthropic>=0.20
pip install "token-meter[all]"          # all providers + CLI
```

### Quick Start

```python
import token_meter

# Minimal — patches all available providers automatically
token_meter.init()

# Full configuration
token_meter.init(
    project="my-chatbot",           # multi-project isolation
    db_path="~/data/usage.db",      # custom SQLite path
    providers=["openai", "anthropic"],  # only patch these
)

# ── After your LLM calls ──────────────────────────────────────────────────────

tracker = token_meter.get_tracker()

# Aggregate stats
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
# [{"provider": "openai", "model": "gpt-4o", "call_count": 30, "total_cost": 0.48}, ...]

# Raw records
records = tracker.query(project="my-chatbot", provider="openai", limit=10)

# Stop tracking / restore original SDK methods
token_meter.disable()
```

### Supported Providers & Models

| Provider | SDK | Streaming |
|----------|-----|:---------:|
| OpenAI | `openai >= 1.0` | ✅ |
| Anthropic | `anthropic >= 0.20` | ✅ |
| Google | `google-genai >= 1.0` | ✅ |

Built-in pricing (USD / 1M tokens, update via `token_meter.pricing.add_custom_pricing()`):

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

Unknown models are recorded with cost = $0.00 and a warning is emitted.

### How it works

token-meter uses **true monkey-patching** — it replaces the `create` method on the SDK's internal class at import time, intercepts every call, extracts usage metadata from the response, then hands the **unmodified** response back to your code. No wrappers, no proxy classes, no changes to your call sites.

For streaming responses:
- **OpenAI**: injects `stream_options={"include_usage": True}` so the final chunk carries token counts
- **Anthropic**: accumulates `input_tokens` from `message_start` and `output_tokens` from `message_delta` events
- **Google**: reads `usage_metadata` from the final chunk

### Custom pricing

```python
from token_meter.pricing import add_custom_pricing

add_custom_pricing("my-finetune-v1", input_per_1m=5.00, output_per_1m=20.00)
```

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TOKEN_METER_DB_PATH` | SQLite database path | `~/.token-meter/usage.db` |
| `TOKEN_METER_PROJECT` | Default project name | `default` |

### Development

```bash
git clone https://github.com/hidearmoon/token-meter
cd token-meter
pip install -e ".[dev]"
pytest --cov=token_meter       # 86% coverage, 99 tests
```

---

## 中文

### token-meter 是什么？

**token-meter** 是一个轻量级 Python SDK，用于自动追踪所有 LLM API 调用的 Token 用量和费用，**无需修改任何现有代码**。数据存储在本地 SQLite 数据库中，不会向任何第三方发送数据。

### 核心优势

相比市场上其他工具：

- **真正的 monkey-patch**：不是 wrapper 模式，无需修改调用代码
- **本地优先**：数据存 SQLite，完全私有，不依赖 SaaS
- **零依赖**：核心包无任何外部依赖（仅标准库）
- **全面覆盖**：同时支持 OpenAI / Anthropic / Google，包括流式响应

### 快速上手（30 秒）

```bash
pip install token-meter
```

```python
import token_meter
token_meter.init(project="我的应用")

# 以下是你已有的代码，无需任何改动：
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "你好"}]
)

# 查看消费统计
stats = token_meter.get_tracker().aggregate()
print(f"今日总费用：${stats['total_cost']:.4f}")
print(f"总调用次数：{stats['call_count']}")
print(f"总 Token 数：{stats['total_tokens']:,}")
```

### 完整 API

```python
import token_meter

# 初始化（自动 patch 所有已安装的 SDK）
token_meter.init(
    project="my-app",               # 项目名，用于多项目隔离
    db_path="~/custom/usage.db",    # 自定义数据库路径
    providers=["openai"],           # 只 patch 指定的 provider
)

tracker = token_meter.get_tracker()

# 聚合统计
stats = tracker.aggregate(project="my-app")
# 返回: call_count, total_tokens, total_cost, avg_latency_ms, ...

# 按模型分组统计
rows = tracker.aggregate_by_model()
# 返回: [{provider, model, call_count, total_tokens, total_cost}, ...]

# 查询原始记录
records = tracker.query(
    project="my-app",
    provider="openai",
    model="gpt-4o",
    limit=100,
)

# 停止追踪，恢复原始 SDK 方法
token_meter.disable()
```

### 支持的模型与定价

| 模型 | 输入价格 (per 1M token) | 输出价格 (per 1M token) |
|------|----------------------:|----------------------:|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4.1 | $2.00 | $8.00 |
| claude-opus-4 | $15.00 | $75.00 |
| claude-sonnet-4 | $3.00 | $15.00 |
| claude-haiku-3.5 | $0.80 | $4.00 |
| gemini-2.5-pro | $1.25 | $10.00 |
| gemini-2.5-flash | $0.30 | $2.50 |

支持模型名模糊匹配（如 `gpt-4o-2024-11-20` 自动映射到 `gpt-4o`）。未知模型记录为 $0.00 并打印警告。

### 自定义定价

```python
from token_meter.pricing import add_custom_pricing

# 注册自定义/微调模型的定价
add_custom_pricing("my-finetune-v1", input_per_1m=5.00, output_per_1m=20.00)
```

### 技术架构

```
token-meter/
├── src/token_meter/
│   ├── __init__.py          # 公开 API: init(), disable(), get_tracker()
│   ├── core.py              # TokenTracker 核心类，管理 patch 生命周期
│   ├── patchers/
│   │   ├── base.py          # BasePatcher 抽象基类
│   │   ├── openai.py        # OpenAI SDK monkey-patch（同步+异步+流式）
│   │   ├── anthropic.py     # Anthropic SDK monkey-patch
│   │   └── google.py        # Google GenAI SDK monkey-patch
│   ├── storage/
│   │   ├── base.py          # BaseStorage 抽象基类
│   │   └── sqlite.py        # SQLite 存储（WAL 模式，线程安全）
│   ├── pricing.py           # 内置定价数据 + 模糊匹配引擎
│   ├── models.py            # UsageRecord 数据模型
│   └── config.py            # 配置管理（支持环境变量）
```

### 开发指南

```bash
git clone https://github.com/hidearmoon/token-meter
cd token-meter
pip install -e ".[dev]"

# 运行测试
pytest --cov=token_meter

# 当前覆盖率：86%，99 个测试用例
```

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `TOKEN_METER_DB_PATH` | SQLite 数据库路径 | `~/.token-meter/usage.db` |
| `TOKEN_METER_PROJECT` | 默认项目名 | `default` |

---

## License

MIT © [OpenForge AI](https://github.com/hidearmoon)
