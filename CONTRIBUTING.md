# Contributing to TokenMeter

Thank you for your interest in contributing! TokenMeter is a project by [OpenForge AI](https://github.com/hidearmoon) focused on LLM observability tooling.

## Getting started

```bash
git clone https://github.com/hidearmoon/token-meter
cd token-meter
pip install -e ".[dev]"
```

This installs the package in editable mode along with all development dependencies (pytest, pytest-cov, pytest-asyncio, ruff, plus the OpenAI and Anthropic SDKs for integration tests).

## Running tests

```bash
pytest --cov=token_meter -v
```

The test suite targets ≥ 80% coverage. CI runs across Python 3.9, 3.10, 3.11, and 3.12.

## Linting

```bash
ruff check src/ tests/
```

We follow standard ruff defaults. Fix any errors before opening a PR.

## Project structure

```
src/token_meter/
├── patchers/       # Per-provider monkey-patches (openai, anthropic, google)
├── storage/        # SQLite storage layer
├── pricing.py      # Built-in pricing table + fuzzy model matching
├── models.py       # UsageRecord dataclass
├── budget.py       # Budget limits and alerting
├── anomaly.py      # Z-score anomaly detection
├── alerts.py       # Webhook delivery
└── cli.py          # Click CLI (tokenmeter / tm)
```

## Adding a new provider

1. Create `src/token_meter/patchers/<provider>.py` subclassing `BasePatcher`
2. Implement `patch()` and `unpatch()` to monkey-patch the SDK's internal `create` (or equivalent) method
3. Extract `input_tokens`, `output_tokens`, model name, and latency from the response
4. Handle streaming by accumulating token counts from intermediate chunks
5. Add unit tests in `tests/test_<provider>_patcher.py`
6. Register the new patcher in `src/token_meter/core.py`

## Adding new pricing entries

Edit `src/token_meter/pricing.py`. Pricing is stored as `USD per 1 million tokens`. Follow the existing format and add both input and output prices.

## Submitting a PR

- Keep PRs focused: one feature or fix per PR
- Include tests for new behaviour
- Update the relevant section of `README.md` if the public API or CLI changes
- All CI checks must pass

## Reporting bugs

Open an issue at https://github.com/hidearmoon/token-meter/issues. Include your Python version, SDK versions, and a minimal reproducible example.
