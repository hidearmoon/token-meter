"""TokenMeter — zero-code LLM API cost & usage observability.

Usage::

    import token_meter
    token_meter.init()          # patches OpenAI/Anthropic/Google automatically

    # ... your existing LLM code unchanged ...

    stats = token_meter.get_tracker().aggregate()
    print(f"Total cost today: ${stats['total_cost']:.4f}")

    token_meter.disable()       # restore original SDK methods
"""
from __future__ import annotations

import logging
from typing import List, Optional

from .config import TokenMeterConfig
from .core import TokenTracker
from .storage.sqlite import SQLiteStorage

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

_tracker: Optional[TokenTracker] = None


def init(
    project: str = "default",
    db_path: Optional[str] = None,
    providers: Optional[List[str]] = None,
) -> TokenTracker:
    """Initialise TokenMeter and start tracking LLM API calls.

    Args:
        project:   Logical project name for multi-project isolation.
        db_path:   Custom path for the SQLite database.
                   Defaults to ``~/.token-meter/usage.db``.
                   Can also be set via ``TOKEN_METER_DB_PATH`` env var.
        providers: Which providers to patch.
                   Defaults to all: ``['openai', 'anthropic', 'google']``.
                   Can also be set via ``TOKEN_METER_PROJECT`` env var (project only).

    Returns:
        The active :class:`~token_meter.core.TokenTracker` instance.
    """
    global _tracker

    if _tracker is not None and _tracker.is_active:
        logger.debug("token-meter: already active")
        return _tracker

    config = TokenMeterConfig.from_kwargs(
        project=project,
        db_path=db_path,
        providers=providers,
    )
    storage = SQLiteStorage(config.db_path)
    _tracker = TokenTracker(config, storage)
    _tracker.start()
    return _tracker


def disable() -> None:
    """Stop tracking and restore original SDK methods."""
    global _tracker
    if _tracker is not None:
        _tracker.stop()


def get_tracker() -> Optional[TokenTracker]:
    """Return the active tracker, or None if not initialised."""
    return _tracker


__all__ = [
    "init",
    "disable",
    "get_tracker",
    "TokenTracker",
    "TokenMeterConfig",
    "__version__",
]
