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
from typing import Any, Dict, List, Optional

from .config import TokenMeterConfig
from .core import TokenTracker
from .storage.sqlite import SQLiteStorage

__version__ = "0.1.1"

logger = logging.getLogger(__name__)

_tracker: Optional[TokenTracker] = None


def init(
    project: str = "default",
    db_path: Optional[str] = None,
    providers: Optional[List[str]] = None,
    budgets: Optional[Dict[str, float]] = None,
    alerts: Optional[List[Dict[str, str]]] = None,
    alert_thresholds: Optional[List[float]] = None,
) -> TokenTracker:
    """Initialise TokenMeter and start tracking LLM API calls.

    Args:
        project:           Logical project name for multi-project isolation.
        db_path:           Custom path for the SQLite database.
                           Defaults to ``~/.token-meter/usage.db``.
                           Can also be set via ``TOKEN_METER_DB_PATH`` env var.
        providers:         Which providers to patch.
                           Defaults to all: ``['openai', 'anthropic', 'google']``.
        budgets:           Optional budget limits (USD).  Dict with any of
                           ``daily``, ``weekly``, ``monthly`` keys.
        alerts:            List of alert destinations.  Each item is a dict
                           with ``type`` (currently only ``'webhook'``) and
                           ``url`` keys.
        alert_thresholds:  Fraction of budget that triggers a notification.
                           Defaults to ``[0.8, 0.9, 1.0]``.

    Returns:
        The active :class:`~token_meter.core.TokenTracker` instance.

    Example::

        import token_meter
        token_meter.init(
            project="my-app",
            budgets={"daily": 10.0, "weekly": 50.0, "monthly": 200.0},
            alerts=[{"type": "webhook", "url": "https://hooks.slack.com/..."}],
            alert_thresholds=[0.8, 0.9, 1.0],
        )
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

    if budgets or alerts:
        from .alerts import AlertSender
        from .anomaly import AnomalyDetector
        from .budget import BudgetConfig, BudgetManager

        sender = AlertSender()
        webhook_urls = [
            a["url"]
            for a in (alerts or [])
            if a.get("type") == "webhook" and a.get("url")
        ]
        thresholds = alert_thresholds or [0.8, 0.9, 1.0]

        cfg = BudgetConfig(
            project=project,
            daily=budgets.get("daily") if budgets else None,
            weekly=budgets.get("weekly") if budgets else None,
            monthly=budgets.get("monthly") if budgets else None,
            thresholds=thresholds,
            webhook_urls=webhook_urls,
        )
        budget_manager = BudgetManager([cfg], storage, sender)
        storage.set_budget_config(project, cfg.to_dict())

        detector = AnomalyDetector(storage, sender, webhook_urls=webhook_urls)

        storage.register_post_save_hook(budget_manager.check)
        storage.register_post_save_hook(detector.check_on_first_write)

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
    "AlertSender",
    "BudgetConfig",
    "BudgetManager",
    "AnomalyDetector",
]
