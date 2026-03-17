"""Budget management and threshold alerting for TokenMeter.

Usage via SDK::

    import token_meter
    token_meter.init(
        project="my-app",
        budgets={"daily": 10.0, "weekly": 50.0, "monthly": 200.0},
        alerts=[{"type": "webhook", "url": "https://hooks.slack.com/..."}],
        alert_thresholds=[0.8, 0.9, 1.0],
    )

Usage via CLI::

    tm budget set --project my-app --daily 10 --weekly 50 --monthly 200
    tm alert add --webhook https://hooks.slack.com/...
    tm budget status
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .alerts import AlertSender
    from .models import UsageRecord
    from .storage.sqlite import SQLiteStorage

logger = logging.getLogger(__name__)

_PERIODS = ("daily", "weekly", "monthly")


@dataclass
class BudgetConfig:
    """Per-project budget limits and alert settings."""

    project: str = "default"
    daily: Optional[float] = None
    weekly: Optional[float] = None
    monthly: Optional[float] = None
    thresholds: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.0])
    webhook_urls: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "daily": self.daily,
            "weekly": self.weekly,
            "monthly": self.monthly,
            "thresholds": self.thresholds,
            "webhook_urls": self.webhook_urls,
        }

    @classmethod
    def from_dict(cls, project: str, data: Dict[str, Any]) -> "BudgetConfig":
        return cls(
            project=project,
            daily=data.get("daily"),
            weekly=data.get("weekly"),
            monthly=data.get("monthly"),
            thresholds=data.get("thresholds", [0.8, 0.9, 1.0]),
            webhook_urls=data.get("webhook_urls", []),
        )


def _period_key(period: str, ref: date) -> str:
    """Return a string key uniquely identifying the current period window."""
    if period == "daily":
        return ref.isoformat()
    if period == "weekly":
        monday = ref - timedelta(days=ref.weekday())
        return monday.isoformat()
    if period == "monthly":
        return f"{ref.year}-{ref.month:02d}"
    return ref.isoformat()


def _period_start(period: str, ref: date) -> datetime:
    """Return the UTC start of the current period for *ref* date."""
    if period == "daily":
        return datetime(ref.year, ref.month, ref.day, tzinfo=timezone.utc)
    if period == "weekly":
        monday = ref - timedelta(days=ref.weekday())
        return datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)
    if period == "monthly":
        return datetime(ref.year, ref.month, 1, tzinfo=timezone.utc)
    return datetime(ref.year, ref.month, ref.day, tzinfo=timezone.utc)


class BudgetManager:
    """Checks budget thresholds after each usage record is saved.

    Register with :meth:`~token_meter.storage.sqlite.SQLiteStorage.register_post_save_hook`
    so it runs automatically on every tracked API call.
    """

    def __init__(
        self,
        configs: List[BudgetConfig],
        storage: "SQLiteStorage",
        sender: "AlertSender",
    ) -> None:
        self._configs: Dict[str, BudgetConfig] = {c.project: c for c in configs}
        self._storage = storage
        self._sender = sender

    def add_config(self, config: BudgetConfig) -> None:
        """Add or replace a budget config and persist it to SQLite."""
        self._configs[config.project] = config
        self._storage.set_budget_config(config.project, config.to_dict())

    def get_config(self, project: str) -> Optional[BudgetConfig]:
        return self._configs.get(project)

    def check(self, record: "UsageRecord") -> None:
        """Post-save hook: check if any budget threshold is now exceeded."""
        project = record.project
        config = self._configs.get(project) or self._configs.get("default")
        if config is None:
            return

        today = datetime.now(timezone.utc).date()

        for period in _PERIODS:
            limit: Optional[float] = getattr(config, period)
            if limit is None or limit <= 0:
                continue

            start = _period_start(period, today)
            spend = self._storage.get_period_spend(project, start)
            pct = spend / limit
            pkey = _period_key(period, today)
            top_models = self._storage.get_top_models(project, start, limit=5)

            for threshold in sorted(config.thresholds):
                if pct >= threshold:
                    if not self._storage.has_budget_alert(
                        project, period, threshold, pkey
                    ):
                        payload: Dict[str, Any] = {
                            "alert": "budget_threshold",
                            "project": project,
                            "period": period,
                            "threshold": threshold,
                            "current_spend": round(spend, 6),
                            "budget_limit": limit,
                            "percentage": round(pct * 100, 2),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "top_models": top_models,
                        }
                        self._storage.log_budget_alert(
                            project, period, threshold, pkey, payload
                        )
                        for url in config.webhook_urls:
                            self._sender.send_webhook(url, payload)
                        logger.warning(
                            "token-meter: budget alert %s %s %.0f%% ($%.4f / $%.2f)",
                            project,
                            period,
                            threshold * 100,
                            spend,
                            limit,
                        )
