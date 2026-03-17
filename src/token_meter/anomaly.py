"""Cost anomaly detection for TokenMeter.

Uses Z-score on a rolling 30-day window of daily costs.  No external ML
libraries required — only the stdlib ``statistics`` module.

Detection is triggered automatically on the first SDK write of each day
(checks yesterday's data), and can also be run manually via CLI::

    tm anomalies check [--project my-app] [--z-score 2.5]
    tm anomalies [--days 30] [--project my-app]
"""
from __future__ import annotations

import logging
import statistics
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .alerts import AlertSender
    from .models import UsageRecord
    from .storage.sqlite import SQLiteStorage

logger = logging.getLogger(__name__)

_MIN_HISTORY_DAYS = 7   # Need at least this many prior days before detecting
_WINDOW_DAYS = 30       # Rolling window for mean/std calculation


class AnomalyDetector:
    """Detects cost spikes using Z-score on rolling daily cost history.

    A project+model pair is flagged when::

        z_score = (today_cost - rolling_mean) / rolling_std  >=  z_threshold

    Anomalies are written to the ``anomalies`` SQLite table and optionally
    dispatched to webhook URLs.
    """

    def __init__(
        self,
        storage: "SQLiteStorage",
        sender: "AlertSender",
        webhook_urls: Optional[List[str]] = None,
        z_threshold: float = 2.0,
        window_days: int = _WINDOW_DAYS,
    ) -> None:
        self._storage = storage
        self._sender = sender
        self._webhook_urls: List[str] = webhook_urls or []
        self._z_threshold = z_threshold
        self._window_days = window_days
        self._last_check_date: Optional[date] = None

    # ------------------------------------------------------------------ #
    # Post-save hook                                                       #
    # ------------------------------------------------------------------ #

    def check_on_first_write(self, record: "UsageRecord") -> None:
        """Post-save hook: run yesterday's detection on the first write of a new day.

        Subsequent writes on the same day are ignored to keep the hot path fast.
        Failures are swallowed so anomaly detection never disrupts tracking.
        """
        today = datetime.now(timezone.utc).date()
        if self._last_check_date == today:
            return
        self._last_check_date = today
        try:
            self.check_yesterday()
        except Exception:  # noqa: BLE001
            logger.exception("token-meter: anomaly detection failed (non-fatal)")

    # ------------------------------------------------------------------ #
    # Public detection entry points                                        #
    # ------------------------------------------------------------------ #

    def check_yesterday(
        self,
        project: Optional[str] = None,
        z_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Detect anomalies for yesterday.  Returns list of anomaly dicts."""
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        return self._check_date(yesterday, project=project, z_threshold=z_threshold)

    def check_date(
        self,
        target_date: date,
        project: Optional[str] = None,
        z_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Detect anomalies for an arbitrary *target_date*.  Useful for back-fill."""
        return self._check_date(target_date, project=project, z_threshold=z_threshold)

    # ------------------------------------------------------------------ #
    # Internal implementation                                              #
    # ------------------------------------------------------------------ #

    def _check_date(
        self,
        target_date: date,
        project: Optional[str],
        z_threshold: Optional[float],
    ) -> List[Dict[str, Any]]:
        zth = z_threshold if z_threshold is not None else self._z_threshold
        anomalies: List[Dict[str, Any]] = []

        combos = self._storage.get_project_model_combos(project)
        for proj, model in combos:
            anomaly = self._check_combo(proj, model, target_date, zth)
            if anomaly:
                anomalies.append(anomaly)

        return anomalies

    def _check_combo(
        self,
        project: str,
        model: str,
        target_date: date,
        z_threshold: float,
    ) -> Optional[Dict[str, Any]]:
        """Check a single project+model pair.  Returns anomaly dict or None."""
        window_start = target_date - timedelta(days=self._window_days)
        history_end = target_date - timedelta(days=1)

        history = self._storage.get_daily_costs(
            project=project,
            model=model,
            start_date=window_start,
            end_date=history_end,
        )

        if len(history) < _MIN_HISTORY_DAYS:
            return None  # Not enough data yet

        costs = [h["daily_cost"] for h in history]
        mean = statistics.mean(costs)

        if len(costs) < 2:
            return None

        try:
            std = statistics.stdev(costs)
        except statistics.StatisticsError:
            return None

        if std == 0:
            return None  # Zero variance — skip to avoid division by zero

        # Get target date's cost
        target_rows = self._storage.get_daily_costs(
            project=project,
            model=model,
            start_date=target_date,
            end_date=target_date,
        )
        if not target_rows:
            return None

        daily_cost = target_rows[0]["daily_cost"]
        z_score = (daily_cost - mean) / std

        if z_score < z_threshold:
            return None

        anomaly: Dict[str, Any] = {
            "project": project,
            "model": model,
            "date": target_date.isoformat(),
            "daily_cost": round(daily_cost, 6),
            "rolling_avg": round(mean, 6),
            "rolling_std": round(std, 6),
            "z_score": round(z_score, 4),
        }

        self._storage.save_anomaly(anomaly)

        payload: Dict[str, Any] = {
            "alert": "cost_anomaly",
            **anomaly,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for url in self._webhook_urls:
            self._sender.send_webhook(url, payload)

        logger.warning(
            "token-meter: cost anomaly %s/%s %s cost=$%.4f z=%.2f",
            project,
            model,
            target_date,
            daily_cost,
            z_score,
        )
        return anomaly
