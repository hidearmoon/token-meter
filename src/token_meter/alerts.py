"""Alert sender for TokenMeter — non-blocking webhook notifications.

Uses stdlib ``urllib.request`` exclusively; no external HTTP library required.
Each webhook is dispatched on a daemon thread so alert delivery never blocks
the hot path of the SDK interceptor.
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Any, Dict

logger = logging.getLogger(__name__)


class AlertSender:
    """Sends HTTP POST webhooks in background daemon threads (fire-and-forget).

    Example::

        sender = AlertSender()
        sender.send_webhook("https://hooks.slack.com/...", {"text": "Budget alert!"})
    """

    def __init__(self, timeout: int = 5) -> None:
        self._timeout = timeout

    def send_webhook(self, url: str, payload: Dict[str, Any]) -> None:
        """Schedule a non-blocking HTTP POST to *url* with JSON *payload*.

        Returns immediately; the actual HTTP request happens on a daemon thread.
        Failures are logged as warnings but never raise.
        """
        t = threading.Thread(
            target=self._post,
            args=(url, payload),
            daemon=True,
            name="token-meter-alert",
        )
        t.start()

    def _post(self, url: str, payload: Dict[str, Any]) -> None:
        """Blocking helper executed on the background thread."""
        try:
            data = json.dumps(payload, default=str).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "token-meter/0.1",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                status = resp.status
                if status >= 400:
                    logger.warning(
                        "token-meter: webhook %s returned HTTP %d", url, status
                    )
        except urllib.error.URLError as exc:
            logger.warning("token-meter: webhook %s failed: %s", url, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("token-meter: webhook %s error: %s", url, exc)
