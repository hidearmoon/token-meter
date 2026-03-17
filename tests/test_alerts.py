"""Tests for AlertSender — non-blocking webhook dispatch."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from token_meter.alerts import AlertSender


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

class _CaptureHandler(BaseHTTPRequestHandler):
    """Minimal HTTP server that captures POST bodies."""

    captured: List[bytes] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _CaptureHandler.captured.append(body)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args) -> None:  # silence access log
        pass


# ------------------------------------------------------------------ #
# send_webhook — basic behaviour                                        #
# ------------------------------------------------------------------ #

class TestSendWebhook:
    def test_returns_immediately(self) -> None:
        """send_webhook must not block the caller."""
        sender = AlertSender(timeout=1)
        barrier = threading.Event()

        # Patch _post so it blocks until we release
        original_post = sender._post

        def slow_post(url, payload):
            barrier.wait(timeout=2)

        sender._post = slow_post  # type: ignore[method-assign]

        start = time.perf_counter()
        sender.send_webhook("http://example.com/hook", {"key": "value"})
        elapsed_ms = (time.perf_counter() - start) * 1000

        barrier.set()  # let the background thread finish
        assert elapsed_ms < 100, "send_webhook should return in < 100 ms"

    def test_sends_json_payload(self) -> None:
        """The HTTP POST body must be valid JSON matching the payload."""
        _CaptureHandler.captured.clear()

        server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        sender = AlertSender(timeout=3)
        payload = {"alert": "budget_threshold", "project": "test", "percentage": 80.5}
        sender.send_webhook(f"http://127.0.0.1:{port}/hook", payload)

        t.join(timeout=3)
        server.server_close()

        assert len(_CaptureHandler.captured) == 1
        body = json.loads(_CaptureHandler.captured[0])
        assert body["alert"] == "budget_threshold"
        assert body["percentage"] == 80.5

    def test_failure_does_not_raise(self) -> None:
        """A failed webhook must log a warning but never raise."""
        sender = AlertSender(timeout=1)
        # Bad URL — should just log warning
        sender.send_webhook("http://127.0.0.1:1/unreachable", {"x": 1})
        # Give the daemon thread time to try and fail
        time.sleep(0.3)
        # No exception means pass

    def test_daemon_thread(self) -> None:
        """Background thread must be a daemon so it doesn't block process exit."""
        threads_before = set(t.ident for t in threading.enumerate())
        sender = AlertSender(timeout=1)

        barrier = threading.Barrier(2)
        started = threading.Event()

        def slow_post(url, payload):
            started.set()
            barrier.wait(timeout=2)

        sender._post = slow_post  # type: ignore[method-assign]
        sender.send_webhook("http://example.com/hook", {})

        started.wait(timeout=2)
        new_threads = [t for t in threading.enumerate() if t.ident not in threads_before]
        alert_threads = [t for t in new_threads if "token-meter-alert" in t.name]
        assert all(t.daemon for t in alert_threads), "alert threads must be daemons"
        barrier.wait(timeout=2)

    def test_content_type_header(self) -> None:
        """Webhook POST must include Content-Type: application/json."""
        received_headers: List[dict] = []

        class _HeaderCapture(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                received_headers.append(dict(self.headers))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args) -> None:
                pass

        server = HTTPServer(("127.0.0.1", 0), _HeaderCapture)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        sender = AlertSender(timeout=3)
        sender.send_webhook(f"http://127.0.0.1:{port}/hook", {"x": 1})
        t.join(timeout=3)
        server.server_close()

        assert len(received_headers) == 1
        ct = received_headers[0].get("Content-Type", "")
        assert "application/json" in ct

    def test_http_4xx_logs_warning(self, caplog) -> None:
        """A 4xx response should log a warning, not raise."""
        import logging

        class _Error400Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                self.send_response(400)
                self.end_headers()

            def log_message(self, *args) -> None:
                pass

        server = HTTPServer(("127.0.0.1", 0), _Error400Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        sender = AlertSender(timeout=3)
        with caplog.at_level(logging.WARNING, logger="token_meter.alerts"):
            sender.send_webhook(f"http://127.0.0.1:{port}/hook", {"x": 1})
            t.join(timeout=3)
            time.sleep(0.1)

        server.server_close()
        assert any("400" in r.message for r in caplog.records), (
            "Expected a warning mentioning HTTP 400"
        )
