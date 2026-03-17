"""Monkey-patch for the Anthropic Python SDK (anthropic >= 0.20.0).

Patches:
  anthropic.resources.messages.Messages.create      (sync)
  anthropic.resources.messages.AsyncMessages.create (async)
  anthropic.resources.messages.Messages.stream      (sync context manager, if present)
  anthropic.resources.messages.AsyncMessages.stream (async context manager, if present)

For streaming responses the SDK returns a context-manager/iterator.
We wrap it to intercept the SSE events and accumulate token counts:
  - message_start  → input_tokens
  - message_delta  → output_tokens (usage.output_tokens field)

For the ``client.messages.stream(...)`` context-manager pattern we wrap the
returned ``MessageStreamManager`` so we can call ``get_final_message()`` on
exit and record usage from the fully-assembled message object.
"""
from __future__ import annotations

import time
import logging
from typing import Any, Iterator, AsyncIterator

from ..models import UsageRecord
from ..pricing import get_cost
from .base import BasePatcher

logger = logging.getLogger(__name__)


class _SyncStreamManagerWrapper:
    """Wrap a ``MessageStreamManager`` to capture usage on context-manager exit."""

    def __init__(self, manager: Any, patcher: "AnthropicPatcher", start_time: float) -> None:
        self._manager = manager
        self._patcher = patcher
        self._start_time = start_time
        self._stream: Any = None

    def __enter__(self) -> Any:
        self._stream = self._manager.__enter__()
        return self._stream

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        result = self._manager.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is None and self._stream is not None:
            try:
                msg = self._stream.get_final_message()
                elapsed = (time.perf_counter() - self._start_time) * 1000
                in_tok = getattr(getattr(msg, "usage", None), "input_tokens", 0) or 0
                out_tok = getattr(getattr(msg, "usage", None), "output_tokens", 0) or 0
                model = getattr(msg, "model", "") or ""
                if in_tok or out_tok:
                    self._patcher._record(in_tok, out_tok, in_tok + out_tok, model, elapsed, True)
            except Exception:
                logger.debug("token-meter: failed to record Anthropic stream() usage", exc_info=True)
        return result


class _AsyncStreamManagerWrapper:
    """Async variant of ``_SyncStreamManagerWrapper``."""

    def __init__(self, manager: Any, patcher: "AnthropicPatcher", start_time: float) -> None:
        self._manager = manager
        self._patcher = patcher
        self._start_time = start_time
        self._stream: Any = None

    async def __aenter__(self) -> Any:
        self._stream = await self._manager.__aenter__()
        return self._stream

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        result = await self._manager.__aexit__(exc_type, exc_val, exc_tb)
        if exc_type is None and self._stream is not None:
            try:
                msg = self._stream.get_final_message()
                elapsed = (time.perf_counter() - self._start_time) * 1000
                in_tok = getattr(getattr(msg, "usage", None), "input_tokens", 0) or 0
                out_tok = getattr(getattr(msg, "usage", None), "output_tokens", 0) or 0
                model = getattr(msg, "model", "") or ""
                if in_tok or out_tok:
                    self._patcher._record(in_tok, out_tok, in_tok + out_tok, model, elapsed, True)
            except Exception:
                logger.debug("token-meter: failed to record Anthropic async stream() usage", exc_info=True)
        return result


def _extract_usage_sync(response: Any) -> tuple[int, int, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    in_tok = getattr(usage, "input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", 0) or 0
    return in_tok, out_tok, in_tok + out_tok


def _extract_model(response: Any) -> str:
    return getattr(response, "model", "") or ""


class AnthropicPatcher(BasePatcher):
    provider = "anthropic"

    def _do_patch(self) -> bool:
        import anthropic.resources.messages as _mod  # noqa: PLC0415

        self._module = _mod
        self._original = _mod.Messages.create
        self._original_async = _mod.AsyncMessages.create
        # stream() is a context-manager helper added in newer SDK versions
        self._original_stream = getattr(_mod.Messages, "stream", None)
        self._original_async_stream = getattr(_mod.AsyncMessages, "stream", None)

        patcher = self

        def _sync_create(self_client, *args, **kwargs):
            stream = kwargs.get("stream", False)
            start = time.perf_counter()
            response = patcher._original(self_client, *args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000

            if stream:
                return patcher._wrap_sync_stream(response, elapsed)

            usage_tuple = _extract_usage_sync(response)
            model = _extract_model(response)
            if usage_tuple:
                patcher._record(*usage_tuple, model, elapsed, is_stream=False)
            return response

        async def _async_create(self_client, *args, **kwargs):
            stream = kwargs.get("stream", False)
            start = time.perf_counter()
            response = await patcher._original_async(self_client, *args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000

            if stream:
                return patcher._wrap_async_stream(response, elapsed)

            usage_tuple = _extract_usage_sync(response)
            model = _extract_model(response)
            if usage_tuple:
                patcher._record(*usage_tuple, model, elapsed, is_stream=False)
            return response

        _mod.Messages.create = _sync_create
        _mod.AsyncMessages.create = _async_create

        # Patch the context-manager stream() helper only if it exists
        if self._original_stream is not None:
            def _sync_stream(self_client, *args, **kwargs):
                start = time.perf_counter()
                manager = patcher._original_stream(self_client, *args, **kwargs)
                return _SyncStreamManagerWrapper(manager, patcher, start)

            _mod.Messages.stream = _sync_stream

        if self._original_async_stream is not None:
            def _async_stream(self_client, *args, **kwargs):
                start = time.perf_counter()
                manager = patcher._original_async_stream(self_client, *args, **kwargs)
                return _AsyncStreamManagerWrapper(manager, patcher, start)

            _mod.AsyncMessages.stream = _async_stream

        return True

    def _do_unpatch(self) -> None:
        if self._original is not None:
            self._module.Messages.create = self._original
        if self._original_async is not None:
            self._module.AsyncMessages.create = self._original_async
        if self._original_stream is not None:
            self._module.Messages.stream = self._original_stream
        if self._original_async_stream is not None:
            self._module.AsyncMessages.stream = self._original_async_stream

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _record(
        self,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        model: str,
        latency_ms: float,
        is_stream: bool,
    ) -> None:
        in_cost, out_cost, total_cost = get_cost(
            model, input_tokens, output_tokens,
            warn_unknown=self._config.warn_unknown_models,
        )
        record = UsageRecord(
            provider=self.provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=in_cost,
            output_cost=out_cost,
            total_cost=total_cost,
            latency_ms=latency_ms,
            project=self._config.project,
            is_stream=is_stream,
        )
        self._storage.save(record)

    def _wrap_sync_stream(self, stream: Any, start_elapsed: float) -> Any:
        """
        Anthropic streams can be context-managers (MessageStreamManager)
        or plain iterables. We handle the iterable case here.
        For the context-manager pattern, users typically use `with client.messages.stream()`
        which is a different method — `create(stream=True)` returns a raw SSE stream.
        """
        start = time.perf_counter()
        in_tok = 0
        out_tok = 0
        model = ""

        try:
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "message_start":
                    msg = getattr(event, "message", None)
                    if msg:
                        model = model or (getattr(msg, "model", "") or "")
                        usage = getattr(msg, "usage", None)
                        if usage:
                            in_tok = getattr(usage, "input_tokens", 0) or 0
                elif event_type == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage:
                        out_tok = getattr(usage, "output_tokens", 0) or 0
                yield event
        finally:
            elapsed = start_elapsed + (time.perf_counter() - start) * 1000
            if in_tok or out_tok:
                self._record(
                    in_tok, out_tok, in_tok + out_tok, model, elapsed, True
                )

    async def _wrap_async_stream(self, stream: Any, start_elapsed: float) -> Any:
        start = time.perf_counter()
        in_tok = 0
        out_tok = 0
        model = ""

        try:
            async for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "message_start":
                    msg = getattr(event, "message", None)
                    if msg:
                        model = model or (getattr(msg, "model", "") or "")
                        usage = getattr(msg, "usage", None)
                        if usage:
                            in_tok = getattr(usage, "input_tokens", 0) or 0
                elif event_type == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage:
                        out_tok = getattr(usage, "output_tokens", 0) or 0
                yield event
        finally:
            elapsed = start_elapsed + (time.perf_counter() - start) * 1000
            if in_tok or out_tok:
                self._record(
                    in_tok, out_tok, in_tok + out_tok, model, elapsed, True
                )
