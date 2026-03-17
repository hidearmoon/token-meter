"""Monkey-patch for the OpenAI Python SDK (openai >= 1.0.0).

Patches:
  openai.resources.chat.completions.Completions.create  (sync)
  openai.resources.chat.completions.AsyncCompletions.create  (async)

For streaming, we inject stream_options={'include_usage': True} so the
final chunk contains token counts. We wrap the returned iterator to
capture usage after the stream ends.
"""
from __future__ import annotations

import time
import logging
from typing import Any, Iterator, AsyncIterator

from ..models import UsageRecord
from ..pricing import get_cost
from .base import BasePatcher

logger = logging.getLogger(__name__)


def _extract_usage_sync(response: Any) -> tuple[int, int, int] | None:
    """Extract (input, output, total) tokens from a non-streaming response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    in_tok = getattr(usage, "prompt_tokens", 0) or 0
    out_tok = getattr(usage, "completion_tokens", 0) or 0
    return in_tok, out_tok, in_tok + out_tok


def _extract_model(response: Any) -> str:
    return getattr(response, "model", "") or ""


class OpenAIPatcher(BasePatcher):
    provider = "openai"

    def _do_patch(self) -> bool:
        import openai.resources.chat.completions as _mod  # noqa: PLC0415

        self._module = _mod
        self._original = _mod.Completions.create
        self._original_async = _mod.AsyncCompletions.create

        patcher = self  # capture for closures

        # ── sync wrapper ─────────────────────────────────────────────────
        def _sync_create(self_client, *args, **kwargs):
            stream = kwargs.get("stream", False)
            if stream:
                # Inject include_usage so last chunk carries usage data
                so = dict(kwargs.pop("stream_options", {}) or {})
                so["include_usage"] = True
                kwargs["stream_options"] = so

            start = time.perf_counter()
            response = patcher._original(self_client, *args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000

            if stream:
                return patcher._wrap_sync_stream(response, elapsed, kwargs)

            usage_tuple = _extract_usage_sync(response)
            model = _extract_model(response)
            if usage_tuple:
                patcher._record(*usage_tuple, model, elapsed, is_stream=False)
            return response

        # ── async wrapper ────────────────────────────────────────────────
        async def _async_create(self_client, *args, **kwargs):
            stream = kwargs.get("stream", False)
            if stream:
                so = dict(kwargs.pop("stream_options", {}) or {})
                so["include_usage"] = True
                kwargs["stream_options"] = so

            start = time.perf_counter()
            response = await patcher._original_async(self_client, *args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000

            if stream:
                return patcher._wrap_async_stream(response, elapsed, kwargs)

            usage_tuple = _extract_usage_sync(response)
            model = _extract_model(response)
            if usage_tuple:
                patcher._record(*usage_tuple, model, elapsed, is_stream=False)
            return response

        _mod.Completions.create = _sync_create
        _mod.AsyncCompletions.create = _async_create
        return True

    def _do_unpatch(self) -> None:
        if self._original is not None:
            self._module.Completions.create = self._original
        if self._original_async is not None:
            self._module.AsyncCompletions.create = self._original_async

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

    def _wrap_sync_stream(
        self, stream: Any, start_elapsed: float, kwargs: dict
    ) -> Iterator[Any]:
        """Yield chunks; capture usage from the last chunk."""
        start = time.perf_counter()
        usage_chunk = None
        model = ""
        try:
            for chunk in stream:
                model = model or (getattr(chunk, "model", "") or "")
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    usage_chunk = usage
                yield chunk
        finally:
            elapsed = start_elapsed + (time.perf_counter() - start) * 1000
            if usage_chunk is not None:
                in_tok = getattr(usage_chunk, "prompt_tokens", 0) or 0
                out_tok = getattr(usage_chunk, "completion_tokens", 0) or 0
                self._record(in_tok, out_tok, in_tok + out_tok, model, elapsed, True)

    async def _wrap_async_stream(
        self, stream: Any, start_elapsed: float, kwargs: dict
    ) -> AsyncIterator[Any]:
        """Async version of _wrap_sync_stream."""
        start = time.perf_counter()
        usage_chunk = None
        model = ""
        try:
            async for chunk in stream:
                model = model or (getattr(chunk, "model", "") or "")
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    usage_chunk = usage
                yield chunk
        finally:
            elapsed = start_elapsed + (time.perf_counter() - start) * 1000
            if usage_chunk is not None:
                in_tok = getattr(usage_chunk, "prompt_tokens", 0) or 0
                out_tok = getattr(usage_chunk, "completion_tokens", 0) or 0
                self._record(in_tok, out_tok, in_tok + out_tok, model, elapsed, True)
