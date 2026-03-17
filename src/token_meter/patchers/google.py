"""Monkey-patch for the Google GenAI Python SDK (google-genai >= 1.0.0).

Patches:
  google.genai.models.Models.generate_content  (sync)
  google.genai.models.AsyncModels.generate_content  (async)

For streaming, the final chunk's usage_metadata contains complete counts.
"""
from __future__ import annotations

import time
import logging
from typing import Any, Iterator, AsyncIterator

from ..models import UsageRecord
from ..pricing import get_cost
from .base import BasePatcher

logger = logging.getLogger(__name__)


def _extract_usage(response: Any) -> tuple[int, int, int] | None:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None
    in_tok = getattr(meta, "prompt_token_count", 0) or 0
    out_tok = getattr(meta, "candidates_token_count", 0) or 0
    total = getattr(meta, "total_token_count", None)
    if total is None:
        total = in_tok + out_tok
    return in_tok, out_tok, total


def _extract_model_from_kwargs(kwargs: dict, response: Any) -> str:
    # google-genai uses model= kwarg on the call
    model = kwargs.get("model", "") or ""
    if not model:
        model = getattr(response, "model", "") or ""
    # Strip leading "models/" prefix if present
    if model.startswith("models/"):
        model = model[len("models/"):]
    return model


class GooglePatcher(BasePatcher):
    provider = "google"

    def _do_patch(self) -> bool:
        try:
            import google.genai.models as _mod  # noqa: PLC0415
        except ImportError:
            raise  # Let base class handle

        self._module = _mod
        self._original = _mod.Models.generate_content
        # AsyncModels may not exist in all versions
        self._original_async = getattr(_mod, "AsyncModels", None)
        if self._original_async is not None:
            self._original_async_method = self._original_async.generate_content
        else:
            self._original_async_method = None

        patcher = self

        def _sync_create(self_client, *args, **kwargs):
            stream = kwargs.get("stream", False)
            start = time.perf_counter()
            response = patcher._original(self_client, *args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            model = _extract_model_from_kwargs(kwargs, response)

            if stream:
                return patcher._wrap_sync_stream(response, elapsed, model)

            usage_tuple = _extract_usage(response)
            if usage_tuple:
                patcher._record(*usage_tuple, model, elapsed, is_stream=False)
            return response

        _mod.Models.generate_content = _sync_create

        if self._original_async is not None:
            async def _async_create(self_client, *args, **kwargs):
                stream = kwargs.get("stream", False)
                start = time.perf_counter()
                response = await patcher._original_async_method(
                    self_client, *args, **kwargs
                )
                elapsed = (time.perf_counter() - start) * 1000
                model = _extract_model_from_kwargs(kwargs, response)

                if stream:
                    return patcher._wrap_async_stream(response, elapsed, model)

                usage_tuple = _extract_usage(response)
                if usage_tuple:
                    patcher._record(*usage_tuple, model, elapsed, is_stream=False)
                return response

            self._original_async.generate_content = _async_create

        return True

    def _do_unpatch(self) -> None:
        if self._original is not None:
            self._module.Models.generate_content = self._original
        if self._original_async is not None and self._original_async_method is not None:
            self._original_async.generate_content = self._original_async_method

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
        self, stream: Any, start_elapsed: float, model: str
    ) -> Iterator[Any]:
        start = time.perf_counter()
        last_usage: tuple[int, int, int] | None = None

        try:
            for chunk in stream:
                usage_tuple = _extract_usage(chunk)
                if usage_tuple:
                    last_usage = usage_tuple
                yield chunk
        finally:
            elapsed = start_elapsed + (time.perf_counter() - start) * 1000
            if last_usage:
                self._record(*last_usage, model, elapsed, True)

    async def _wrap_async_stream(
        self, stream: Any, start_elapsed: float, model: str
    ) -> AsyncIterator[Any]:
        start = time.perf_counter()
        last_usage: tuple[int, int, int] | None = None

        try:
            async for chunk in stream:
                usage_tuple = _extract_usage(chunk)
                if usage_tuple:
                    last_usage = usage_tuple
                yield chunk
        finally:
            elapsed = start_elapsed + (time.perf_counter() - start) * 1000
            if last_usage:
                self._record(*last_usage, model, elapsed, True)
