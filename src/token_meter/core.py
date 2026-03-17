"""TokenTracker: the central coordinator for the TokenMeter SDK.

Manages the patch lifecycle and provides query/aggregate access to storage.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from .config import TokenMeterConfig
from .patchers.openai import OpenAIPatcher
from .patchers.anthropic import AnthropicPatcher
from .patchers.google import GooglePatcher
from .storage.sqlite import SQLiteStorage
from .storage.base import BaseStorage
from .models import UsageRecord

logger = logging.getLogger(__name__)

_PATCHER_MAP = {
    "openai": OpenAIPatcher,
    "anthropic": AnthropicPatcher,
    "google": GooglePatcher,
}


class TokenTracker:
    """Manages monkey-patch lifecycle and provides access to stored records."""

    def __init__(self, config: TokenMeterConfig, storage: BaseStorage) -> None:
        self._config = config
        self._storage = storage
        self._patchers: List[Any] = []
        self._active = False

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._active:
            logger.debug("token-meter: already active, skipping init")
            return

        patched_providers = []
        for provider in self._config.providers:
            cls = _PATCHER_MAP.get(provider)
            if cls is None:
                continue
            patcher = cls(self._config, self._storage)
            if patcher.patch():
                self._patchers.append(patcher)
                patched_providers.append(provider)

        self._active = True
        if patched_providers:
            logger.info("token-meter: tracking %s", ", ".join(patched_providers))
        else:
            logger.warning("token-meter: no providers were patched")

    def stop(self) -> None:
        if not self._active:
            return
        for patcher in self._patchers:
            patcher.unpatch()
        self._patchers.clear()
        self._active = False
        logger.info("token-meter: stopped")

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------ #
    # Storage passthrough                                                  #
    # ------------------------------------------------------------------ #

    def query(
        self,
        project: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[UsageRecord]:
        return self._storage.query(
            project=project,
            provider=provider,
            model=model,
            start=start,
            end=end,
            limit=limit,
        )

    def aggregate(
        self,
        project: Optional[str] = None,
        provider: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        return self._storage.aggregate(
            project=project,
            provider=provider,
            start=start,
            end=end,
        )

    def aggregate_by_model(
        self,
        project: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        return self._storage.aggregate_by_model(
            project=project,
            start=start,
            end=end,
        )

    def close(self) -> None:
        self.stop()
        self._storage.close()
