"""Abstract base class for SDK monkey-patchers."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..config import TokenMeterConfig
from ..storage.base import BaseStorage

logger = logging.getLogger(__name__)


class BasePatcher(ABC):
    """Manages patching/unpatching a single provider SDK."""

    provider: str = ""

    def __init__(self, config: TokenMeterConfig, storage: BaseStorage) -> None:
        self._config = config
        self._storage = storage
        self._active = False
        self._original: Optional[Callable] = None
        self._original_async: Optional[Callable] = None

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def patch(self) -> bool:
        """Apply the monkey-patch. Returns True if successful."""
        if self._active:
            return True
        try:
            result = self._do_patch()
            if result:
                self._active = True
                logger.debug("token-meter: patched %s", self.provider)
            return result
        except ImportError:
            logger.debug(
                "token-meter: %s SDK not installed — skipping", self.provider
            )
            return False
        except Exception:
            logger.exception(
                "token-meter: unexpected error patching %s", self.provider
            )
            return False

    def unpatch(self) -> None:
        """Restore the original methods."""
        if not self._active:
            return
        try:
            self._do_unpatch()
        except Exception:
            logger.exception(
                "token-meter: error unpatching %s", self.provider
            )
        finally:
            self._active = False
            logger.debug("token-meter: unpatched %s", self.provider)

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------ #
    # Subclass hooks                                                       #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _do_patch(self) -> bool:
        """Install patches; return True on success."""

    @abstractmethod
    def _do_unpatch(self) -> None:
        """Restore original methods."""
