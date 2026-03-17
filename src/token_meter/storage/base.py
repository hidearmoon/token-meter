"""Abstract base class for TokenMeter storage backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import UsageRecord


class BaseStorage(ABC):
    """All storage backends must implement this interface."""

    @abstractmethod
    def save(self, record: UsageRecord) -> None:
        """Persist a usage record."""

    @abstractmethod
    def query(
        self,
        project: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[UsageRecord]:
        """Return matching records, newest first."""

    @abstractmethod
    def aggregate(
        self,
        project: Optional[str] = None,
        provider: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Return aggregated stats: total_tokens, total_cost, call_count, etc."""

    @abstractmethod
    def close(self) -> None:
        """Release any held resources."""
