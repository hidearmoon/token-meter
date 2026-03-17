"""Data models for TokenMeter usage records."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class UsageRecord:
    """Represents a single LLM API call with token usage and cost details."""

    provider: str                   # 'openai' | 'anthropic' | 'google'
    model: str                      # e.g. 'gpt-4o', 'claude-sonnet-4-20250514'
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float               # USD
    output_cost: float              # USD
    total_cost: float               # USD
    latency_ms: float               # wall-clock call duration
    project: str = "default"
    is_stream: bool = False
    metadata: Optional[Dict[str, Any]] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------ #
    # Serialisation helpers                                                #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "latency_ms": self.latency_ms,
            "project": self.project,
            "is_stream": self.is_stream,
            "metadata": self.metadata,
        }

    # ------------------------------------------------------------------ #
    # Row helpers for SQLite                                               #
    # ------------------------------------------------------------------ #

    def to_row(self) -> tuple:
        return (
            self.id,
            self.timestamp.isoformat(),
            self.provider,
            self.model,
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
            round(self.input_cost, 6),
            round(self.output_cost, 6),
            round(self.total_cost, 6),
            self.latency_ms,
            self.project,
            self.is_stream,
            json.dumps(self.metadata) if self.metadata else None,
        )

    @classmethod
    def from_row(cls, row: tuple) -> "UsageRecord":
        (
            id_, ts, provider, model,
            in_tok, out_tok, total_tok,
            in_cost, out_cost, total_cost,
            latency, project, is_stream, metadata_json,
        ) = row
        return cls(
            id=id_,
            timestamp=datetime.fromisoformat(ts),
            provider=provider,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=total_tok,
            input_cost=float(in_cost),
            output_cost=float(out_cost),
            total_cost=float(total_cost),
            latency_ms=float(latency),
            project=project,
            is_stream=bool(is_stream),
            metadata=json.loads(metadata_json) if metadata_json else None,
        )
