"""Configuration management for TokenMeter."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_SUPPORTED_PROVIDERS = ("openai", "anthropic", "google")

DEFAULT_DB_PATH = Path.home() / ".token-meter" / "usage.db"


@dataclass
class TokenMeterConfig:
    """Runtime configuration for the TokenMeter SDK."""

    project: str = "default"
    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)
    providers: List[str] = field(default_factory=lambda: list(_SUPPORTED_PROVIDERS))
    warn_unknown_models: bool = True

    def __post_init__(self) -> None:
        # Normalise db_path
        if isinstance(self.db_path, str):
            self.db_path = Path(self.db_path)
        self.db_path = self.db_path.expanduser().resolve()

        # Normalise providers
        self.providers = [p.lower().strip() for p in self.providers]
        unknown = [p for p in self.providers if p not in _SUPPORTED_PROVIDERS]
        if unknown:
            raise ValueError(
                f"Unknown provider(s): {unknown}. "
                f"Supported: {list(_SUPPORTED_PROVIDERS)}"
            )

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_kwargs(
        cls,
        project: str = "default",
        db_path: Optional[str] = None,
        providers: Optional[List[str]] = None,
    ) -> "TokenMeterConfig":
        kwargs = {"project": project}
        if db_path is not None:
            kwargs["db_path"] = db_path  # type: ignore[assignment]
        if providers is not None:
            kwargs["providers"] = providers  # type: ignore[assignment]

        # Also check env vars
        if db_path is None:
            env_path = os.environ.get("TOKEN_METER_DB_PATH")
            if env_path:
                kwargs["db_path"] = env_path  # type: ignore[assignment]

        if not project or project == "default":
            env_project = os.environ.get("TOKEN_METER_PROJECT")
            if env_project:
                kwargs["project"] = env_project

        return cls(**kwargs)
