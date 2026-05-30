"""Settings loaded from environment variables.

Single source of truth for runtime knobs. Modules import `settings` rather than
re-reading os.environ — keeps overrides predictable and testable.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _read_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class Settings(BaseModel):
    """Runtime configuration. Loaded once at import time; override per-test via `Settings(...)`."""

    gemini_api_key: str | None = Field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY"),
    )
    groq_api_key: str | None = Field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY"),
    )
    model: str = Field(
        default_factory=lambda: _read_env("AI_QUANT_LAB_MODEL", "gemini-3.1-flash-lite"),
    )

    max_llm_calls: int = Field(
        default_factory=lambda: int(_read_env("AI_QUANT_LAB_MAX_LLM_CALLS", "200")),
        ge=1,
    )
    target_survivors: int = Field(
        default_factory=lambda: int(_read_env("AI_QUANT_LAB_TARGET_SURVIVORS", "3")),
        ge=1,
    )

    dsr_pvalue_max: float = Field(
        default_factory=lambda: float(_read_env("AI_QUANT_LAB_DSR_PVALUE_MAX", "0.05")),
        gt=0.0,
        lt=1.0,
    )
    max_correlation: float = Field(
        default_factory=lambda: float(_read_env("AI_QUANT_LAB_MAX_CORRELATION", "0.6")),
        ge=0.0,
        le=1.0,
    )

    cost_bps: float = Field(
        default_factory=lambda: float(_read_env("AI_QUANT_LAB_COST_BPS", "8.0")),
        ge=0.0,
    )
    annualization: int = Field(
        default_factory=lambda: int(_read_env("AI_QUANT_LAB_ANNUALIZATION", "252")),
        ge=1,
    )

    memory_db: Path = Field(
        default_factory=lambda: Path(_read_env("AI_QUANT_LAB_MEMORY_DB", "./memory.db")),
    )

    def require_api_key(self, model: str | None = None) -> tuple[str, str]:
        """Return (api_key, provider) for the given model, or the default model.

        Provider is 'gemini' or 'groq'.
        """
        use_model = model or self.model
        if use_model.startswith("gemini"):
            if not self.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is not set.")
            return self.gemini_api_key, "gemini"
        else:
            if not self.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set.")
            return self.groq_api_key, "groq"


settings = Settings()
