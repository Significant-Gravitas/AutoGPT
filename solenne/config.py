from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Runtime configuration for Solenne."""

    openai_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMB_MODEL: str = "text-embedding-3-small"
    root: str = field(default_factory=lambda: os.path.abspath("."))


CFG = Config()
