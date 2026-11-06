"""Per-turn token accounting, priced on the platform's rate card.

Each result the CLI reports is priced from its tokens: Anthropic's rates for
an Anthropic model (cache writes split by lifetime, as the API reports
them), the Moonshot rate card for Kimi, the conservative fallback for any
other model. The CLI's own ``total_cost_usd`` is never charged: its price
table can lag the provider's. It is kept to log against the rate card and to
notice spend the CLI ran up without reporting a result.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backend.copilot.anthropic_rate_card import (
    anthropic_usage_cost_usd,
    fallback_usage_cost_usd,
)
from backend.copilot.moonshot import moonshot_cost_usd
from backend.copilot.transcript import cli_session_cost_state, cli_session_path

logger = logging.getLogger(__name__)

_TOKEN_FIELDS = {
    "input_tokens": "inputTokens",
    "output_tokens": "outputTokens",
    "cache_read_input_tokens": "cacheReadInputTokens",
    "cache_creation_input_tokens": "cacheCreationInputTokens",
}


class CLISessionUsage(BaseModel):
    """The running totals restored from a native CLI session file."""

    cost_usd: float = 0.0
    tokens: dict[str, int] = Field(default_factory=dict)


def read_cli_session_usage(
    sdk_cwd: str, session_id: str | None, log_prefix: str
) -> CLISessionUsage:
    """Read the CLI's own baseline; alert when an unreadable file forces zero."""
    if not session_id or not sdk_cwd:
        return CLISessionUsage()
    try:
        content = Path(cli_session_path(sdk_cwd, session_id)).read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeDecodeError) as e:
        logger.error(
            "%s Over-charge fallback: could not read the usage of CLI session %s "
            "(%s); counting from zero, a result the CLI fails to report would "
            "charge the session's earlier tokens too",
            log_prefix,
            session_id,
            type(e).__name__,
        )
        return CLISessionUsage()
    entry = cli_session_cost_state(content, session_id)
    if entry is None:
        return CLISessionUsage()
    return CLISessionUsage(
        cost_usd=float(entry["totalCostUSD"]), tokens=_snapshot_tokens(entry)
    )


def _snapshot_tokens(entry: dict[str, Any]) -> dict[str, int]:
    """Normalize cumulative per-model CLI counters to ResultMessage usage keys."""
    models = entry.get("modelUsage")
    if not isinstance(models, dict):
        return {}
    tokens: dict[str, int] = {}
    for usage in models.values():
        if not isinstance(usage, dict):
            continue
        for key, native_key in _TOKEN_FIELDS.items():
            value = usage.get(native_key)
            if isinstance(value, int) and value >= 0:
                tokens[key] = tokens.get(key, 0) + value
    return tokens


class TokenUsage(BaseModel):
    """Usage accumulated across all queries, retries, and CLI sessions in a turn.

    ``priced`` is False on a route whose usage no rate card prices (Codex):
    tokens are counted and ``cost_usd`` stays None.
    """

    priced: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    # The part of ``cache_creation_tokens`` written for an hour, not five
    # minutes; the two are priced differently.
    cache_write_1h_tokens: int = 0
    cost_usd: float | None = None
    cli_cost_usd: float = 0.0
    cli_session_total_usd: float | None = None
    cli_accounted_tokens: dict[str, int] = Field(default_factory=dict)

    def start_cli_session(self, snapshot: CLISessionUsage) -> None:
        """Set the new session's baselines without discarding earlier turn usage."""
        self.cli_session_total_usd = snapshot.cost_usd
        self.cli_accounted_tokens = dict(snapshot.tokens)

    def record_result(
        self,
        usage: dict[str, Any] | None,
        total_cost_usd: float | None,
        model: str | None,
        log_prefix: str,
    ) -> None:
        """Add this query's tokens to the turn and price them on the rate card."""
        tokens = {key: (usage or {}).get(key) or 0 for key in _TOKEN_FIELDS}
        if total_cost_usd is not None:
            self.cli_cost_usd += self._spend_since_last_result(
                total_cost_usd, log_prefix
            )
        self.prompt_tokens += tokens["input_tokens"]
        self.completion_tokens += tokens["output_tokens"]
        self.cache_read_tokens += tokens["cache_read_input_tokens"]
        self.cache_creation_tokens += tokens["cache_creation_input_tokens"]
        write_1h = _cache_write_1h(usage, tokens["cache_creation_input_tokens"])
        self.cache_write_1h_tokens += write_1h
        for key, value in tokens.items():
            self.cli_accounted_tokens[key] = (
                self.cli_accounted_tokens.get(key, 0) + value
            )
        if not self.priced or not any(tokens.values()):
            return
        priced = rate_card_usd(model, tokens, cache_write_1h=write_1h)
        self.cost_usd = (self.cost_usd or 0.0) + priced

    def record_unreported(
        self, snapshot: CLISessionUsage, model: str | None, log_prefix: str
    ) -> None:
        """Account for a missing result without charging earlier results again.

        The session file holds no cache-lifetime split, so its cache writes
        are priced at five minutes, the lifetime the CLI gives the
        conversation's cache.
        """
        if snapshot.cost_usd <= (self.cli_session_total_usd or 0.0):
            return
        tokens = {
            key: max(0, value - self.cli_accounted_tokens.get(key, 0))
            for key, value in snapshot.tokens.items()
        }
        if not any(tokens.values()):
            logger.error(
                "%s Incomplete CLI response ran up spend with no recoverable "
                "tokens; nothing is charged for it",
                log_prefix,
            )
        self.record_result(tokens, snapshot.cost_usd, model, log_prefix)

    def _spend_since_last_result(self, total_cost_usd: float, log_prefix: str) -> float:
        """The CLI's increment, for the log; a reset CLI total counts in full."""
        previous_total = self.cli_session_total_usd or 0.0
        spend = total_cost_usd - previous_total
        if spend < 0:
            logger.error(
                "%s CLI total_cost_usd $%.6f is below the previous session "
                "total ($%.6f); counting the full CLI total",
                log_prefix,
                total_cost_usd,
                previous_total,
            )
            spend = total_cost_usd
            self.cli_accounted_tokens.clear()
        self.cli_session_total_usd = total_cost_usd
        return spend


def rate_card_usd(
    model: str | None, tokens: Mapping[str, int], *, cache_write_1h: int = 0
) -> float:
    """The platform's price for one result's tokens on *model*."""
    input_tokens = tokens.get("input_tokens", 0)
    output_tokens = tokens.get("output_tokens", 0)
    cache_read = tokens.get("cache_read_input_tokens", 0)
    cache_write = tokens.get("cache_creation_input_tokens", 0)
    moonshot = moonshot_cost_usd(
        model=model,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_write,
    )
    if moonshot is not None:
        return moonshot
    write_1h = min(max(0, cache_write_1h), cache_write)
    anthropic = anthropic_usage_cost_usd(
        model=_rate_card_name(model or ""),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_5m_tokens=cache_write - write_1h,
        cache_write_1h_tokens=write_1h,
    )
    if anthropic is not None:
        return anthropic
    logger.error(
        "No rate card prices model=%s; charging the conservative fallback rates",
        model,
    )
    return fallback_usage_cost_usd(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def _rate_card_name(model: str) -> str:
    """``anthropic/claude-opus-4.6`` as the rate card keys it: ``claude-opus-4-6``."""
    if model.startswith("anthropic/"):
        return model.removeprefix("anthropic/").replace(".", "-")
    return model


def _cache_write_1h(usage: dict[str, Any] | None, cache_write: int) -> int:
    """The hour-long part of a result's cache writes (``usage.cache_creation``)."""
    creation = (usage or {}).get("cache_creation")
    if not isinstance(creation, dict):
        return 0
    value = creation.get("ephemeral_1h_input_tokens")
    return min(value, cache_write) if isinstance(value, int) and value > 0 else 0
