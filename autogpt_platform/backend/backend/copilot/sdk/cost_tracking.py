"""Per-turn accounting for the CLI's cumulative session cost and token snapshots."""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backend.copilot.moonshot import is_moonshot_model, override_cost_usd
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
            "%s Over-charge fallback: could not read the cost of CLI session %s "
            "(%s); counting from $0 may charge the full session total on resume",
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
    """Usage accumulated across all queries, retries, and CLI sessions in a turn."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
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
        """Price this query using its serving model, then add it to the turn."""
        tokens = {key: (usage or {}).get(key) or 0 for key in _TOKEN_FIELDS}
        spend = (
            self._spend_since_last_result(total_cost_usd, log_prefix)
            if total_cost_usd is not None
            else None
        )
        self.prompt_tokens += tokens["input_tokens"]
        self.completion_tokens += tokens["output_tokens"]
        self.cache_read_tokens += tokens["cache_read_input_tokens"]
        self.cache_creation_tokens += tokens["cache_creation_input_tokens"]
        for key, value in tokens.items():
            self.cli_accounted_tokens[key] = (
                self.cli_accounted_tokens.get(key, 0) + value
            )
        if spend is None:
            return
        self.cli_cost_usd += spend
        priced_spend = override_cost_usd(
            model=model,
            sdk_reported_usd=spend,
            prompt_tokens=tokens["input_tokens"],
            completion_tokens=tokens["output_tokens"],
            cache_read_tokens=tokens["cache_read_input_tokens"],
            cache_creation_tokens=tokens["cache_creation_input_tokens"],
        )
        self.cost_usd = (self.cost_usd or 0.0) + priced_spend

    def record_unreported(
        self, snapshot: CLISessionUsage, model: str | None, log_prefix: str
    ) -> None:
        """Account for a missing result without charging earlier results again."""
        if snapshot.cost_usd <= (self.cli_session_total_usd or 0.0):
            return
        tokens = {
            key: max(0, value - self.cli_accounted_tokens.get(key, 0))
            for key, value in snapshot.tokens.items()
        }
        if is_moonshot_model(model) and not any(tokens.values()):
            logger.error(
                "%s Over-charge fallback: incomplete CLI response has spend but no "
                "recoverable Moonshot tokens; charging the CLI-priced increment",
                log_prefix,
            )
            model = None
        self.record_result(tokens, snapshot.cost_usd, model, log_prefix)

    def _spend_since_last_result(self, total_cost_usd: float, log_prefix: str) -> float:
        """Take an increment, retaining the alerted fallback for a reset CLI total."""
        previous_total = self.cli_session_total_usd or 0.0
        spend = total_cost_usd - previous_total
        if spend < 0:
            logger.error(
                "%s Over-charge fallback: CLI total_cost_usd $%.6f is below the "
                "previous session total ($%.6f); charging the full CLI total",
                log_prefix,
                total_cost_usd,
                previous_total,
            )
            spend = total_cost_usd
            self.cli_accounted_tokens.clear()
        self.cli_session_total_usd = total_cost_usd
        return spend
