"""
TWZRD Agent Intel integration for AutoGPT.

Adds trust verification capability to AutoGPT agents. Before AutoGPT delegates
tasks to external agents or authorizes x402 micropayments, it can use this
module to verify trust scores via the TWZRD Agent Intel MCP server.

The TWZRD Agent Intel server provides:
  - score_agent(wallet)       — 0-100 trust score + risk flags (free)
  - preflight_check(wallet)   — PASS/FAIL gate for x402 payments (free)
  - get_trust_receipt(wallet) — signed trust receipt (HTTP 402 paid)

MCP endpoint: https://intel.twzrd.xyz/mcp  (streamable-http, no auth)
Website: https://intel.twzrd.xyz

Install:
    pip install mcp

Usage:
    from autogpt.app.twzrd_agent_intel import check_agent_trust, is_trusted
"""
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

TWZRD_MCP_URL = "https://intel.twzrd.xyz/mcp"
TRUST_THRESHOLD = 60  # Minimum score for x402 payment authorization


@dataclass
class TrustResult:
    wallet: str
    score: int
    trust_level: str
    risk_flags: list
    preflight_passed: bool
    authorized: bool


async def _call_mcp_tool(tool_name: str, wallet: str) -> str:
    """Call a TWZRD Agent Intel MCP tool and return the raw text response."""
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:
        raise ImportError("mcp package not installed. Run: pip install mcp")

    async with streamablehttp_client(TWZRD_MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, {"wallet": wallet})

    return result.content[0].text if result.content else ""


async def check_agent_trust_async(wallet: str) -> TrustResult:
    """
    Asynchronously check an agent's trust score.

    Args:
        wallet: Solana wallet address (base58 encoded)

    Returns:
        TrustResult with score, level, flags, and authorization decision
    """
    logger.info(f"Checking TWZRD trust for wallet {wallet[:8]}...")

    # Get score
    score_text = await _call_mcp_tool("score_agent", wallet)
    try:
        score_data = json.loads(score_text)
    except (json.JSONDecodeError, AttributeError):
        score_data = {}

    score = score_data.get("score", 0)
    flags = score_data.get("risk_flags", [])

    # Get preflight
    preflight_text = await _call_mcp_tool("preflight_check", wallet)
    preflight_passed = "PASS" in preflight_text.upper()

    # Determine trust level
    if score >= 80:
        trust_level = "HIGH"
    elif score >= TRUST_THRESHOLD:
        trust_level = "MEDIUM"
    else:
        trust_level = "LOW"

    authorized = score >= TRUST_THRESHOLD and preflight_passed and not flags

    result = TrustResult(
        wallet=wallet,
        score=score,
        trust_level=trust_level,
        risk_flags=flags,
        preflight_passed=preflight_passed,
        authorized=authorized,
    )

    logger.info(
        f"Trust result for {wallet[:8]}...: score={score} "
        f"level={trust_level} preflight={'PASS' if preflight_passed else 'FAIL'} "
        f"authorized={authorized}"
    )
    return result


def check_agent_trust(wallet: str) -> TrustResult:
    """
    Synchronous wrapper for check_agent_trust_async.

    Args:
        wallet: Solana wallet address (base58 encoded)

    Returns:
        TrustResult with score, level, flags, and authorization decision
    """
    return asyncio.run(check_agent_trust_async(wallet))


def is_trusted(wallet: str, threshold: Optional[int] = None) -> bool:
    """
    Quick boolean trust check for a wallet.

    Args:
        wallet: Solana wallet address (base58 encoded)
        threshold: Minimum score to consider trusted (default: TRUST_THRESHOLD)

    Returns:
        True if agent passes trust check, False otherwise
    """
    min_score = threshold or TRUST_THRESHOLD
    result = check_agent_trust(wallet)
    return result.score >= min_score and result.preflight_passed


if __name__ == "__main__":
    # Example usage
    wallet = "4LkEFjHsF2ubC8K4oF2r3rCFqPZQVGBjL9mV6xkNPZdf"

    print(f"Checking TWZRD trust for: {wallet[:8]}...")
    result = check_agent_trust(wallet)

    print(f"\nTrust Score:   {result.score}/100")
    print(f"Trust Level:   {result.trust_level}")
    print(f"Risk Flags:    {result.risk_flags or 'none'}")
    print(f"Preflight:     {'PASS' if result.preflight_passed else 'FAIL'}")
    print(f"Authorized:    {result.authorized}")

    print(f"\nis_trusted():  {is_trusted(wallet)}")
