"""Flag-off regression guarantee for the copilot system prompt.

When the ``hire-experts`` flag is OFF a user can never open an expert session,
so ``expert_id`` is always ``None`` and both engines must serve the exact
pre-experts system prompt. This suite composes the plain-session prompt through
``compose_system_prompt`` — the single composition point all three production
call sites use (SDK main turn, SDK building-mode restart, baseline) — with the
real expert-suffix builder, and pins the SHA-256 for every deterministic engine
configuration (``use_e2b`` and Graphiti are orthogonal toggles that exist
flag-off too, so each combination is pinned). Any edit that changes flag-off
prompt bytes fails CI loudly.

Scope, stated honestly: in production ``_build_system_prompt`` may source the
base prompt from Langfuse at runtime; that content lives outside this repo and
cannot be pinned by CI. This suite pins the in-repo fallback base
(``CACHEABLE_SYSTEM_PROMPT`` — the exact base whenever Langfuse is
unconfigured, itself snapshot-pinned in ``expert_context_test.py``); flag-off
parity of a Langfuse-hosted base must be maintained in the Langfuse template.
"""

import hashlib

import pytest

from backend.copilot.expert_context import build_expert_identity_suffix
from backend.copilot.prompting import (
    SHARED_TOOL_NOTES,
    compose_system_prompt,
    get_graphiti_supplement,
    get_sdk_supplement,
)
from backend.copilot.service import CACHEABLE_SYSTEM_PROMPT

_FLAG_OFF_CHANGED_MSG = (
    "flag-off prompt changed; if intentional, update hash + call out in PR description"
)


def _engine_supplement(engine: str, use_e2b: bool) -> str:
    if engine == "sdk":
        return get_sdk_supplement(use_e2b=use_e2b)
    return SHARED_TOOL_NOTES


class TestFlagOffPlainSessionPrompt:
    @pytest.mark.asyncio
    async def test_plain_session_expert_suffix_is_empty(self):
        # hire-experts OFF -> no expert_id -> experts contributes nothing.
        suffix = await build_expert_identity_suffix("flag-off-user", None)
        assert suffix == "", _FLAG_OFF_CHANGED_MSG

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "engine,use_e2b,graphiti_enabled,expected_sha256",
        [
            (
                "sdk",
                False,
                False,
                "ccd74050220a0aa4fba9edefeba283d737540558d3de1ddf21cf01c0a7529aec",
            ),
            (
                "sdk",
                False,
                True,
                "a50dacd30e9f15a2b55e6764d526074b2dba038fd42846f29814143fd0376a07",
            ),
            (
                "sdk",
                True,
                False,
                "7a8a11acfd42edc87a8bfe9eec5c54bc5d88716b46e17c7cf5af6a1d60fdf453",
            ),
            (
                "sdk",
                True,
                True,
                "31e2058efd71bc6df46bab58b2cbc0950ed125d07c4c221012e6b3481706fd06",
            ),
            (
                "baseline",
                False,
                False,
                "99d8c272de5c3b8fe8c4cd1546bed8a186b218c789c7e4ad545fa91dc9282b02",
            ),
            (
                "baseline",
                False,
                True,
                "9aa9eee140ef64210ecddba44ed67ceeb759fea7e3f8d00c9e9ea74586cfa7c2",
            ),
        ],
    )
    async def test_plain_session_prompt_hash_is_pinned(
        self,
        engine: str,
        use_e2b: bool,
        graphiti_enabled: bool,
        expected_sha256: str,
    ):
        expert_suffix = await build_expert_identity_suffix("flag-off-user", None)
        plain_prompt = compose_system_prompt(
            CACHEABLE_SYSTEM_PROMPT,
            _engine_supplement(engine, use_e2b),
            get_graphiti_supplement() if graphiti_enabled else "",
            "",
            expert_suffix,
        )
        digest = hashlib.sha256(plain_prompt.encode("utf-8")).hexdigest()
        assert digest == expected_sha256, _FLAG_OFF_CHANGED_MSG
