"""Flag-off regression guarantee for the copilot system prompt.

When the ``hire-experts`` flag is OFF a user can never open an expert session,
so ``expert_id`` is always ``None`` and the copilot must serve the exact
pre-experts system prompt. This suite pins the SHA-256 of the prompt a PLAIN
session (no expert) receives, so any edit that would change flag-off behavior
fails CI loudly.

Sibling coverage: ``expert_context_test.py`` pins the base
``_CACHEABLE_SYSTEM_PROMPT`` constant in isolation. This suite pins the
*composed* plain-session prompt (base + the expert-identity suffix a plain
session actually gets), so it also fails if the experts feature ever leaks a
non-empty suffix into the no-expert path.
"""

import hashlib

import pytest

from backend.copilot.expert_context import build_expert_identity_suffix
from backend.copilot.service import CACHEABLE_SYSTEM_PROMPT

# SHA-256 of the prompt a PLAIN copilot session (no expert) receives:
#   CACHEABLE_SYSTEM_PROMPT + build_expert_identity_suffix(user, expert_id=None)
# Flag-off = byte-identical prod behavior.
_FLAG_OFF_PLAIN_PROMPT_SHA256 = (
    "22d1897a44ec751b36e4938f087dc49ad9dcae6c452842ed057ba7ebe3de4545"
)

_FLAG_OFF_CHANGED_MSG = (
    "flag-off prompt changed; if intentional, update hash + call out in PR description"
)


class TestFlagOffPlainSessionPrompt:
    @pytest.mark.asyncio
    async def test_plain_session_expert_suffix_is_empty(self):
        # hire-experts OFF -> no expert_id -> experts contributes nothing.
        suffix = await build_expert_identity_suffix("flag-off-user", None)
        assert suffix == "", _FLAG_OFF_CHANGED_MSG

    @pytest.mark.asyncio
    async def test_plain_session_prompt_hash_is_pinned(self):
        suffix = await build_expert_identity_suffix("flag-off-user", None)
        plain_prompt = CACHEABLE_SYSTEM_PROMPT + suffix
        digest = hashlib.sha256(plain_prompt.encode("utf-8")).hexdigest()
        assert digest == _FLAG_OFF_PLAIN_PROMPT_SHA256, _FLAG_OFF_CHANGED_MSG
