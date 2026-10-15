"""The SDK path names AutoPilot's tools the way it declares them.

The SDK declares every AutoPilot tool as ``mcp__copilot__<name>``, and the
CLI refuses a call to an undeclared name before any hook runs. The texts this
path shares with the baseline engine write the short name, which the model
copies into its call; so on this path the skills index names the declared
tool, and the system prompt maps short names to declared ones.
"""

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.prompting import get_sdk_supplement
from backend.copilot.sdk.service import _first_turn_context_blocks
from backend.copilot.sdk.tool_adapter import MCP_TOOL_PREFIX
from backend.copilot.service import SKILLS_CONTEXT_TAG
from backend.copilot.session_kind import SessionKind


@pytest.mark.asyncio
async def test_the_skills_index_names_the_declared_run_capability(tmp_path):
    session = ChatSession.new("user-1", dry_run=False)

    blocks = await _first_turn_context_blocks(
        session,
        None,
        SessionKind.for_turn(session),
        sdk_cwd=str(tmp_path),
        use_e2b=False,
        understanding=None,
    )
    (skills,) = [block for block in blocks if block.tag == SKILLS_CONTEXT_TAG]

    declared_call = (
        f'`{MCP_TOOL_PREFIX}run_capability(id="tool:read_skill", '
        'input={"name": ...})`'
    )
    assert declared_call in skills.head


@pytest.mark.parametrize("use_e2b", [False, True])
@pytest.mark.parametrize("expert_session", [False, True])
def test_the_system_prompt_maps_short_names_to_declared_ones(
    use_e2b: bool, expert_session: bool
):
    supplement = get_sdk_supplement(use_e2b=use_e2b, expert_session=expert_session)

    assert f"called by its full name, `{MCP_TOOL_PREFIX}<name>`" in supplement
    assert "`run_capability(...)`, `find_capability(...)`" in supplement
    assert f"means the `{MCP_TOOL_PREFIX}` tool of that name" in supplement
