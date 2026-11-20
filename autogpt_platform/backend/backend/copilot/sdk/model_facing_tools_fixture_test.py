"""The frontend labels every tool call it renders from a catalog keyed by tool
name. Its coverage test reads the names from this fixture, built here from the
registries the model is actually offered, so a tool added without a label
fails CI instead of reaching users as "List routines completed".

Regenerate with ``UPDATE_TOOL_NAMES_FIXTURE=1``.
"""

import json
import os
from pathlib import Path

from backend.copilot.tools import TOOL_REGISTRY

from .tool_adapter import MCP_TOOL_PREFIX, get_copilot_tool_names

FIXTURE = (
    Path(__file__).parents[4]
    / "frontend/src/app/(platform)/copilot/components/ToolChain/__tests__"
    / "modelFacingTools.json"
)


def model_facing_tool_names() -> list[str]:
    """Every name a tool call can reach the chat under.

    Deferred registry tools are absent from the SDK's allowed list but still
    stream under their own name once ``run_capability`` dispatches them, so
    the whole registry counts alongside both SDK tool sets.
    """
    sdk_names = {
        name.removeprefix(MCP_TOOL_PREFIX)
        for use_e2b in (False, True)
        for name in get_copilot_tool_names(use_e2b=use_e2b)
    }
    return sorted(set(TOOL_REGISTRY) | sdk_names)


def test_the_frontend_tool_names_fixture_matches_the_registries():
    names = model_facing_tool_names()
    if os.environ.get("UPDATE_TOOL_NAMES_FIXTURE"):
        FIXTURE.write_text(json.dumps(names, indent=2) + "\n")
    assert (
        json.loads(FIXTURE.read_text()) == names
    ), "the model-facing tool names fixture is stale; rerun with UPDATE_TOOL_NAMES_FIXTURE=1"
