"""
Tests for ToolOrchestratorBlock support in agent generator.

Covers:
- AgentFixer.fix_tool_orchestrator_blocks()
- AgentValidator.validate_tool_orchestrator_blocks()
- End-to-end fix → validate → pipeline for ToolOrchestrator agents
"""

import uuid

from backend.copilot.tools.agent_generator.fixer import AgentFixer
from backend.copilot.tools.agent_generator.helpers import (
    AGENT_EXECUTOR_BLOCK_ID,
    AGENT_INPUT_BLOCK_ID,
    AGENT_OUTPUT_BLOCK_ID,
    TOOL_ORCHESTRATOR_BLOCK_ID,
)
from backend.copilot.tools.agent_generator.validator import AgentValidator


def _uid() -> str:
    return str(uuid.uuid4())


def _make_sdm_node(
    node_id: str | None = None,
    input_default: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """Create a ToolOrchestratorBlock node dict."""
    return {
        "id": node_id or _uid(),
        "block_id": TOOL_ORCHESTRATOR_BLOCK_ID,
        "input_default": input_default or {},
        "metadata": metadata or {"position": {"x": 0, "y": 0}},
    }


def _make_agent_executor_node(
    node_id: str | None = None,
    graph_id: str | None = None,
) -> dict:
    """Create an AgentExecutorBlock node dict."""
    return {
        "id": node_id or _uid(),
        "block_id": AGENT_EXECUTOR_BLOCK_ID,
        "input_default": {
            "graph_id": graph_id or _uid(),
            "graph_version": 1,
            "input_schema": {"properties": {"query": {"type": "string"}}},
            "output_schema": {"properties": {"result": {"type": "string"}}},
            "user_id": "",
            "inputs": {},
        },
        "metadata": {"position": {"x": 800, "y": 0}},
    }


def _make_input_node(node_id: str | None = None, name: str = "task") -> dict:
    return {
        "id": node_id or _uid(),
        "block_id": AGENT_INPUT_BLOCK_ID,
        "input_default": {"name": name, "title": name.title()},
        "metadata": {"position": {"x": -800, "y": 0}},
    }


def _make_output_node(node_id: str | None = None, name: str = "result") -> dict:
    return {
        "id": node_id or _uid(),
        "block_id": AGENT_OUTPUT_BLOCK_ID,
        "input_default": {"name": name, "title": name.title()},
        "metadata": {"position": {"x": 1600, "y": 0}},
    }


def _link(
    source_id: str,
    source_name: str,
    sink_id: str,
    sink_name: str,
    is_static: bool = False,
) -> dict:
    return {
        "id": _uid(),
        "source_id": source_id,
        "source_name": source_name,
        "sink_id": sink_id,
        "sink_name": sink_name,
        "is_static": is_static,
    }


def _make_orchestrator_agent() -> dict:
    """Build a complete orchestrator agent with SDM + 2 sub-agent tools."""
    input_node = _make_input_node()
    sdm_node = _make_sdm_node()
    agent_a = _make_agent_executor_node()
    agent_b = _make_agent_executor_node()
    output_node = _make_output_node()

    return {
        "id": _uid(),
        "version": 1,
        "is_active": True,
        "name": "Orchestrator Agent",
        "description": "Uses AI to orchestrate sub-agents",
        "nodes": [input_node, sdm_node, agent_a, agent_b, output_node],
        "links": [
            # Input → SDM prompt
            _link(input_node["id"], "result", sdm_node["id"], "prompt"),
            # SDM tools → Agent A
            _link(sdm_node["id"], "tools", agent_a["id"], "query"),
            # SDM tools → Agent B
            _link(sdm_node["id"], "tools", agent_b["id"], "query"),
            # SDM finished → Output
            _link(sdm_node["id"], "finished", output_node["id"], "value"),
        ],
    }


# ---------------------------------------------------------------------------
# Fixer tests
# ---------------------------------------------------------------------------


class TestFixToolOrchestratorBlocks:
    """Tests for AgentFixer.fix_tool_orchestrator_blocks()."""

    def test_fills_defaults_when_missing(self):
        """All agent-mode defaults are populated for a bare SDM node."""
        fixer = AgentFixer()
        agent = {"nodes": [_make_sdm_node()], "links": []}

        result = fixer.fix_tool_orchestrator_blocks(agent)

        defaults = result["nodes"][0]["input_default"]
        assert defaults["agent_mode_max_iterations"] == 10
        assert defaults["conversation_compaction"] is True
        assert defaults["retry"] == 3
        assert defaults["multiple_tool_calls"] is False
        assert len(fixer.fixes_applied) == 4

    def test_preserves_existing_values(self):
        """Existing user-set values are never overwritten."""
        fixer = AgentFixer()
        agent = {
            "nodes": [
                _make_sdm_node(
                    input_default={
                        "agent_mode_max_iterations": 5,
                        "conversation_compaction": False,
                        "retry": 1,
                        "multiple_tool_calls": True,
                    }
                )
            ],
            "links": [],
        }

        result = fixer.fix_tool_orchestrator_blocks(agent)

        defaults = result["nodes"][0]["input_default"]
        assert defaults["agent_mode_max_iterations"] == 5
        assert defaults["conversation_compaction"] is False
        assert defaults["retry"] == 1
        assert defaults["multiple_tool_calls"] is True
        assert len(fixer.fixes_applied) == 0

    def test_partial_defaults(self):
        """Only missing fields are filled; existing ones are kept."""
        fixer = AgentFixer()
        agent = {
            "nodes": [
                _make_sdm_node(
                    input_default={
                        "agent_mode_max_iterations": 10,
                    }
                )
            ],
            "links": [],
        }

        result = fixer.fix_tool_orchestrator_blocks(agent)

        defaults = result["nodes"][0]["input_default"]
        assert defaults["agent_mode_max_iterations"] == 10  # kept
        assert defaults["conversation_compaction"] is True  # filled
        assert defaults["retry"] == 3  # filled
        assert defaults["multiple_tool_calls"] is False  # filled
        assert len(fixer.fixes_applied) == 3

    def test_skips_non_sdm_nodes(self):
        """Non-ToolOrchestrator nodes are untouched."""
        fixer = AgentFixer()
        other_node = {
            "id": _uid(),
            "block_id": AGENT_INPUT_BLOCK_ID,
            "input_default": {"name": "test"},
            "metadata": {},
        }
        agent = {"nodes": [other_node], "links": []}

        result = fixer.fix_tool_orchestrator_blocks(agent)

        assert "agent_mode_max_iterations" not in result["nodes"][0]["input_default"]
        assert len(fixer.fixes_applied) == 0

    def test_handles_missing_input_default(self):
        """Node with no input_default key gets one created."""
        fixer = AgentFixer()
        node = {
            "id": _uid(),
            "block_id": TOOL_ORCHESTRATOR_BLOCK_ID,
            "metadata": {},
        }
        agent = {"nodes": [node], "links": []}

        result = fixer.fix_tool_orchestrator_blocks(agent)

        assert "input_default" in result["nodes"][0]
        assert result["nodes"][0]["input_default"]["agent_mode_max_iterations"] == 10

    def test_handles_none_input_default(self):
        """Node with input_default set to None gets a dict created."""
        fixer = AgentFixer()
        node = {
            "id": _uid(),
            "block_id": TOOL_ORCHESTRATOR_BLOCK_ID,
            "input_default": None,
            "metadata": {},
        }
        agent = {"nodes": [node], "links": []}

        result = fixer.fix_tool_orchestrator_blocks(agent)

        assert isinstance(result["nodes"][0]["input_default"], dict)
        assert result["nodes"][0]["input_default"]["agent_mode_max_iterations"] == 10

    def test_treats_none_values_as_missing(self):
        """Explicit None values are overwritten with defaults."""
        fixer = AgentFixer()
        agent = {
            "nodes": [
                _make_sdm_node(
                    input_default={
                        "agent_mode_max_iterations": None,
                        "conversation_compaction": None,
                        "retry": 3,
                        "multiple_tool_calls": False,
                    }
                )
            ],
            "links": [],
        }

        result = fixer.fix_tool_orchestrator_blocks(agent)

        defaults = result["nodes"][0]["input_default"]
        assert defaults["agent_mode_max_iterations"] == 10  # None → default
        assert defaults["conversation_compaction"] is True  # None → default
        assert defaults["retry"] == 3  # kept
        assert defaults["multiple_tool_calls"] is False  # kept
        assert len(fixer.fixes_applied) == 2

    def test_multiple_sdm_nodes(self):
        """Multiple SDM nodes are all fixed independently."""
        fixer = AgentFixer()
        agent = {
            "nodes": [
                _make_sdm_node(input_default={"agent_mode_max_iterations": 3}),
                _make_sdm_node(input_default={}),
            ],
            "links": [],
        }

        result = fixer.fix_tool_orchestrator_blocks(agent)

        # First node: 3 defaults filled (agent_mode was already set)
        assert result["nodes"][0]["input_default"]["agent_mode_max_iterations"] == 3
        # Second node: all 4 defaults filled
        assert result["nodes"][1]["input_default"]["agent_mode_max_iterations"] == 10
        assert len(fixer.fixes_applied) == 7  # 3 + 4

    def test_registered_in_apply_all_fixes(self):
        """fix_tool_orchestrator_blocks runs as part of apply_all_fixes."""
        fixer = AgentFixer()
        agent = {
            "nodes": [_make_sdm_node()],
            "links": [],
        }

        result = fixer.apply_all_fixes(agent)

        defaults = result["nodes"][0]["input_default"]
        assert defaults["agent_mode_max_iterations"] == 10
        assert any("ToolOrchestratorBlock" in fix for fix in fixer.fixes_applied)


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------


class TestValidateToolOrchestratorBlocks:
    """Tests for AgentValidator.validate_tool_orchestrator_blocks()."""

    def test_valid_sdm_with_tools(self):
        """SDM with downstream tool links passes validation."""
        validator = AgentValidator()
        agent = _make_orchestrator_agent()

        result = validator.validate_tool_orchestrator_blocks(agent)

        assert result is True
        assert len(validator.errors) == 0

    def test_sdm_without_tools_fails(self):
        """SDM with no 'tools' links fails validation."""
        validator = AgentValidator()
        sdm = _make_sdm_node()
        agent = {
            "nodes": [sdm],
            "links": [],  # no tool links
        }

        result = validator.validate_tool_orchestrator_blocks(agent)

        assert result is False
        assert len(validator.errors) == 1
        assert "no downstream tool blocks" in validator.errors[0]

    def test_sdm_with_non_tools_links_fails(self):
        """Links that don't use source_name='tools' don't count."""
        validator = AgentValidator()
        sdm = _make_sdm_node()
        other = _make_agent_executor_node()
        agent = {
            "nodes": [sdm, other],
            "links": [
                # Link from 'finished' output, not 'tools'
                _link(sdm["id"], "finished", other["id"], "query"),
            ],
        }

        result = validator.validate_tool_orchestrator_blocks(agent)

        assert result is False
        assert len(validator.errors) == 1

    def test_no_sdm_nodes_passes(self):
        """Agent without ToolOrchestrator nodes passes trivially."""
        validator = AgentValidator()
        agent = {
            "nodes": [_make_input_node(), _make_output_node()],
            "links": [],
        }

        result = validator.validate_tool_orchestrator_blocks(agent)

        assert result is True
        assert len(validator.errors) == 0

    def test_error_includes_customized_name(self):
        """Error message includes the node's customized_name if set."""
        validator = AgentValidator()
        sdm = _make_sdm_node(
            metadata={
                "position": {"x": 0, "y": 0},
                "customized_name": "My Orchestrator",
            }
        )
        agent = {"nodes": [sdm], "links": []}

        validator.validate_tool_orchestrator_blocks(agent)

        assert "My Orchestrator" in validator.errors[0]

    def test_multiple_sdm_nodes_mixed(self):
        """One valid and one invalid SDM node: only the invalid one errors."""
        validator = AgentValidator()
        sdm_valid = _make_sdm_node()
        sdm_invalid = _make_sdm_node()
        tool = _make_agent_executor_node()

        agent = {
            "nodes": [sdm_valid, sdm_invalid, tool],
            "links": [
                _link(sdm_valid["id"], "tools", tool["id"], "query"),
                # sdm_invalid has no tool links
            ],
        }

        result = validator.validate_tool_orchestrator_blocks(agent)

        assert result is False
        assert len(validator.errors) == 1
        assert sdm_invalid["id"] in validator.errors[0]

    def test_sdm_with_traditional_mode_fails(self):
        """agent_mode_max_iterations=0 (traditional mode) is rejected."""
        validator = AgentValidator()
        sdm = _make_sdm_node(input_default={"agent_mode_max_iterations": 0})
        tool = _make_agent_executor_node()
        agent = {
            "nodes": [sdm, tool],
            "links": [_link(sdm["id"], "tools", tool["id"], "query")],
        }

        result = validator.validate_tool_orchestrator_blocks(agent)

        assert result is False
        assert any("agent_mode_max_iterations=0" in e for e in validator.errors)

    def test_sdm_with_infinite_iterations_passes(self):
        """agent_mode_max_iterations=-1 (infinite mode) is valid."""
        validator = AgentValidator()
        sdm = _make_sdm_node(input_default={"agent_mode_max_iterations": -1})
        tool = _make_agent_executor_node()
        agent = {
            "nodes": [sdm, tool],
            "links": [_link(sdm["id"], "tools", tool["id"], "query")],
        }

        result = validator.validate_tool_orchestrator_blocks(agent)

        assert result is True
        assert len(validator.errors) == 0

    def test_sdm_with_high_iterations_fails(self):
        """agent_mode_max_iterations > 100 is rejected as unusually high."""
        validator = AgentValidator()
        sdm = _make_sdm_node(input_default={"agent_mode_max_iterations": 999999})
        tool = _make_agent_executor_node()
        agent = {
            "nodes": [sdm, tool],
            "links": [_link(sdm["id"], "tools", tool["id"], "query")],
        }

        result = validator.validate_tool_orchestrator_blocks(agent)

        assert result is False
        assert any("unusually high" in e for e in validator.errors)

    def test_sdm_with_string_iterations_fails(self):
        """Non-integer agent_mode_max_iterations (e.g. string) is rejected."""
        validator = AgentValidator()
        sdm = _make_sdm_node(input_default={"agent_mode_max_iterations": "10"})
        tool = _make_agent_executor_node()
        agent = {
            "nodes": [sdm, tool],
            "links": [_link(sdm["id"], "tools", tool["id"], "query")],
        }

        result = validator.validate_tool_orchestrator_blocks(agent)

        assert result is False
        assert any("non-integer" in e for e in validator.errors)

    def test_sdm_with_negative_iterations_below_minus_one_fails(self):
        """agent_mode_max_iterations < -1 is rejected."""
        validator = AgentValidator()
        sdm = _make_sdm_node(input_default={"agent_mode_max_iterations": -5})
        tool = _make_agent_executor_node()
        agent = {
            "nodes": [sdm, tool],
            "links": [_link(sdm["id"], "tools", tool["id"], "query")],
        }

        result = validator.validate_tool_orchestrator_blocks(agent)

        assert result is False
        assert any("invalid" in e and "-5" in e for e in validator.errors)

    def test_sdm_with_only_interface_block_links_fails(self):
        """Links to AgentInput/OutputBlocks don't count as tool connections."""
        validator = AgentValidator()
        sdm = _make_sdm_node()
        input_node = _make_input_node()
        output_node = _make_output_node()
        agent = {
            "nodes": [sdm, input_node, output_node],
            "links": [
                # These link to interface blocks, not real tools
                _link(sdm["id"], "tools", input_node["id"], "name"),
                _link(sdm["id"], "tools", output_node["id"], "value"),
            ],
        }

        result = validator.validate_tool_orchestrator_blocks(agent)

        assert result is False
        assert len(validator.errors) == 1
        assert "no downstream tool blocks" in validator.errors[0]

    def test_registered_in_validate(self):
        """validate_tool_orchestrator_blocks runs as part of validate()."""
        validator = AgentValidator()
        sdm = _make_sdm_node()
        agent = {
            "id": _uid(),
            "version": 1,
            "is_active": True,
            "name": "Test",
            "description": "test",
            "nodes": [sdm, _make_input_node(), _make_output_node()],
            "links": [],
        }

        # Build a minimal blocks list with the SDM block info
        blocks = [
            {
                "id": TOOL_ORCHESTRATOR_BLOCK_ID,
                "name": "ToolOrchestratorBlock",
                "inputSchema": {"properties": {"prompt": {"type": "string"}}},
                "outputSchema": {
                    "properties": {
                        "tools": {},
                        "finished": {"type": "string"},
                        "conversations": {"type": "array"},
                    }
                },
            },
            {
                "id": AGENT_INPUT_BLOCK_ID,
                "name": "AgentInputBlock",
                "inputSchema": {
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                "outputSchema": {"properties": {"result": {}}},
            },
            {
                "id": AGENT_OUTPUT_BLOCK_ID,
                "name": "AgentOutputBlock",
                "inputSchema": {
                    "properties": {
                        "name": {"type": "string"},
                        "value": {},
                    },
                    "required": ["name"],
                },
                "outputSchema": {"properties": {"output": {}}},
            },
        ]

        is_valid, error_msg = validator.validate(agent, blocks)

        assert is_valid is False
        assert error_msg is not None
        assert "no downstream tool blocks" in error_msg


# ---------------------------------------------------------------------------
# E2E pipeline test: fix → validate for a complete orchestrator agent
# ---------------------------------------------------------------------------


class TestToolOrchestratorE2EPipeline:
    """End-to-end tests: build agent JSON → fix → validate."""

    def test_orchestrator_agent_fix_then_validate(self):
        """A well-formed orchestrator agent passes fix + validate."""
        agent = _make_orchestrator_agent()

        # Fix
        fixer = AgentFixer()
        fixed = fixer.apply_all_fixes(agent)

        # Verify defaults were applied
        sdm_nodes = [
            n for n in fixed["nodes"] if n["block_id"] == TOOL_ORCHESTRATOR_BLOCK_ID
        ]
        assert len(sdm_nodes) == 1
        assert sdm_nodes[0]["input_default"]["agent_mode_max_iterations"] == 10
        assert sdm_nodes[0]["input_default"]["conversation_compaction"] is True

        # Validate (standalone SDM check)
        validator = AgentValidator()
        assert validator.validate_tool_orchestrator_blocks(fixed) is True

    def test_bare_sdm_no_tools_fix_then_validate(self):
        """SDM without tools: fixer fills defaults, validator catches error."""
        input_node = _make_input_node()
        sdm_node = _make_sdm_node()
        output_node = _make_output_node()

        agent = {
            "id": _uid(),
            "version": 1,
            "is_active": True,
            "name": "Bare SDM Agent",
            "description": "SDM with no tools",
            "nodes": [input_node, sdm_node, output_node],
            "links": [
                _link(input_node["id"], "result", sdm_node["id"], "prompt"),
                _link(sdm_node["id"], "finished", output_node["id"], "value"),
            ],
        }

        # Fix fills defaults fine
        fixer = AgentFixer()
        fixed = fixer.apply_all_fixes(agent)
        assert fixed["nodes"][1]["input_default"]["agent_mode_max_iterations"] == 10

        # Validate catches missing tools
        validator = AgentValidator()
        assert validator.validate_tool_orchestrator_blocks(fixed) is False
        assert any("no downstream tool blocks" in e for e in validator.errors)

    def test_sdm_with_user_set_bounded_iterations(self):
        """User-set bounded iterations are preserved through fix pipeline."""
        agent = _make_orchestrator_agent()
        # Simulate user setting bounded iterations
        for node in agent["nodes"]:
            if node["block_id"] == TOOL_ORCHESTRATOR_BLOCK_ID:
                node["input_default"]["agent_mode_max_iterations"] = 5
                node["input_default"]["sys_prompt"] = "You are a helpful orchestrator"

        fixer = AgentFixer()
        fixed = fixer.apply_all_fixes(agent)

        sdm = next(
            n for n in fixed["nodes"] if n["block_id"] == TOOL_ORCHESTRATOR_BLOCK_ID
        )
        assert sdm["input_default"]["agent_mode_max_iterations"] == 5
        assert sdm["input_default"]["sys_prompt"] == "You are a helpful orchestrator"
        # Other defaults still filled
        assert sdm["input_default"]["conversation_compaction"] is True
        assert sdm["input_default"]["retry"] == 3

    def test_full_pipeline_with_blocks_list(self):
        """Full validate() with blocks list for a valid orchestrator agent."""
        agent = _make_orchestrator_agent()
        fixer = AgentFixer()
        fixed = fixer.apply_all_fixes(agent)

        blocks = [
            {
                "id": TOOL_ORCHESTRATOR_BLOCK_ID,
                "name": "ToolOrchestratorBlock",
                "inputSchema": {
                    "properties": {
                        "prompt": {"type": "string"},
                        "model": {"type": "object"},
                        "sys_prompt": {"type": "string"},
                        "agent_mode_max_iterations": {"type": "integer"},
                        "conversation_compaction": {"type": "boolean"},
                        "retry": {"type": "integer"},
                        "multiple_tool_calls": {"type": "boolean"},
                    },
                    "required": ["prompt"],
                },
                "outputSchema": {
                    "properties": {
                        "tools": {},
                        "finished": {"type": "string"},
                        "conversations": {"type": "array"},
                    }
                },
            },
            {
                "id": AGENT_EXECUTOR_BLOCK_ID,
                "name": "AgentExecutorBlock",
                "inputSchema": {
                    "properties": {
                        "graph_id": {"type": "string"},
                        "graph_version": {"type": "integer"},
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                        "user_id": {"type": "string"},
                        "inputs": {"type": "object"},
                        "query": {"type": "string"},
                    },
                    "required": ["graph_id"],
                },
                "outputSchema": {
                    "properties": {"result": {"type": "string"}},
                },
            },
            {
                "id": AGENT_INPUT_BLOCK_ID,
                "name": "AgentInputBlock",
                "inputSchema": {
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                "outputSchema": {"properties": {"result": {}}},
            },
            {
                "id": AGENT_OUTPUT_BLOCK_ID,
                "name": "AgentOutputBlock",
                "inputSchema": {
                    "properties": {
                        "name": {"type": "string"},
                        "value": {},
                    },
                    "required": ["name"],
                },
                "outputSchema": {"properties": {"output": {}}},
            },
        ]

        validator = AgentValidator()
        is_valid, error_msg = validator.validate(fixed, blocks)

        # Full graph validation should pass
        assert is_valid, f"Validation failed: {error_msg}"

        # SDM-specific validation should pass (has tool links)
        sdm_errors = [e for e in validator.errors if "ToolOrchestratorBlock" in e]
        assert len(sdm_errors) == 0, f"Unexpected SDM errors: {sdm_errors}"
