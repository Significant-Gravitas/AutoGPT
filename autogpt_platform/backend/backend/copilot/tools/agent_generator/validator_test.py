"""Unit tests for AgentValidator."""

from .helpers import (
    AGENT_EXECUTOR_BLOCK_ID,
    AGENT_INPUT_BLOCK_ID,
    AGENT_OUTPUT_BLOCK_ID,
    MCP_TOOL_BLOCK_ID,
    generate_uuid,
)
from .validator import AgentValidator


def _make_agent(
    nodes: list | None = None,
    links: list | None = None,
    agent_id: str | None = None,
) -> dict:
    """Build a minimal agent dict for testing."""
    return {
        "id": agent_id or generate_uuid(),
        "name": "Test Agent",
        "nodes": nodes or [],
        "links": links or [],
    }


def _make_node(
    node_id: str | None = None,
    block_id: str = "block-1",
    input_default: dict | None = None,
    position: tuple[int, int] = (0, 0),
) -> dict:
    return {
        "id": node_id or generate_uuid(),
        "block_id": block_id,
        "input_default": input_default or {},
        "metadata": {"position": {"x": position[0], "y": position[1]}},
    }


def _make_link(
    link_id: str | None = None,
    source_id: str = "",
    source_name: str = "output",
    sink_id: str = "",
    sink_name: str = "input",
) -> dict:
    return {
        "id": link_id or generate_uuid(),
        "source_id": source_id,
        "source_name": source_name,
        "sink_id": sink_id,
        "sink_name": sink_name,
    }


def _make_block(
    block_id: str = "block-1",
    name: str = "TestBlock",
    input_schema: dict | None = None,
    output_schema: dict | None = None,
    categories: list | None = None,
    static_output: bool = False,
) -> dict:
    return {
        "id": block_id,
        "name": name,
        "inputSchema": input_schema or {"properties": {}, "required": []},
        "outputSchema": output_schema or {"properties": {}},
        "categories": categories or [],
        "staticOutput": static_output,
    }


# ============================================================================
# validate_block_existence
# ============================================================================


class TestValidateBlockExistence:
    def test_valid_blocks_pass(self):
        v = AgentValidator()
        node = _make_node(block_id="b1")
        block = _make_block(block_id="b1")
        agent = _make_agent(nodes=[node])

        assert v.validate_block_existence(agent, [block]) is True
        assert v.errors == []

    def test_missing_block_fails(self):
        v = AgentValidator()
        node = _make_node(block_id="nonexistent")
        agent = _make_agent(nodes=[node])

        assert v.validate_block_existence(agent, []) is False
        assert len(v.errors) == 1
        assert "does not exist" in v.errors[0]

    def test_missing_block_id_field(self):
        v = AgentValidator()
        node = {"id": "n1", "input_default": {}, "metadata": {}}
        agent = _make_agent(nodes=[node])

        assert v.validate_block_existence(agent, []) is False
        assert "missing a 'block_id'" in v.errors[0]


# ============================================================================
# validate_link_node_references
# ============================================================================


class TestValidateLinkNodeReferences:
    def test_valid_references_pass(self):
        v = AgentValidator()
        n1 = _make_node(node_id="n1")
        n2 = _make_node(node_id="n2")
        link = _make_link(source_id="n1", sink_id="n2")
        agent = _make_agent(nodes=[n1, n2], links=[link])

        assert v.validate_link_node_references(agent) is True
        assert v.errors == []

    def test_invalid_source_fails(self):
        v = AgentValidator()
        n1 = _make_node(node_id="n1")
        link = _make_link(source_id="missing", sink_id="n1")
        agent = _make_agent(nodes=[n1], links=[link])

        assert v.validate_link_node_references(agent) is False
        assert any("source_id" in e for e in v.errors)

    def test_invalid_sink_fails(self):
        v = AgentValidator()
        n1 = _make_node(node_id="n1")
        link = _make_link(source_id="n1", sink_id="missing")
        agent = _make_agent(nodes=[n1], links=[link])

        assert v.validate_link_node_references(agent) is False
        assert any("sink_id" in e for e in v.errors)


# ============================================================================
# validate_required_inputs
# ============================================================================


class TestValidateRequiredInputs:
    def test_satisfied_by_default_passes(self):
        v = AgentValidator()
        block = _make_block(
            block_id="b1",
            input_schema={
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        )
        node = _make_node(block_id="b1", input_default={"url": "http://example.com"})
        agent = _make_agent(nodes=[node])

        assert v.validate_required_inputs(agent, [block]) is True
        assert v.errors == []

    def test_satisfied_by_link_passes(self):
        v = AgentValidator()
        block = _make_block(
            block_id="b1",
            input_schema={
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        )
        node = _make_node(node_id="n1", block_id="b1")
        link = _make_link(source_id="n2", sink_id="n1", sink_name="url")
        agent = _make_agent(nodes=[node], links=[link])

        assert v.validate_required_inputs(agent, [block]) is True

    def test_missing_required_input_fails(self):
        v = AgentValidator()
        block = _make_block(
            block_id="b1",
            input_schema={
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        )
        node = _make_node(block_id="b1", input_default={})
        agent = _make_agent(nodes=[node])

        assert v.validate_required_inputs(agent, [block]) is False
        assert any("missing required input" in e for e in v.errors)

    def test_credentials_always_allowed_missing(self):
        v = AgentValidator()
        block = _make_block(
            block_id="b1",
            input_schema={
                "properties": {"credentials": {"type": "object"}},
                "required": ["credentials"],
            },
        )
        node = _make_node(block_id="b1", input_default={})
        agent = _make_agent(nodes=[node])

        assert v.validate_required_inputs(agent, [block]) is True


# ============================================================================
# validate_data_type_compatibility
# ============================================================================


class TestValidateDataTypeCompatibility:
    def test_matching_types_pass(self):
        v = AgentValidator()
        src_block = _make_block(
            block_id="src-b",
            output_schema={"properties": {"out": {"type": "string"}}},
        )
        sink_block = _make_block(
            block_id="sink-b",
            input_schema={"properties": {"inp": {"type": "string"}}, "required": []},
        )
        src_node = _make_node(node_id="n1", block_id="src-b")
        sink_node = _make_node(node_id="n2", block_id="sink-b")
        link = _make_link(
            source_id="n1", source_name="out", sink_id="n2", sink_name="inp"
        )
        agent = _make_agent(nodes=[src_node, sink_node], links=[link])

        assert (
            v.validate_data_type_compatibility(agent, [src_block, sink_block]) is True
        )

    def test_int_number_compatible(self):
        v = AgentValidator()
        src_block = _make_block(
            block_id="src-b",
            output_schema={"properties": {"out": {"type": "integer"}}},
        )
        sink_block = _make_block(
            block_id="sink-b",
            input_schema={"properties": {"inp": {"type": "number"}}, "required": []},
        )
        src_node = _make_node(node_id="n1", block_id="src-b")
        sink_node = _make_node(node_id="n2", block_id="sink-b")
        link = _make_link(
            source_id="n1", source_name="out", sink_id="n2", sink_name="inp"
        )
        agent = _make_agent(nodes=[src_node, sink_node], links=[link])

        assert (
            v.validate_data_type_compatibility(agent, [src_block, sink_block]) is True
        )

    def test_mismatched_types_fail(self):
        v = AgentValidator()
        src_block = _make_block(
            block_id="src-b",
            output_schema={"properties": {"out": {"type": "string"}}},
        )
        sink_block = _make_block(
            block_id="sink-b",
            input_schema={"properties": {"inp": {"type": "integer"}}, "required": []},
        )
        src_node = _make_node(node_id="n1", block_id="src-b")
        sink_node = _make_node(node_id="n2", block_id="sink-b")
        link = _make_link(
            source_id="n1", source_name="out", sink_id="n2", sink_name="inp"
        )
        agent = _make_agent(nodes=[src_node, sink_node], links=[link])

        assert (
            v.validate_data_type_compatibility(agent, [src_block, sink_block]) is False
        )
        assert any("mismatch" in e.lower() for e in v.errors)


# ============================================================================
# validate_source_output_existence
# ============================================================================


class TestValidateSourceOutputExistence:
    def test_valid_source_output_passes(self):
        v = AgentValidator()
        block = _make_block(
            block_id="b1",
            output_schema={"properties": {"result": {"type": "string"}}},
        )
        node = _make_node(node_id="n1", block_id="b1")
        link = _make_link(source_id="n1", source_name="result", sink_id="n2")
        agent = _make_agent(nodes=[node], links=[link])

        assert v.validate_source_output_existence(agent, [block]) is True

    def test_invalid_source_output_fails(self):
        v = AgentValidator()
        block = _make_block(
            block_id="b1",
            output_schema={"properties": {"result": {"type": "string"}}},
        )
        node = _make_node(node_id="n1", block_id="b1")
        link = _make_link(source_id="n1", source_name="nonexistent", sink_id="n2")
        agent = _make_agent(nodes=[node], links=[link])

        assert v.validate_source_output_existence(agent, [block]) is False
        assert any("does not exist" in e for e in v.errors)


# ============================================================================
# validate_prompt_double_curly_braces_spaces
# ============================================================================


class TestValidatePromptDoubleCurlyBracesSpaces:
    def test_no_spaces_passes(self):
        v = AgentValidator()
        node = _make_node(input_default={"prompt": "Hello {{name}}!"})
        agent = _make_agent(nodes=[node])

        assert v.validate_prompt_double_curly_braces_spaces(agent) is True

    def test_spaces_in_braces_fails(self):
        v = AgentValidator()
        node = _make_node(input_default={"prompt": "Hello {{user name}}!"})
        agent = _make_agent(nodes=[node])

        assert v.validate_prompt_double_curly_braces_spaces(agent) is False
        assert any("spaces" in e for e in v.errors)


# ============================================================================
# validate_agent_executor_block_schemas
# ============================================================================


class TestValidateAgentExecutorBlockSchemas:
    def test_valid_schemas_pass(self):
        v = AgentValidator()
        node = _make_node(
            block_id=AGENT_EXECUTOR_BLOCK_ID,
            input_default={
                "graph_id": generate_uuid(),
                "input_schema": {"properties": {"q": {"type": "string"}}},
                "output_schema": {"properties": {"result": {"type": "string"}}},
            },
        )
        agent = _make_agent(nodes=[node])

        assert v.validate_agent_executor_block_schemas(agent) is True
        assert v.errors == []

    def test_empty_input_schema_fails(self):
        v = AgentValidator()
        node = _make_node(
            block_id=AGENT_EXECUTOR_BLOCK_ID,
            input_default={
                "graph_id": generate_uuid(),
                "input_schema": {},
                "output_schema": {"properties": {"result": {"type": "string"}}},
            },
        )
        agent = _make_agent(nodes=[node])

        assert v.validate_agent_executor_block_schemas(agent) is False
        assert any("empty input_schema" in e for e in v.errors)

    def test_missing_output_schema_fails(self):
        v = AgentValidator()
        node = _make_node(
            block_id=AGENT_EXECUTOR_BLOCK_ID,
            input_default={
                "graph_id": generate_uuid(),
                "input_schema": {"properties": {"q": {"type": "string"}}},
            },
        )
        agent = _make_agent(nodes=[node])

        assert v.validate_agent_executor_block_schemas(agent) is False
        assert any("output_schema" in e for e in v.errors)


# ============================================================================
# validate_agent_executor_blocks
# ============================================================================


class TestValidateAgentExecutorBlocks:
    def test_missing_graph_id_fails(self):
        v = AgentValidator()
        node = _make_node(
            block_id=AGENT_EXECUTOR_BLOCK_ID,
            input_default={},
        )
        agent = _make_agent(nodes=[node])

        assert v.validate_agent_executor_blocks(agent) is False
        assert any("graph_id" in e for e in v.errors)

    def test_valid_graph_id_passes(self):
        v = AgentValidator()
        node = _make_node(
            block_id=AGENT_EXECUTOR_BLOCK_ID,
            input_default={"graph_id": generate_uuid()},
        )
        agent = _make_agent(nodes=[node])

        assert v.validate_agent_executor_blocks(agent) is True

    def test_version_mismatch_with_library_agent(self):
        v = AgentValidator()
        lib_id = generate_uuid()
        node = _make_node(
            node_id="n1",
            block_id=AGENT_EXECUTOR_BLOCK_ID,
            input_default={"graph_id": lib_id, "graph_version": 1},
        )
        agent = _make_agent(nodes=[node])

        library_agents = [{"graph_id": lib_id, "graph_version": 3, "name": "Sub Agent"}]

        assert v.validate_agent_executor_blocks(agent, library_agents) is False
        assert any("mismatched graph_version" in e for e in v.errors)

    def test_required_input_satisfied_by_schema_default_passes(self):
        """Required sub-agent inputs filled with their schema default by the fixer
        should NOT be flagged as missing."""
        v = AgentValidator()
        lib_id = generate_uuid()
        node = _make_node(
            node_id="n1",
            block_id=AGENT_EXECUTOR_BLOCK_ID,
            input_default={
                "graph_id": lib_id,
                "input_schema": {
                    "properties": {"mode": {"type": "string", "default": "fast"}}
                },
                "inputs": {"mode": "fast"},  # fixer populated with schema default
            },
        )
        agent = _make_agent(nodes=[node])
        library_agents = [
            {
                "graph_id": lib_id,
                "graph_version": 1,
                "name": "Sub",
                "input_schema": {
                    "required": ["mode"],
                    "properties": {"mode": {"type": "string", "default": "fast"}},
                },
                "output_schema": {},
            }
        ]

        assert v.validate_agent_executor_blocks(agent, library_agents) is True
        assert v.errors == []

    def test_required_input_not_linked_and_no_default_fails(self):
        """Required sub-agent inputs without a link or schema default must fail."""
        v = AgentValidator()
        lib_id = generate_uuid()
        node = _make_node(
            node_id="n1",
            block_id=AGENT_EXECUTOR_BLOCK_ID,
            input_default={
                "graph_id": lib_id,
                "input_schema": {"properties": {"query": {"type": "string"}}},
                "inputs": {},
            },
        )
        agent = _make_agent(nodes=[node])
        library_agents = [
            {
                "graph_id": lib_id,
                "graph_version": 1,
                "name": "Sub",
                "input_schema": {
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}},
                },
                "output_schema": {},
            }
        ]

        assert v.validate_agent_executor_blocks(agent, library_agents) is False
        assert any("missing required sub-agent input" in e for e in v.errors)


# ============================================================================
# validate_io_blocks
# ============================================================================


class TestValidateIoBlocks:
    def test_missing_input_block_reports_error(self):
        v = AgentValidator()
        # Agent has output block but no input block
        node = _make_node(block_id=AGENT_OUTPUT_BLOCK_ID)
        agent = _make_agent(nodes=[node])

        assert v.validate_io_blocks(agent) is False
        assert len(v.errors) == 1
        assert "AgentInputBlock" in v.errors[0]

    def test_missing_output_block_reports_error(self):
        v = AgentValidator()
        # Agent has input block but no output block
        node = _make_node(block_id=AGENT_INPUT_BLOCK_ID)
        agent = _make_agent(nodes=[node])

        assert v.validate_io_blocks(agent) is False
        assert len(v.errors) == 1
        assert "AgentOutputBlock" in v.errors[0]

    def test_missing_both_io_blocks_reports_two_errors(self):
        v = AgentValidator()
        node = _make_node(block_id="some-other-block")
        agent = _make_agent(nodes=[node])

        assert v.validate_io_blocks(agent) is False
        assert len(v.errors) == 2

    def test_both_io_blocks_present_no_error(self):
        v = AgentValidator()
        input_node = _make_node(block_id=AGENT_INPUT_BLOCK_ID)
        output_node = _make_node(block_id=AGENT_OUTPUT_BLOCK_ID)
        agent = _make_agent(nodes=[input_node, output_node])

        assert v.validate_io_blocks(agent) is True
        assert v.errors == []

    def test_empty_agent_reports_both_missing(self):
        v = AgentValidator()
        agent = _make_agent(nodes=[])

        assert v.validate_io_blocks(agent) is False
        assert len(v.errors) == 2


# ============================================================================
# validate (integration)
# ============================================================================


class TestValidate:
    def test_valid_agent_passes(self):
        v = AgentValidator()
        block = _make_block(
            block_id="b1",
            input_schema={
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            output_schema={"properties": {"result": {"type": "string"}}},
        )
        input_block = _make_block(
            block_id=AGENT_INPUT_BLOCK_ID,
            name="AgentInputBlock",
            input_schema={
                "properties": {
                    "name": {"type": "string"},
                    "title": {"type": "string"},
                    "value": {},
                    "description": {"type": "string"},
                },
                "required": ["name"],
            },
            output_schema={"properties": {"result": {}}},
        )
        output_block = _make_block(
            block_id=AGENT_OUTPUT_BLOCK_ID,
            name="AgentOutputBlock",
            input_schema={
                "properties": {
                    "name": {"type": "string"},
                    "title": {"type": "string"},
                    "value": {},
                },
                "required": ["name"],
            },
        )
        input_node = _make_node(
            node_id="n-in",
            block_id=AGENT_INPUT_BLOCK_ID,
            input_default={"name": "url"},
        )
        n1 = _make_node(
            node_id="n1", block_id="b1", input_default={"url": "http://example.com"}
        )
        n2 = _make_node(
            node_id="n2", block_id="b1", input_default={"url": "http://example2.com"}
        )
        output_node = _make_node(
            node_id="n-out",
            block_id=AGENT_OUTPUT_BLOCK_ID,
            input_default={"name": "result"},
        )
        link = _make_link(
            source_id="n1", source_name="result", sink_id="n2", sink_name="url"
        )
        agent = _make_agent(nodes=[input_node, n1, n2, output_node], links=[link])

        is_valid, error_message = v.validate(agent, [block, input_block, output_block])

        assert is_valid is True
        assert error_message is None

    def test_invalid_agent_returns_errors(self):
        v = AgentValidator()
        node = _make_node(block_id="nonexistent")
        agent = _make_agent(nodes=[node])

        is_valid, error_message = v.validate(agent, [])

        assert is_valid is False
        assert error_message is not None
        assert "does not exist" in error_message

    def test_empty_agent_fails_io_validation(self):
        v = AgentValidator()
        agent = _make_agent()

        is_valid, error_message = v.validate(agent, [])

        assert is_valid is False
        assert error_message is not None
        assert "AgentInputBlock" in error_message
        assert "AgentOutputBlock" in error_message


class TestValidateSinkInputExistence:
    """Tests for validate_sink_input_existence."""

    def test_valid_sink_name_passes(self):
        v = AgentValidator()
        block = _make_block(
            block_id="b1",
            input_schema={"properties": {"url": {"type": "string"}}, "required": []},
        )
        node = _make_node(node_id="n1", block_id="b1")
        link = _make_link(
            source_id="src", source_name="out", sink_id="n1", sink_name="url"
        )
        agent = _make_agent(nodes=[node], links=[link])

        assert v.validate_sink_input_existence(agent, [block]) is True

    def test_invalid_sink_name_fails(self):
        v = AgentValidator()
        block = _make_block(
            block_id="b1",
            input_schema={"properties": {"url": {"type": "string"}}, "required": []},
        )
        node = _make_node(node_id="n1", block_id="b1")
        link = _make_link(
            source_id="src", source_name="out", sink_id="n1", sink_name="nonexistent"
        )
        agent = _make_agent(nodes=[node], links=[link])

        assert v.validate_sink_input_existence(agent, [block]) is False
        assert any("nonexistent" in e for e in v.errors)

    def test_valid_nested_link_passes(self):
        v = AgentValidator()
        block = _make_block(
            block_id="b1",
            input_schema={
                "properties": {
                    "config": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                    }
                },
                "required": [],
            },
        )
        node = _make_node(node_id="n1", block_id="b1")
        link = _make_link(
            source_id="src",
            source_name="out",
            sink_id="n1",
            sink_name="config_#_key",
        )
        agent = _make_agent(nodes=[node], links=[link])

        assert v.validate_sink_input_existence(agent, [block]) is True

    def test_invalid_nested_child_fails(self):
        v = AgentValidator()
        block = _make_block(
            block_id="b1",
            input_schema={
                "properties": {
                    "config": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                    }
                },
                "required": [],
            },
        )
        node = _make_node(node_id="n1", block_id="b1")
        link = _make_link(
            source_id="src",
            source_name="out",
            sink_id="n1",
            sink_name="config_#_missing",
        )
        agent = _make_agent(nodes=[node], links=[link])

        assert v.validate_sink_input_existence(agent, [block]) is False

    def test_unknown_input_default_key_fails(self):
        v = AgentValidator()
        block = _make_block(
            block_id="b1",
            input_schema={"properties": {"url": {"type": "string"}}, "required": []},
        )
        node = _make_node(
            node_id="n1", block_id="b1", input_default={"nonexistent_key": "value"}
        )
        agent = _make_agent(nodes=[node])

        assert v.validate_sink_input_existence(agent, [block]) is False
        assert any("nonexistent_key" in e for e in v.errors)

    def test_credentials_key_skipped(self):
        v = AgentValidator()
        block = _make_block(
            block_id="b1",
            input_schema={"properties": {"url": {"type": "string"}}, "required": []},
        )
        node = _make_node(
            node_id="n1",
            block_id="b1",
            input_default={
                "url": "http://example.com",
                "credentials": {"api_key": "x"},
            },
        )
        agent = _make_agent(nodes=[node])

        assert v.validate_sink_input_existence(agent, [block]) is True

    def test_agent_executor_dynamic_schema_passes(self):
        v = AgentValidator()
        block = _make_block(
            block_id=AGENT_EXECUTOR_BLOCK_ID,
            input_schema={
                "properties": {
                    "graph_id": {"type": "string"},
                    "input_schema": {"type": "object"},
                },
                "required": ["graph_id"],
            },
        )
        node = _make_node(
            node_id="n1",
            block_id=AGENT_EXECUTOR_BLOCK_ID,
            input_default={
                "graph_id": "abc",
                "input_schema": {
                    "properties": {"query": {"type": "string"}},
                    "required": [],
                },
            },
        )
        link = _make_link(
            source_id="src",
            source_name="out",
            sink_id="n1",
            sink_name="query",
        )
        agent = _make_agent(nodes=[node], links=[link])

        assert v.validate_sink_input_existence(agent, [block]) is True

    def test_input_default_nested_invalid_child_fails(self):
        v = AgentValidator()
        block = _make_block(
            block_id="b1",
            input_schema={
                "properties": {
                    "config": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                    }
                },
                "required": [],
            },
        )
        node = _make_node(
            node_id="n1",
            block_id="b1",
            input_default={"config_#_invalid_child": "value"},
        )
        agent = _make_agent(nodes=[node])

        assert v.validate_sink_input_existence(agent, [block]) is False
        assert any("invalid_child" in e for e in v.errors)

    def test_input_default_nested_valid_child_passes(self):
        v = AgentValidator()
        block = _make_block(
            block_id="b1",
            input_schema={
                "properties": {
                    "config": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                    }
                },
                "required": [],
            },
        )
        node = _make_node(
            node_id="n1",
            block_id="b1",
            input_default={"config_#_key": "value"},
        )
        agent = _make_agent(nodes=[node])

        assert v.validate_sink_input_existence(agent, [block]) is True


class TestValidateMCPToolBlocks:
    """Tests for validate_mcp_tool_blocks."""

    def test_missing_server_url_reports_error(self):
        v = AgentValidator()
        node = _make_node(
            block_id=MCP_TOOL_BLOCK_ID,
            input_default={"selected_tool": "my_tool"},
        )
        agent = _make_agent(nodes=[node])

        result = v.validate_mcp_tool_blocks(agent)

        assert result is False
        assert any("server_url" in e for e in v.errors)

    def test_missing_selected_tool_reports_error(self):
        v = AgentValidator()
        node = _make_node(
            block_id=MCP_TOOL_BLOCK_ID,
            input_default={"server_url": "https://mcp.example.com/sse"},
        )
        agent = _make_agent(nodes=[node])

        result = v.validate_mcp_tool_blocks(agent)

        assert result is False
        assert any("selected_tool" in e for e in v.errors)

    def test_valid_mcp_block_passes(self):
        v = AgentValidator()
        node = _make_node(
            block_id=MCP_TOOL_BLOCK_ID,
            input_default={
                "server_url": "https://mcp.example.com/sse",
                "selected_tool": "search",
                "tool_input_schema": {"properties": {"query": {"type": "string"}}},
                "tool_arguments": {},
            },
        )
        agent = _make_agent(nodes=[node])

        result = v.validate_mcp_tool_blocks(agent)

        assert result is True
        assert len(v.errors) == 0

    def test_both_missing_reports_two_errors(self):
        v = AgentValidator()
        node = _make_node(
            block_id=MCP_TOOL_BLOCK_ID,
            input_default={},
        )
        agent = _make_agent(nodes=[node])

        v.validate_mcp_tool_blocks(agent)

        assert len(v.errors) == 2
