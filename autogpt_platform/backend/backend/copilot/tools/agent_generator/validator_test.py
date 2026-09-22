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
    ui_type: str | None = None,
) -> dict:
    block: dict = {
        "id": block_id,
        "name": name,
        "inputSchema": input_schema or {"properties": {}, "required": []},
        "outputSchema": output_schema or {"properties": {}},
        "categories": categories or [],
        "staticOutput": static_output,
    }
    if ui_type is not None:
        block["uiType"] = ui_type
    return block


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

    def test_subclass_input_block_satisfies_requirement(self):
        # AgentGoogleDriveFileInputBlock is a subclass of AgentInputBlock with
        # a different block_id, but it still exposes a user-facing input, so
        # it should satisfy the input-block requirement on its own.
        v = AgentValidator()
        drive_input_block = _make_block(
            block_id="d3b32f15-6fd7-40e3-be52-e083f51b19a2",
            name="AgentGoogleDriveFileInputBlock",
            ui_type="Input",
        )
        output_block = _make_block(
            block_id=AGENT_OUTPUT_BLOCK_ID,
            name="AgentOutputBlock",
            ui_type="Output",
        )
        drive_node = _make_node(block_id="d3b32f15-6fd7-40e3-be52-e083f51b19a2")
        output_node = _make_node(block_id=AGENT_OUTPUT_BLOCK_ID)
        agent = _make_agent(nodes=[drive_node, output_node])

        assert v.validate_io_blocks(agent, [drive_input_block, output_block]) is True
        assert v.errors == []

    def test_subclass_output_block_satisfies_requirement(self):
        v = AgentValidator()
        input_block = _make_block(
            block_id=AGENT_INPUT_BLOCK_ID,
            name="AgentInputBlock",
            ui_type="Input",
        )
        custom_output_block = _make_block(
            block_id="custom-output-id",
            name="CustomOutputBlock",
            ui_type="Output",
        )
        input_node = _make_node(block_id=AGENT_INPUT_BLOCK_ID)
        output_node = _make_node(block_id="custom-output-id")
        agent = _make_agent(nodes=[input_node, output_node])

        assert v.validate_io_blocks(agent, [input_block, custom_output_block]) is True
        assert v.errors == []

    def test_trigger_block_satisfies_input_requirement(self):
        # A webhook trigger replaces the input block: a triggered agent is
        # started by an external event and needs no user-facing input block.
        v = AgentValidator()
        trigger_block = _make_block(
            block_id="webhook-trigger-id",
            name="GithubTriggerBlock",
            ui_type="Webhook",
        )
        output_block = _make_block(
            block_id=AGENT_OUTPUT_BLOCK_ID,
            name="AgentOutputBlock",
            ui_type="Output",
        )
        trigger_node = _make_node(block_id="webhook-trigger-id")
        output_node = _make_node(block_id=AGENT_OUTPUT_BLOCK_ID)
        agent = _make_agent(nodes=[trigger_node, output_node])

        assert v.validate_io_blocks(agent, [trigger_block, output_block]) is True
        assert v.errors == []

    def test_manual_webhook_trigger_satisfies_input_requirement(self):
        v = AgentValidator()
        trigger_block = _make_block(
            block_id="manual-webhook-id",
            name="ManualWebhookTriggerBlock",
            ui_type="Webhook (manual)",
        )
        output_block = _make_block(
            block_id=AGENT_OUTPUT_BLOCK_ID,
            name="AgentOutputBlock",
            ui_type="Output",
        )
        trigger_node = _make_node(block_id="manual-webhook-id")
        output_node = _make_node(block_id=AGENT_OUTPUT_BLOCK_ID)
        agent = _make_agent(nodes=[trigger_node, output_node])

        assert v.validate_io_blocks(agent, [trigger_block, output_block]) is True
        assert v.errors == []

    def test_trigger_does_not_satisfy_output_requirement(self):
        # The trigger only covers the input side; an output block is still
        # required.
        v = AgentValidator()
        trigger_block = _make_block(
            block_id="webhook-trigger-id",
            name="GithubTriggerBlock",
            ui_type="Webhook",
        )
        trigger_node = _make_node(block_id="webhook-trigger-id")
        agent = _make_agent(nodes=[trigger_node])

        assert v.validate_io_blocks(agent, [trigger_block]) is False
        assert len(v.errors) == 1
        assert "AgentOutputBlock" in v.errors[0]

    def test_missing_input_error_mentions_trigger_alternative(self):
        v = AgentValidator()
        node = _make_node(block_id=AGENT_OUTPUT_BLOCK_ID)
        agent = _make_agent(nodes=[node])

        assert v.validate_io_blocks(agent) is False
        assert len(v.errors) == 1
        assert "trigger block" in v.errors[0]


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


class TestMCPToolDynamicSinks:
    """Dynamic tool arguments declared by a node's tool_input_schema are valid
    sinks / input_default keys on MCPToolBlock nodes — the executor collects
    them into tool_arguments at run time."""

    def _mcp_block(self) -> dict:
        return _make_block(
            block_id=MCP_TOOL_BLOCK_ID,
            name="MCPToolBlock",
            input_schema={
                "properties": {
                    "server_url": {"type": "string"},
                    "selected_tool": {"type": "string"},
                    "tool_input_schema": {"type": "object"},
                    "tool_arguments": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                },
                "required": ["server_url"],
            },
        )

    def _mcp_node(self, **extra_defaults) -> dict:
        return _make_node(
            node_id="mcp-1",
            block_id=MCP_TOOL_BLOCK_ID,
            input_default={
                "server_url": "https://mcp.notion.com/mcp",
                "selected_tool": "notion-query-data-sources",
                "tool_input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["data"],
                },
                **extra_defaults,
            },
        )

    def test_dynamic_tool_arg_sink_passes(self):
        v = AgentValidator()
        link = _make_link(
            source_id="src", source_name="out", sink_id="mcp-1", sink_name="data"
        )
        agent = _make_agent(nodes=[self._mcp_node()], links=[link])
        assert v.validate_sink_input_existence(agent, [self._mcp_block()]) is True

    def test_unknown_sink_still_fails(self):
        v = AgentValidator()
        link = _make_link(
            source_id="src", source_name="out", sink_id="mcp-1", sink_name="bogus"
        )
        agent = _make_agent(nodes=[self._mcp_node()], links=[link])
        v_result = v.validate_sink_input_existence(agent, [self._mcp_block()])
        assert v_result is False
        assert any("bogus" in e for e in v.errors)

    def test_dynamic_tool_arg_in_input_default_passes(self):
        v = AgentValidator()
        node = self._mcp_node(limit=10)
        agent = _make_agent(nodes=[node])
        assert v.validate_sink_input_existence(agent, [self._mcp_block()]) is True


class TestNestedChainsThroughUntypedFields:
    """_#_ chains through untyped/Any levels are resolved dynamically at run
    time and must not be rejected statically; scalar levels still reject."""

    def _code_block(self) -> dict:
        return _make_block(
            block_id="code-1",
            name="ExecuteCodeBlock",
            output_schema={
                "properties": {
                    "main_result": {
                        "type": "object",
                        "properties": {
                            "text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "json": {"anyOf": [{}, {"type": "null"}]},
                        },
                    },
                    "response": {"type": "string"},
                }
            },
        )

    def _link(self, source_name: str) -> dict:
        return _make_link(
            source_id="n1", source_name=source_name, sink_id="n2", sink_name="input"
        )

    def _agent(self, source_name: str) -> dict:
        nodes = [
            _make_node(node_id="n1", block_id="code-1"),
            _make_node(node_id="n2", block_id="sink-1"),
        ]
        return _make_agent(nodes=nodes, links=[self._link(source_name)])

    def _blocks(self) -> list[dict]:
        return [
            self._code_block(),
            _make_block(
                block_id="sink-1",
                input_schema={"properties": {"input": {}}, "required": []},
            ),
        ]

    def test_chain_through_untyped_field_passes(self):
        v = AgentValidator()
        agent = self._agent("main_result_#_json_#_has_new")
        assert v.validate_source_output_existence(agent, self._blocks()) is True

    def test_declared_child_still_verified(self):
        v = AgentValidator()
        agent = self._agent("main_result_#_nonexistent")
        assert v.validate_source_output_existence(agent, self._blocks()) is False
        assert any("nonexistent" in e for e in v.errors)

    def test_chain_into_scalar_fails(self):
        v = AgentValidator()
        agent = self._agent("response_#_field")
        assert v.validate_source_output_existence(agent, self._blocks()) is False
        assert any("no extractable sub-fields" in e for e in v.errors)

    def test_chain_through_scalar_anyof_fails(self):
        v = AgentValidator()
        agent = self._agent("main_result_#_text_#_deeper")
        assert v.validate_source_output_existence(agent, self._blocks()) is False
        assert any("no extractable sub-fields" in e for e in v.errors)
        # _resolve_optional_union unwraps [string, null] to its non-null
        # branch, so the message names the concrete type — never 'None'.
        assert any("type 'string'" in e for e in v.errors)

    def test_sink_side_chain_through_untyped_field_passes(self):
        """The shared walker also guards sink chains — an untyped declared
        input accepts nested delivery."""
        v = AgentValidator()
        link = _make_link(
            source_id="n1",
            source_name="response",
            sink_id="n2",
            sink_name="input_#_nested_#_deep",
        )
        nodes = [
            _make_node(node_id="n1", block_id="code-1"),
            _make_node(node_id="n2", block_id="sink-1"),
        ]
        agent = _make_agent(nodes=nodes, links=[link])
        assert v.validate_sink_input_existence(agent, self._blocks()) is True


class TestNestedChainEdgeCases:
    """Regressions from review: empty schemas are present-but-untyped, and
    Optional[X] unions resolve to X for strict validation."""

    def _agent_with_source(self, source_name: str, output_schema: dict) -> tuple:
        blocks = [
            _make_block(block_id="src-1", output_schema=output_schema),
            _make_block(
                block_id="sink-1",
                input_schema={"properties": {"input": {}}, "required": []},
            ),
        ]
        nodes = [
            _make_node(node_id="n1", block_id="src-1"),
            _make_node(node_id="n2", block_id="sink-1"),
        ]
        link = _make_link(
            source_id="n1", source_name=source_name, sink_id="n2", sink_name="input"
        )
        return _make_agent(nodes=nodes, links=[link]), blocks

    def test_empty_schema_property_is_present_and_untyped(self):
        v = AgentValidator()
        agent, blocks = self._agent_with_source(
            "value_#_child", {"properties": {"value": {}}}
        )
        assert v.validate_source_output_existence(agent, blocks) is True

    def test_optional_declared_object_valid_child_passes(self):
        v = AgentValidator()
        schema = {
            "properties": {
                "cfg": {
                    "anyOf": [
                        {"type": "object", "properties": {"key": {"type": "string"}}},
                        {"type": "null"},
                    ]
                }
            }
        }
        agent, blocks = self._agent_with_source("cfg_#_key", schema)
        assert v.validate_source_output_existence(agent, blocks) is True

    def test_optional_declared_object_invalid_child_fails(self):
        v = AgentValidator()
        schema = {
            "properties": {
                "cfg": {
                    "anyOf": [
                        {"type": "object", "properties": {"key": {"type": "string"}}},
                        {"type": "null"},
                    ]
                }
            }
        }
        agent, blocks = self._agent_with_source("cfg_#_missing", schema)
        assert v.validate_source_output_existence(agent, blocks) is False
        assert any("missing" in e for e in v.errors)

    def test_optional_scalar_still_rejects_descent(self):
        v = AgentValidator()
        schema = {
            "properties": {"name": {"anyOf": [{"type": "string"}, {"type": "null"}]}}
        }
        agent, blocks = self._agent_with_source("name_#_x", schema)
        assert v.validate_source_output_existence(agent, blocks) is False


class TestDynamicRequiredInputs:
    """Required arguments declared in a node's dynamic schema (MCP tool /
    sub-agent input schema) must be enforced by validate_required_inputs —
    the static block schema says nothing about them."""

    def _mcp_node(self, input_default: dict) -> dict:
        return _make_node(block_id=MCP_TOOL_BLOCK_ID, input_default=input_default)

    def _mcp_block(self) -> dict:
        return _make_block(block_id=MCP_TOOL_BLOCK_ID, name="MCPToolBlock")

    def test_missing_required_dynamic_argument_fails(self):
        v = AgentValidator()
        node = self._mcp_node(
            {
                "server_url": "https://mcp.example.com/sse",
                "selected_tool": "search",
                "tool_input_schema": {
                    "properties": {"data": {"type": "string"}},
                    "required": ["data"],
                },
            }
        )
        agent = _make_agent(nodes=[node])

        assert v.validate_required_inputs(agent, [self._mcp_block()]) is False
        assert any("'data'" in e for e in v.errors)

    def test_required_dynamic_argument_via_default_passes(self):
        v = AgentValidator()
        node = self._mcp_node(
            {
                "server_url": "https://mcp.example.com/sse",
                "selected_tool": "search",
                "tool_input_schema": {
                    "properties": {"data": {"type": "string"}},
                    "required": ["data"],
                },
                "data": "hello",
            }
        )
        agent = _make_agent(nodes=[node])

        assert v.validate_required_inputs(agent, [self._mcp_block()]) is True

    def test_required_dynamic_argument_via_link_passes(self):
        v = AgentValidator()
        node = self._mcp_node(
            {
                "server_url": "https://mcp.example.com/sse",
                "selected_tool": "search",
                "tool_input_schema": {
                    "properties": {"data": {"type": "string"}},
                    "required": ["data"],
                },
            }
        )
        source = _make_node(block_id="block-1")
        link = _make_link(
            source_id=source["id"],
            source_name="output",
            sink_id=node["id"],
            sink_name="data",
        )
        agent = _make_agent(nodes=[source, node], links=[link])

        assert (
            v.validate_required_inputs(agent, [self._mcp_block(), _make_block()])
            is True
        )


class TestAdditionalPropertiesChains:
    """Levels that allow additionalProperties accept the remaining chain —
    the executor delivers arbitrary keys into such dicts at run time."""

    def _agent_and_blocks(self, output_schema: dict) -> tuple[dict, list[dict]]:
        nodes = [
            _make_node(node_id="n1", block_id="src-1"),
            _make_node(node_id="n2", block_id="sink-1"),
        ]
        link = _make_link(
            source_id="n1",
            source_name="payload_#_anything_#_deeper",
            sink_id="n2",
            sink_name="input",
        )
        agent = _make_agent(nodes=nodes, links=[link])
        blocks = [
            _make_block(block_id="src-1", output_schema=output_schema),
            _make_block(
                block_id="sink-1",
                input_schema={"properties": {"input": {}}, "required": []},
            ),
        ]
        return agent, blocks

    def test_direct_additional_properties_accepts_chain(self):
        v = AgentValidator()
        agent, blocks = self._agent_and_blocks(
            {
                "properties": {
                    "payload": {"type": "object", "additionalProperties": True}
                }
            }
        )
        assert v.validate_source_output_existence(agent, blocks) is True

    def test_anyof_additional_properties_accepts_chain(self):
        v = AgentValidator()
        agent, blocks = self._agent_and_blocks(
            {
                "properties": {
                    "payload": {
                        "anyOf": [
                            {"type": "object", "additionalProperties": True},
                            {"type": "null"},
                        ]
                    }
                }
            }
        )
        assert v.validate_source_output_existence(agent, blocks) is True

    def test_anyof_items_additional_properties_accepts_chain(self):
        v = AgentValidator()
        agent, blocks = self._agent_and_blocks(
            {
                "properties": {
                    "payload": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": True,
                                },
                            },
                            {"type": "null"},
                        ]
                    }
                }
            }
        )
        assert v.validate_source_output_existence(agent, blocks) is True

    def test_missing_top_level_property_lists_available(self):
        v = AgentValidator()
        agent, blocks = self._agent_and_blocks(
            {"properties": {"other": {"type": "string"}}}
        )
        assert v.validate_source_output_existence(agent, blocks) is False
        assert any("Available properties" in e and "other" in e for e in v.errors)


class TestDeepDeclaredChains:
    """Arbitrary-depth strict validation through multiple *declared* object
    levels — each declared level is verified, and a miss names the level."""

    def _agent_and_blocks(self, source_name: str) -> tuple[dict, list[dict]]:
        nodes = [
            _make_node(node_id="n1", block_id="src-1"),
            _make_node(node_id="n2", block_id="sink-1"),
        ]
        link = _make_link(
            source_id="n1", source_name=source_name, sink_id="n2", sink_name="input"
        )
        agent = _make_agent(nodes=nodes, links=[link])
        blocks = [
            _make_block(
                block_id="src-1",
                output_schema={
                    "properties": {
                        "a": {
                            "type": "object",
                            "properties": {
                                "b": {
                                    "type": "object",
                                    "properties": {"c": {}},
                                }
                            },
                        }
                    }
                },
            ),
            _make_block(
                block_id="sink-1",
                input_schema={"properties": {"input": {}}, "required": []},
            ),
        ]
        return agent, blocks

    def test_three_declared_levels_pass(self):
        v = AgentValidator()
        agent, blocks = self._agent_and_blocks("a_#_b_#_c")
        assert v.validate_source_output_existence(agent, blocks) is True

    def test_miss_at_third_declared_level_names_it(self):
        v = AgentValidator()
        agent, blocks = self._agent_and_blocks("a_#_b_#_x")
        assert v.validate_source_output_existence(agent, blocks) is False
        assert any("'x'" in e and "'a_#_b'" in e for e in v.errors)

    def test_miss_at_second_declared_level_names_it(self):
        v = AgentValidator()
        agent, blocks = self._agent_and_blocks("a_#_x_#_c")
        assert v.validate_source_output_existence(agent, blocks) is False
        assert any("'x'" in e and "'a'" in e for e in v.errors)

    def test_untyped_leaf_after_declared_levels_accepts_further_descent(self):
        v = AgentValidator()
        agent, blocks = self._agent_and_blocks("a_#_b_#_c_#_anything")
        assert v.validate_source_output_existence(agent, blocks) is True
