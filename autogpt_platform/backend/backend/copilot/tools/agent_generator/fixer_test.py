"""Unit tests for AgentFixer."""

from .fixer import (
    _ADDTODICTIONARY_BLOCK_ID,
    _ADDTOLIST_BLOCK_ID,
    _CODE_EXECUTION_BLOCK_ID,
    _DATA_SAMPLING_BLOCK_ID,
    _GET_CURRENT_DATE_BLOCK_ID,
    _STORE_VALUE_BLOCK_ID,
    _TEXT_REPLACE_BLOCK_ID,
    AGENT_EXECUTOR_BLOCK_ID,
    AgentFixer,
)
from .helpers import generate_uuid


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
    """Build a minimal node dict for testing."""
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
    is_static: bool = False,
) -> dict:
    """Build a minimal link dict for testing."""
    return {
        "id": link_id or generate_uuid(),
        "source_id": source_id,
        "source_name": source_name,
        "sink_id": sink_id,
        "sink_name": sink_name,
        "is_static": is_static,
    }


class TestFixAgentIds:
    """Tests for fix_agent_ids."""

    def test_valid_uuids_unchanged(self):
        fixer = AgentFixer()
        agent_id = generate_uuid()
        link_id = generate_uuid()
        agent = _make_agent(agent_id=agent_id, links=[{"id": link_id}])

        result = fixer.fix_agent_ids(agent)

        assert result["id"] == agent_id
        assert result["links"][0]["id"] == link_id
        assert fixer.fixes_applied == []

    def test_invalid_agent_id_replaced(self):
        fixer = AgentFixer()
        agent = _make_agent(agent_id="bad-id")

        result = fixer.fix_agent_ids(agent)

        assert result["id"] != "bad-id"
        assert len(fixer.fixes_applied) == 1
        assert "agent ID" in fixer.fixes_applied[0]

    def test_invalid_link_id_replaced(self):
        fixer = AgentFixer()
        agent = _make_agent(links=[{"id": "not-a-uuid"}])

        result = fixer.fix_agent_ids(agent)

        assert result["links"][0]["id"] != "not-a-uuid"
        assert len(fixer.fixes_applied) == 1


class TestFixDoubleCurlyBraces:
    """Tests for fix_double_curly_braces."""

    def test_single_braces_converted_to_double(self):
        fixer = AgentFixer()
        node = _make_node(input_default={"prompt": "Hello {name}!"})
        agent = _make_agent(nodes=[node])

        result = fixer.fix_double_curly_braces(agent)

        assert result["nodes"][0]["input_default"]["prompt"] == "Hello {{name}}!"

    def test_double_braces_unchanged(self):
        fixer = AgentFixer()
        node = _make_node(input_default={"prompt": "Hello {{name}}!"})
        agent = _make_agent(nodes=[node])

        result = fixer.fix_double_curly_braces(agent)

        assert result["nodes"][0]["input_default"]["prompt"] == "Hello {{name}}!"
        assert fixer.fixes_applied == []

    def test_non_string_prompt_skipped(self):
        fixer = AgentFixer()
        node = _make_node(input_default={"prompt": 42})
        agent = _make_agent(nodes=[node])

        result = fixer.fix_double_curly_braces(agent)

        assert result["nodes"][0]["input_default"]["prompt"] == 42

    def test_non_string_prompt_with_prompt_values_skipped(self):
        """Ensure non-string prompt fields don't crash re.search in the
        prompt_values path."""
        fixer = AgentFixer()
        node_id = generate_uuid()
        source_id = generate_uuid()
        node = _make_node(
            node_id=node_id, input_default={"prompt": None, "prompt_values": {}}
        )
        source_node = _make_node(node_id=source_id)
        link = _make_link(
            source_id=source_id,
            source_name="output",
            sink_id=node_id,
            sink_name="prompt_values_$_name",
        )
        agent = _make_agent(nodes=[node, source_node], links=[link])

        result = fixer.fix_double_curly_braces(agent)

        # Should not crash and prompt stays None
        assert result["nodes"][0]["input_default"]["prompt"] is None


class TestFixCredentials:
    """Tests for fix_credentials."""

    def test_credentials_removed(self):
        fixer = AgentFixer()
        node = _make_node(
            input_default={
                "credentials": {"key": "secret"},
                "url": "http://example.com",
            }
        )
        agent = _make_agent(nodes=[node])

        result = fixer.fix_credentials(agent)

        assert "credentials" not in result["nodes"][0]["input_default"]
        assert result["nodes"][0]["input_default"]["url"] == "http://example.com"
        assert len(fixer.fixes_applied) == 1

    def test_no_credentials_unchanged(self):
        fixer = AgentFixer()
        node = _make_node(input_default={"url": "http://example.com"})
        agent = _make_agent(nodes=[node])

        result = fixer.fix_credentials(agent)

        assert result["nodes"][0]["input_default"]["url"] == "http://example.com"
        assert fixer.fixes_applied == []


class TestFixCodeExecutionOutput:
    """Tests for fix_code_execution_output."""

    def test_response_renamed_to_stdout_logs(self):
        fixer = AgentFixer()
        node = _make_node(node_id="n1", block_id=_CODE_EXECUTION_BLOCK_ID)
        link = _make_link(source_id="n1", source_name="response", sink_id="n2")
        agent = _make_agent(nodes=[node], links=[link])

        result = fixer.fix_code_execution_output(agent)

        assert result["links"][0]["source_name"] == "stdout_logs"
        assert len(fixer.fixes_applied) == 1

    def test_non_response_source_unchanged(self):
        fixer = AgentFixer()
        node = _make_node(node_id="n1", block_id=_CODE_EXECUTION_BLOCK_ID)
        link = _make_link(source_id="n1", source_name="stdout_logs", sink_id="n2")
        agent = _make_agent(nodes=[node], links=[link])

        result = fixer.fix_code_execution_output(agent)

        assert result["links"][0]["source_name"] == "stdout_logs"
        assert fixer.fixes_applied == []


class TestFixDataSamplingSampleSize:
    """Tests for fix_data_sampling_sample_size."""

    def test_sample_size_set_to_1(self):
        fixer = AgentFixer()
        node = _make_node(
            node_id="n1",
            block_id=_DATA_SAMPLING_BLOCK_ID,
            input_default={"sample_size": 10},
        )
        agent = _make_agent(nodes=[node])

        result = fixer.fix_data_sampling_sample_size(agent)

        assert result["nodes"][0]["input_default"]["sample_size"] == 1

    def test_removes_links_to_sample_size(self):
        fixer = AgentFixer()
        node = _make_node(node_id="n1", block_id=_DATA_SAMPLING_BLOCK_ID)
        link = _make_link(sink_id="n1", sink_name="sample_size", source_id="n2")
        agent = _make_agent(nodes=[node], links=[link])

        result = fixer.fix_data_sampling_sample_size(agent)

        assert len(result["links"]) == 0
        assert result["nodes"][0]["input_default"]["sample_size"] == 1


class TestFixTextReplaceNewParameter:
    """Tests for fix_text_replace_new_parameter."""

    def test_empty_new_changed_to_space(self):
        fixer = AgentFixer()
        node = _make_node(
            block_id=_TEXT_REPLACE_BLOCK_ID,
            input_default={"new": ""},
        )
        agent = _make_agent(nodes=[node])

        result = fixer.fix_text_replace_new_parameter(agent)

        assert result["nodes"][0]["input_default"]["new"] == " "

    def test_nonempty_new_unchanged(self):
        fixer = AgentFixer()
        node = _make_node(
            block_id=_TEXT_REPLACE_BLOCK_ID,
            input_default={"new": "replacement"},
        )
        agent = _make_agent(nodes=[node])

        result = fixer.fix_text_replace_new_parameter(agent)

        assert result["nodes"][0]["input_default"]["new"] == "replacement"
        assert fixer.fixes_applied == []


class TestFixGetCurrentDateOffset:
    """Tests for fix_getcurrentdate_offset."""

    def test_negative_offset_made_positive(self):
        fixer = AgentFixer()
        node = _make_node(
            block_id=_GET_CURRENT_DATE_BLOCK_ID,
            input_default={"offset": -5},
        )
        agent = _make_agent(nodes=[node])

        result = fixer.fix_getcurrentdate_offset(agent)

        assert result["nodes"][0]["input_default"]["offset"] == 5

    def test_positive_offset_unchanged(self):
        fixer = AgentFixer()
        node = _make_node(
            block_id=_GET_CURRENT_DATE_BLOCK_ID,
            input_default={"offset": 3},
        )
        agent = _make_agent(nodes=[node])

        result = fixer.fix_getcurrentdate_offset(agent)

        assert result["nodes"][0]["input_default"]["offset"] == 3
        assert fixer.fixes_applied == []


class TestFixNodeXCoordinates:
    """Tests for fix_node_x_coordinates."""

    def test_close_nodes_spread_apart(self):
        fixer = AgentFixer()
        src_node = _make_node(node_id="src", position=(0, 0))
        sink_node = _make_node(node_id="sink", position=(100, 0))
        link = _make_link(source_id="src", sink_id="sink")
        agent = _make_agent(nodes=[src_node, sink_node], links=[link])

        result = fixer.fix_node_x_coordinates(agent)

        sink = next(n for n in result["nodes"] if n["id"] == "sink")
        assert sink["metadata"]["position"]["x"] >= 800

    def test_far_apart_nodes_unchanged(self):
        fixer = AgentFixer()
        src_node = _make_node(node_id="src", position=(0, 0))
        sink_node = _make_node(node_id="sink", position=(1000, 0))
        link = _make_link(source_id="src", sink_id="sink")
        agent = _make_agent(nodes=[src_node, sink_node], links=[link])

        result = fixer.fix_node_x_coordinates(agent)

        sink = next(n for n in result["nodes"] if n["id"] == "sink")
        assert sink["metadata"]["position"]["x"] == 1000
        assert fixer.fixes_applied == []


class TestFixAddToDictionaryBlocks:
    """Tests for fix_addtodictionary_blocks."""

    def test_removes_create_dictionary_nodes(self):
        fixer = AgentFixer()
        create_dict_id = "b924ddf4-de4f-4b56-9a85-358930dcbc91"
        dict_node = _make_node(node_id="dict-1", block_id=create_dict_id)
        add_to_dict_node = _make_node(
            node_id="add-1", block_id=_ADDTODICTIONARY_BLOCK_ID
        )
        link = _make_link(source_id="dict-1", sink_id="add-1")
        agent = _make_agent(nodes=[dict_node, add_to_dict_node], links=[link])

        result = fixer.fix_addtodictionary_blocks(agent)

        node_ids = [n["id"] for n in result["nodes"]]
        assert "dict-1" not in node_ids
        assert "add-1" in node_ids
        assert len(result["links"]) == 0


class TestFixStoreValueBeforeCondition:
    """Tests for fix_storevalue_before_condition."""

    def test_inserts_storevalue_block(self):
        fixer = AgentFixer()
        condition_block_id = "715696a0-e1da-45c8-b209-c2fa9c3b0be6"
        src_node = _make_node(node_id="src")
        cond_node = _make_node(node_id="cond", block_id=condition_block_id)
        link = _make_link(
            source_id="src", source_name="output", sink_id="cond", sink_name="value2"
        )
        agent = _make_agent(nodes=[src_node, cond_node], links=[link])

        result = fixer.fix_storevalue_before_condition(agent)

        # Should have 3 nodes now (original 2 + new StoreValueBlock)
        assert len(result["nodes"]) == 3
        store_nodes = [
            n for n in result["nodes"] if n["block_id"] == _STORE_VALUE_BLOCK_ID
        ]
        assert len(store_nodes) == 1
        assert store_nodes[0]["input_default"]["data"] is None


class TestFixAddToListBlocks:
    """Tests for fix_addtolist_blocks - self-reference links."""

    def test_addtolist_gets_self_reference_link(self):
        fixer = AgentFixer()
        node = _make_node(node_id="atl-1", block_id=_ADDTOLIST_BLOCK_ID)
        # Source link to AddToList (from some other node)
        link = _make_link(
            source_id="other",
            source_name="output",
            sink_id="atl-1",
            sink_name="item",
        )
        other_node = _make_node(node_id="other")
        agent = _make_agent(nodes=[other_node, node], links=[link])

        result = fixer.fix_addtolist_blocks(agent)

        # Should have a self-reference link: atl-1.updated_list -> atl-1.list
        self_ref_links = [
            lnk
            for lnk in result["links"]
            if lnk["source_id"] == "atl-1"
            and lnk["sink_id"] == "atl-1"
            and lnk["source_name"] == "updated_list"
            and lnk["sink_name"] == "list"
        ]
        assert len(self_ref_links) == 1


class TestFixLinkStaticProperties:
    """Tests for fix_link_static_properties."""

    def test_sets_is_static_from_block_schema(self):
        fixer = AgentFixer()
        block_id = generate_uuid()
        node = _make_node(node_id="n1", block_id=block_id)
        link = _make_link(source_id="n1", sink_id="n2", is_static=False)
        agent = _make_agent(nodes=[node], links=[link])

        blocks = [{"id": block_id, "staticOutput": True}]

        result = fixer.fix_link_static_properties(agent, blocks)

        assert result["links"][0]["is_static"] is True

    def test_unknown_block_leaves_link_unchanged(self):
        fixer = AgentFixer()
        node = _make_node(node_id="n1", block_id="unknown-block")
        link = _make_link(source_id="n1", sink_id="n2", is_static=True)
        agent = _make_agent(nodes=[node], links=[link])

        result = fixer.fix_link_static_properties(agent, blocks=[])

        # Unknown block → skipped, link stays as-is
        assert result["links"][0]["is_static"] is True


class TestFixAiModelParameter:
    """Tests for fix_ai_model_parameter."""

    def test_missing_model_gets_default(self):
        fixer = AgentFixer()
        block_id = generate_uuid()
        node = _make_node(node_id="n1", block_id=block_id, input_default={})
        agent = _make_agent(nodes=[node])

        blocks = [
            {
                "id": block_id,
                "categories": [{"category": "AI"}],
                "inputSchema": {
                    "properties": {"model": {"type": "string"}},
                },
            }
        ]

        result = fixer.fix_ai_model_parameter(agent, blocks)

        assert result["nodes"][0]["input_default"]["model"] == "gpt-4o"

    def test_valid_model_unchanged(self):
        fixer = AgentFixer()
        block_id = generate_uuid()
        node = _make_node(
            node_id="n1",
            block_id=block_id,
            input_default={"model": "claude-opus-4-6"},
        )
        agent = _make_agent(nodes=[node])

        blocks = [
            {
                "id": block_id,
                "categories": [{"category": "AI"}],
                "inputSchema": {
                    "properties": {"model": {"type": "string"}},
                },
            }
        ]

        result = fixer.fix_ai_model_parameter(agent, blocks)

        assert result["nodes"][0]["input_default"]["model"] == "claude-opus-4-6"


class TestFixAgentExecutorBlocks:
    """Tests for fix_agent_executor_blocks."""

    def test_fills_schemas_from_library_agent(self):
        fixer = AgentFixer()
        lib_agent_id = generate_uuid()
        node = _make_node(
            node_id="n1",
            block_id=AGENT_EXECUTOR_BLOCK_ID,
            input_default={
                "graph_id": lib_agent_id,
                "graph_version": 1,
                "user_id": "user-1",
            },
        )
        agent = _make_agent(nodes=[node])

        # Library agents use graph_id as the lookup key
        library_agents = [
            {
                "graph_id": lib_agent_id,
                "graph_version": 2,
                "input_schema": {"field1": {"type": "string"}},
                "output_schema": {"result": {"type": "string"}},
            }
        ]

        result = fixer.fix_agent_executor_blocks(agent, library_agents)

        node_result = result["nodes"][0]["input_default"]
        assert node_result["graph_version"] == 2
        assert node_result["input_schema"] == {"field1": {"type": "string"}}
        assert node_result["output_schema"] == {"result": {"type": "string"}}


class TestFixInvalidNestedSinkLinks:
    """Tests for fix_invalid_nested_sink_links."""

    def test_removes_numeric_index_links(self):
        fixer = AgentFixer()
        block_id = generate_uuid()
        node = _make_node(node_id="n1", block_id=block_id)
        link = _make_link(source_id="n2", sink_id="n1", sink_name="values_#_0")
        agent = _make_agent(nodes=[node], links=[link])

        blocks = [
            {
                "id": block_id,
                "inputSchema": {"properties": {"values": {"type": "array"}}},
            }
        ]

        result = fixer.fix_invalid_nested_sink_links(agent, blocks)

        assert len(result["links"]) == 0

    def test_valid_nested_links_kept(self):
        fixer = AgentFixer()
        block_id = generate_uuid()
        node = _make_node(node_id="n1", block_id=block_id)
        link = _make_link(source_id="n2", sink_id="n1", sink_name="values_#_name")
        agent = _make_agent(nodes=[node], links=[link])

        blocks = [
            {
                "id": block_id,
                "inputSchema": {
                    "properties": {"values": {"type": "object"}},
                },
            }
        ]

        result = fixer.fix_invalid_nested_sink_links(agent, blocks)

        assert len(result["links"]) == 1


class TestApplyAllFixes:
    """Tests for apply_all_fixes orchestration."""

    def test_is_sync(self):
        """apply_all_fixes should be a sync function."""
        import inspect

        assert not inspect.iscoroutinefunction(AgentFixer.apply_all_fixes)

    def test_applies_multiple_fixes(self):
        fixer = AgentFixer()
        agent = _make_agent(
            agent_id="bad-id",
            nodes=[
                _make_node(
                    block_id=_TEXT_REPLACE_BLOCK_ID,
                    input_default={"new": "", "credentials": {"key": "secret"}},
                )
            ],
        )

        result = fixer.apply_all_fixes(agent)

        # Agent ID should be fixed
        assert result["id"] != "bad-id"
        # Credentials should be removed
        assert "credentials" not in result["nodes"][0]["input_default"]
        # Text replace "new" should be space
        assert result["nodes"][0]["input_default"]["new"] == " "
        # Multiple fixes applied
        assert len(fixer.fixes_applied) >= 3

    def test_empty_agent_no_crash(self):
        fixer = AgentFixer()
        agent = _make_agent()

        result = fixer.apply_all_fixes(agent)

        assert "nodes" in result
        assert "links" in result

    def test_returns_deep_copy_behavior(self):
        """Fixer mutates in place — verify the same dict is returned."""
        fixer = AgentFixer()
        agent = _make_agent()
        result = fixer.apply_all_fixes(agent)
        assert result is agent
