"""Unit tests for AgentFixer."""

import time

import pytest

from .fixer import (
    _ADDTODICTIONARY_BLOCK_ID,
    _ADDTOLIST_BLOCK_ID,
    _CODE_EXECUTION_BLOCK_ID,
    _CREATE_DICT_BLOCK_ID,
    _CREATE_LIST_BLOCK_ID,
    _DATA_SAMPLING_BLOCK_ID,
    _FIX_VALUE2_EMPTY_STRING_BLOCK_IDS,
    _GET_CURRENT_DATE_BLOCK_ID,
    _GMAIL_SEND_BLOCK_ID,
    _STORE_VALUE_BLOCK_ID,
    _TEXT_REPLACE_BLOCK_ID,
    _UNIVERSAL_TYPE_CONVERTER_BLOCK_ID,
    AGENT_EXECUTOR_BLOCK_ID,
    MCP_TOOL_BLOCK_ID,
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

    def test_prompt_feeding_a_code_execution_block_is_left_alone(self):
        fixer = AgentFixer()
        to_code = _make_node(node_id="to-code", input_default={"prompt": "{a}"})
        to_text = _make_node(node_id="to-text", input_default={"prompt": "{b}"})
        code = _make_node(node_id="code", block_id=_CODE_EXECUTION_BLOCK_ID)
        text = _make_node(node_id="text")
        agent = _make_agent(
            nodes=[to_code, to_text, code, text],
            links=[
                _make_link(source_id="to-code", sink_id="text"),
                _make_link(source_id="to-code", sink_id="code"),
                _make_link(source_id="to-text", sink_id="text"),
            ],
        )

        result = fixer.fix_double_curly_braces(agent)

        assert result["nodes"][0]["input_default"]["prompt"] == "{a}"
        assert result["nodes"][1]["input_default"]["prompt"] == "{{b}}"

    def test_linked_prompt_values_missing_from_the_prompt_are_added(self):
        fixer = AgentFixer()
        node = _make_node(node_id="n1", input_default={"prompt": "Hi {{name}}"})
        agent = _make_agent(
            nodes=[node],
            links=[
                _make_link(sink_id="n1", sink_name="prompt_values_#_name"),
                _make_link(sink_id="n1", sink_name="prompt_values_#_topic"),
            ],
        )

        result = fixer.fix_double_curly_braces(agent)

        assert result["nodes"][0]["input_default"]["prompt"] == "Hi {{name}}\n{{topic}}"


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

    def test_keeps_links_to_other_fields_and_other_nodes(self):
        fixer = AgentFixer()
        node = _make_node(node_id="n1", block_id=_DATA_SAMPLING_BLOCK_ID)
        other = _make_node(node_id="n2")
        kept_links = [
            _make_link(sink_id="n1", sink_name="data", source_id="n2"),
            _make_link(sink_id="n2", sink_name="sample_size", source_id="n1"),
        ]
        agent = _make_agent(
            nodes=[node, other],
            links=[
                kept_links[0],
                _make_link(sink_id="n1", sink_name="sample_size", source_id="n2"),
                kept_links[1],
            ],
        )

        result = fixer.fix_data_sampling_sample_size(agent)

        assert result["links"] == kept_links


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

    def test_keeps_the_other_links(self):
        fixer = AgentFixer()
        dict_node = _make_node(node_id="dict-1", block_id=_CREATE_DICT_BLOCK_ID)
        add_to_dict_node = _make_node(
            node_id="add-1", block_id=_ADDTODICTIONARY_BLOCK_ID
        )
        other = _make_node(node_id="other")
        kept_link = _make_link(source_id="other", sink_id="add-1", sink_name="value")
        agent = _make_agent(
            nodes=[dict_node, add_to_dict_node, other],
            links=[_make_link(source_id="dict-1", sink_id="add-1"), kept_link],
        )

        result = fixer.fix_addtodictionary_blocks(agent)

        assert result["links"] == [kept_link]


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

    def test_link_from_a_storevalue_output_is_kept(self):
        fixer = AgentFixer()
        store_node = _make_node(node_id="store", block_id=_STORE_VALUE_BLOCK_ID)
        cond_node = _make_node(
            node_id="cond", block_id=_FIX_VALUE2_EMPTY_STRING_BLOCK_IDS[0]
        )
        link = _make_link(
            source_id="store", source_name="output", sink_id="cond", sink_name="value2"
        )
        agent = _make_agent(nodes=[store_node, cond_node], links=[link])

        result = fixer.fix_storevalue_before_condition(agent)

        assert result["nodes"] == [store_node, cond_node]
        assert result["links"] == [link]
        assert fixer.fixes_applied == []

    def test_condition_with_a_storevalue_gets_no_second_one(self):
        fixer = AgentFixer()
        src_node = _make_node(node_id="src")
        store_node = _make_node(node_id="store", block_id=_STORE_VALUE_BLOCK_ID)
        cond_node = _make_node(
            node_id="cond", block_id=_FIX_VALUE2_EMPTY_STRING_BLOCK_IDS[0]
        )
        from_src = _make_link(source_id="src", sink_id="cond", sink_name="value2")
        from_store = _make_link(source_id="store", sink_id="cond", sink_name="value2")
        agent = _make_agent(
            nodes=[src_node, store_node, cond_node], links=[from_src, from_store]
        )

        result = fixer.fix_storevalue_before_condition(agent)

        assert len(result["nodes"]) == 3
        assert result["links"] == [from_src, from_store]
        assert len(fixer.fixes_applied) == 1
        assert "cond - already has one connected" in fixer.fixes_applied[0]


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

    def test_links_into_list_are_replaced_by_a_prerequisite_block(self):
        fixer = AgentFixer()
        create_list = _make_node(node_id="create", block_id=_CREATE_LIST_BLOCK_ID)
        other = _make_node(node_id="other")
        add_to_list = _make_node(node_id="atl-1", block_id=_ADDTOLIST_BLOCK_ID)
        agent = _make_agent(
            nodes=[create_list, other, add_to_list],
            links=[
                _make_link(source_id="create", sink_id="atl-1", sink_name="list"),
                _make_link(source_id="other", sink_id="atl-1", sink_name="list"),
                _make_link(source_id="other", sink_id="atl-1", sink_name="entry"),
            ],
        )

        result = fixer.fix_addtolist_blocks(agent)

        # The CreateListBlock is gone, and an empty AddToList block stands
        # before the original one.
        assert [n["id"] for n in result["nodes"]][::2] == ["other", "atl-1"]
        prerequisite = result["nodes"][1]
        assert prerequisite["block_id"] == _ADDTOLIST_BLOCK_ID
        assert prerequisite["input_default"]["list"] == []
        assert sorted(
            (lnk["source_id"], lnk["source_name"], lnk["sink_id"], lnk["sink_name"])
            for lnk in result["links"]
        ) == sorted(
            [
                (prerequisite["id"], "updated_list", "atl-1", "list"),
                ("other", "output", "atl-1", "entry"),
                ("atl-1", "updated_list", "atl-1", "list"),
            ]
        )

    def test_storevalue_fed_by_a_createlist_loses_only_its_link(self):
        fixer = AgentFixer()
        create_list = _make_node(node_id="create", block_id=_CREATE_LIST_BLOCK_ID)
        fed_store = _make_node(node_id="fed-store", block_id=_STORE_VALUE_BLOCK_ID)
        plain_store = _make_node(node_id="plain-store", block_id=_STORE_VALUE_BLOCK_ID)
        add_to_list = _make_node(node_id="atl-1", block_id=_ADDTOLIST_BLOCK_ID)
        agent = _make_agent(
            nodes=[create_list, fed_store, plain_store, add_to_list],
            links=[
                _make_link(source_id="create", sink_id="fed-store"),
                _make_link(source_id="fed-store", sink_id="atl-1", sink_name="entry"),
                _make_link(source_id="plain-store", sink_id="atl-1", sink_name="entry"),
            ],
        )

        result = fixer.fix_addtolist_blocks(agent)

        node_ids = [n["id"] for n in result["nodes"]]
        assert {"create", "fed-store", "plain-store", "atl-1"} <= set(node_ids)
        link_ends = [(lnk["source_id"], lnk["sink_id"]) for lnk in result["links"]]
        assert ("create", "fed-store") in link_ends
        assert ("plain-store", "atl-1") in link_ends
        assert ("fed-store", "atl-1") not in link_ends

    def test_existing_prerequisite_block_is_not_added_again(self):
        fixer = AgentFixer()
        prerequisite = _make_node(
            node_id="pre",
            block_id=_ADDTOLIST_BLOCK_ID,
            input_default={"list": [], "entry": None, "entries": []},
        )
        add_to_list = _make_node(node_id="atl-1", block_id=_ADDTOLIST_BLOCK_ID)
        links = [
            _make_link(
                source_id="pre",
                source_name="updated_list",
                sink_id="atl-1",
                sink_name="list",
            ),
            _make_link(
                source_id="atl-1",
                source_name="updated_list",
                sink_id="atl-1",
                sink_name="list",
            ),
        ]
        agent = _make_agent(nodes=[prerequisite, add_to_list], links=list(links))

        result = fixer.fix_addtolist_blocks(agent)

        assert result["nodes"] == [prerequisite, add_to_list]
        assert result["links"] == links


class TestFixAddToListGmailSelfReference:
    """Tests for fix_addtolist_gmail_self_reference."""

    def test_self_reference_removed_from_addtolist_linked_to_gmail(self):
        fixer = AgentFixer()
        to_gmail = _make_node(node_id="to-gmail", block_id=_ADDTOLIST_BLOCK_ID)
        elsewhere = _make_node(node_id="elsewhere", block_id=_ADDTOLIST_BLOCK_ID)
        gmail = _make_node(node_id="gmail", block_id=_GMAIL_SEND_BLOCK_ID)
        other = _make_node(node_id="other")
        self_reference = {
            "source_name": "updated_list",
            "sink_name": "list",
        }
        kept_links = [
            _make_link(source_id="to-gmail", sink_id="gmail"),
            _make_link(source_id="elsewhere", sink_id="other"),
            _make_link(source_id="elsewhere", sink_id="elsewhere", **self_reference),
        ]
        agent = _make_agent(
            nodes=[to_gmail, elsewhere, gmail, other],
            links=[
                _make_link(source_id="to-gmail", sink_id="to-gmail", **self_reference),
                *kept_links,
            ],
        )

        result = fixer.fix_addtolist_gmail_self_reference(agent)

        assert result["links"] == kept_links


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

    def test_new_generic_model_survives_string_model_field(self):
        """claude-opus-5-5 was added to generic_allowed_models; a no-enum AI
        block (plain string model field) must preserve it instead of
        falling back to the generic default."""
        fixer = AgentFixer()
        block_id = generate_uuid()
        node = _make_node(
            node_id="n1",
            block_id=block_id,
            input_default={"model": "claude-opus-5-5"},
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

        assert result["nodes"][0]["input_default"]["model"] == "claude-opus-5-5"

    def test_block_specific_enum_uses_block_default(self):
        """Blocks with their own model enum (e.g. PerplexityBlock) should use
        the block's allowed models and default, not the generic ones."""
        fixer = AgentFixer()
        block_id = generate_uuid()
        node = _make_node(
            node_id="n1",
            block_id=block_id,
            input_default={"model": "gpt-5.2-2025-12-11"},
        )
        agent = _make_agent(nodes=[node])

        blocks = [
            {
                "id": block_id,
                "name": "PerplexityBlock",
                "categories": [{"category": "AI"}],
                "inputSchema": {
                    "properties": {
                        "model": {
                            "type": "string",
                            "enum": [
                                "perplexity/sonar",
                                "perplexity/sonar-pro",
                                "perplexity/sonar-deep-research",
                            ],
                            "default": "perplexity/sonar",
                        }
                    },
                },
            }
        ]

        result = fixer.fix_ai_model_parameter(agent, blocks)

        assert result["nodes"][0]["input_default"]["model"] == "perplexity/sonar"

    def test_block_specific_enum_valid_model_unchanged(self):
        """A valid block-specific model should not be replaced."""
        fixer = AgentFixer()
        block_id = generate_uuid()
        node = _make_node(
            node_id="n1",
            block_id=block_id,
            input_default={"model": "perplexity/sonar-pro"},
        )
        agent = _make_agent(nodes=[node])

        blocks = [
            {
                "id": block_id,
                "name": "PerplexityBlock",
                "categories": [{"category": "AI"}],
                "inputSchema": {
                    "properties": {
                        "model": {
                            "type": "string",
                            "enum": [
                                "perplexity/sonar",
                                "perplexity/sonar-pro",
                                "perplexity/sonar-deep-research",
                            ],
                            "default": "perplexity/sonar",
                        }
                    },
                },
            }
        ]

        result = fixer.fix_ai_model_parameter(agent, blocks)

        assert result["nodes"][0]["input_default"]["model"] == "perplexity/sonar-pro"

    def test_block_specific_enum_missing_model_gets_block_default(self):
        """Missing model on a block with enum should use the block's default."""
        fixer = AgentFixer()
        block_id = generate_uuid()
        node = _make_node(node_id="n1", block_id=block_id, input_default={})
        agent = _make_agent(nodes=[node])

        blocks = [
            {
                "id": block_id,
                "name": "PerplexityBlock",
                "categories": [{"category": "AI"}],
                "inputSchema": {
                    "properties": {
                        "model": {
                            "type": "string",
                            "enum": [
                                "perplexity/sonar",
                                "perplexity/sonar-pro",
                                "perplexity/sonar-deep-research",
                            ],
                            "default": "perplexity/sonar",
                        }
                    },
                },
            }
        ]

        result = fixer.fix_ai_model_parameter(agent, blocks)

        assert result["nodes"][0]["input_default"]["model"] == "perplexity/sonar"

    def test_ai_block_without_model_property_is_skipped(self):
        """AI-category blocks that have no 'model' input property should not
        have a model injected — they simply don't expose a model selector."""
        fixer = AgentFixer()
        block_id = generate_uuid()
        node = _make_node(node_id="n1", block_id=block_id, input_default={})
        agent = _make_agent(nodes=[node])

        blocks = [
            {
                "id": block_id,
                "name": "SomeAIBlock",
                "categories": [{"category": "AI"}],
                "inputSchema": {
                    "properties": {"prompt": {"type": "string"}},
                },
            }
        ]

        result = fixer.fix_ai_model_parameter(agent, blocks)

        assert "model" not in result["nodes"][0]["input_default"]


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

    def test_fixes_logged_are_bounded(self):
        """Every fix is applied, but only so many are listed, each cut short."""
        fixer = AgentFixer()
        node = _make_node(
            node_id="n" * 10_000,
            block_id=MCP_TOOL_BLOCK_ID,
            input_default={
                "tool_input_schema": {
                    "properties": {str(i): {"default": i} for i in range(5_000)}
                }
            },
        )
        agent = _make_agent(nodes=[node], links=[{} for _ in range(5_000)])

        result = fixer.apply_all_fixes(agent)

        assert all(link["id"] for link in result["links"])
        assert len(result["nodes"][0]["input_default"]["tool_arguments"]) == 5_000
        assert len(fixer.fixes_applied) == AgentFixer.MAX_FIXES_LOGGED + 1
        assert "more fix(es) not listed" in fixer.fixes_applied[-1]
        assert all(
            len(fix) <= AgentFixer.MAX_FIX_LOG_LENGTH + 3 for fix in fixer.fixes_applied
        )

    def test_omitted_fix_count_is_reset_between_runs(self):
        fixer = AgentFixer()
        fixer.apply_all_fixes(_make_agent(links=[{} for _ in range(5_000)]))

        fixer.apply_all_fixes(_make_agent(links=[{}]))

        assert not any("not listed" in fix for fix in fixer.fixes_applied)


def _links_naming_no_node(count: int) -> list[dict]:
    """Links no node lookup can stop early for."""
    return [{"source_id": "missing", "sink_id": "missing"} for _ in range(count)]


_SUB_AGENT = {
    "graph_id": "sub-agent",
    "graph_version": 1,
    "name": "Sub-agent",
    "input_schema": {"properties": {"query": {}}, "required": ["query"]},
    "output_schema": {"properties": {}},
}

# One graph per fix, sized so that a scan of a list per node or link would
# take far longer than the bound.
_LARGE_GRAPHS = [
    pytest.param(
        AgentFixer.apply_all_fixes,
        lambda: ([{} for _ in range(40_000)], _links_naming_no_node(8_000)),
        id="apply_all_fixes",
    ),
    pytest.param(
        AgentFixer.fix_addtolist_gmail_self_reference,
        lambda: ([{} for _ in range(20_000)], _links_naming_no_node(20_000)),
        id="fix_addtolist_gmail_self_reference",
    ),
    pytest.param(
        AgentFixer.fix_double_curly_braces,
        lambda: (
            [
                {"id": f"n{i}", "input_default": {"prompt": "{a}"}}
                for i in range(40_000)
            ],
            _links_naming_no_node(40_000),
        ),
        id="fix_double_curly_braces",
    ),
    pytest.param(
        AgentFixer.fix_double_curly_braces,
        lambda: (
            [{"id": "n", "input_default": {"prompt": "x" * 1_000_000}}],
            [
                {"sink_id": "n", "sink_name": f"prompt_values_#_v{i}"}
                for i in range(40_000)
            ],
        ),
        id="fix_double_curly_braces-prompt_values",
    ),
    pytest.param(
        AgentFixer.fix_storevalue_before_condition,
        lambda: (
            [{"id": "cond", "block_id": _FIX_VALUE2_EMPTY_STRING_BLOCK_IDS[0]}],
            [
                {"source_id": "missing", "sink_id": "cond", "sink_name": "value2"}
                for _ in range(12_000)
            ],
        ),
        id="fix_storevalue_before_condition",
    ),
    pytest.param(
        AgentFixer.fix_addtolist_blocks,
        lambda: (
            [{"id": f"n{i}", "block_id": _ADDTOLIST_BLOCK_ID} for i in range(16_000)],
            _links_naming_no_node(16_000),
        ),
        id="fix_addtolist_blocks",
    ),
    pytest.param(
        AgentFixer.fix_addtolist_blocks,
        lambda: (
            [{"id": "same", "block_id": _ADDTOLIST_BLOCK_ID} for _ in range(1_500)],
            [
                {"id": str(i), "sink_id": "same", "sink_name": "list"}
                for i in range(1_500)
            ],
        ),
        id="fix_addtolist_blocks-nodes-sharing-an-id",
    ),
    pytest.param(
        AgentFixer.fix_addtodictionary_blocks,
        lambda: (
            [
                {"id": "create", "block_id": _CREATE_DICT_BLOCK_ID},
                {"id": "add", "block_id": _ADDTODICTIONARY_BLOCK_ID},
            ],
            [
                {"id": str(i), "source_id": "create", "sink_id": "add"}
                for i in range(60_000)
            ],
        ),
        id="fix_addtodictionary_blocks",
    ),
    pytest.param(
        AgentFixer.fix_data_sampling_sample_size,
        lambda: (
            [
                {"id": f"n{i}", "block_id": _DATA_SAMPLING_BLOCK_ID}
                for i in range(40_000)
            ],
            _links_naming_no_node(40_000),
        ),
        id="fix_data_sampling_sample_size",
    ),
    pytest.param(
        lambda fixer, agent: fixer.fix_agent_executor_blocks(agent, [_SUB_AGENT]),
        lambda: (
            [
                {
                    "id": f"n{i}",
                    "block_id": AGENT_EXECUTOR_BLOCK_ID,
                    "input_default": {"graph_id": "sub-agent"},
                }
                for i in range(40_000)
            ],
            _links_naming_no_node(40_000),
        ),
        id="fix_agent_executor_blocks",
    ),
]


class TestLargeGraphs:
    """The fixer's work grows with the graph, not with nodes x links."""

    @pytest.mark.parametrize("fix, make_nodes_and_links", _LARGE_GRAPHS)
    def test_fix_scans_no_list_per_node_or_link(self, fix, make_nodes_and_links):
        nodes, links = make_nodes_and_links()
        agent = _make_agent(nodes=nodes, links=links)

        started = time.thread_time()
        fix(AgentFixer(), agent)

        # One pass over the nodes and the links takes well under a second of
        # CPU time.
        assert time.thread_time() - started < 10.0

    def test_nodes_sharing_an_id_do_not_multiply_the_fixes(self):
        """Links into an id several nodes share are handled once."""
        fixer = AgentFixer()
        samplers = [
            {"id": "sampler", "block_id": _DATA_SAMPLING_BLOCK_ID} for _ in range(1_000)
        ]
        prompts = [
            {"id": "prompt", "input_default": {"prompt": ""}} for _ in range(1_000)
        ]
        links = [
            {"source_id": "missing", "sink_id": "sampler", "sink_name": "sample_size"}
            for _ in range(1_000)
        ] + [
            {
                "source_id": "missing",
                "sink_id": "prompt",
                "sink_name": f"prompt_values_#_v{i}",
            }
            for i in range(1_000)
        ]
        agent = _make_agent(nodes=samplers + prompts, links=links)

        result = fixer.apply_all_fixes(agent)

        extended_prompts = [
            node
            for node in result["nodes"]
            if node["id"] == "prompt" and node["input_default"]["prompt"]
        ]
        assert len(extended_prompts) == 1


class TestFixMCPToolBlocks:
    """Tests for fix_mcp_tool_blocks."""

    def test_adds_missing_tool_arguments(self):
        fixer = AgentFixer()
        node = _make_node(
            node_id="n1",
            block_id=MCP_TOOL_BLOCK_ID,
            input_default={
                "server_url": "https://mcp.example.com/sse",
                "selected_tool": "search",
                "tool_input_schema": {},
            },
        )
        agent = _make_agent(nodes=[node])

        result = fixer.fix_mcp_tool_blocks(agent)

        assert result["nodes"][0]["input_default"]["tool_arguments"] == {}
        assert any("tool_arguments" in f for f in fixer.fixes_applied)

    def test_adds_missing_tool_input_schema(self):
        fixer = AgentFixer()
        node = _make_node(
            node_id="n1",
            block_id=MCP_TOOL_BLOCK_ID,
            input_default={
                "server_url": "https://mcp.example.com/sse",
                "selected_tool": "search",
                "tool_arguments": {},
            },
        )
        agent = _make_agent(nodes=[node])

        result = fixer.fix_mcp_tool_blocks(agent)

        assert result["nodes"][0]["input_default"]["tool_input_schema"] == {}
        assert any("tool_input_schema" in f for f in fixer.fixes_applied)

    def test_populates_tool_arguments_from_schema(self):
        fixer = AgentFixer()
        node = _make_node(
            node_id="n1",
            block_id=MCP_TOOL_BLOCK_ID,
            input_default={
                "server_url": "https://mcp.example.com/sse",
                "selected_tool": "search",
                "tool_input_schema": {
                    "properties": {
                        "query": {"type": "string", "default": "hello"},
                        "limit": {"type": "integer"},
                    }
                },
                "tool_arguments": {},
            },
        )
        agent = _make_agent(nodes=[node])

        result = fixer.fix_mcp_tool_blocks(agent)

        tool_args = result["nodes"][0]["input_default"]["tool_arguments"]
        assert tool_args["query"] == "hello"
        assert tool_args["limit"] is None

    def test_no_op_when_already_complete(self):
        fixer = AgentFixer()
        node = _make_node(
            node_id="n1",
            block_id=MCP_TOOL_BLOCK_ID,
            input_default={
                "server_url": "https://mcp.example.com/sse",
                "selected_tool": "search",
                "tool_input_schema": {},
                "tool_arguments": {},
            },
        )
        agent = _make_agent(nodes=[node])

        fixer.fix_mcp_tool_blocks(agent)

        assert len(fixer.fixes_applied) == 0


class TestFixDynamicBlockSinkNames:
    """Tests for fix_dynamic_block_sink_names."""

    def test_mcp_tool_arguments_prefix_removed(self):
        fixer = AgentFixer()
        node = _make_node(node_id="n1", block_id=MCP_TOOL_BLOCK_ID)
        link = _make_link(
            source_id="src", sink_id="n1", sink_name="tool_arguments_#_query"
        )
        agent = _make_agent(nodes=[node], links=[link])

        fixer.fix_dynamic_block_sink_names(agent)

        assert agent["links"][0]["sink_name"] == "query"
        assert len(fixer.fixes_applied) == 1

    def test_agent_executor_inputs_prefix_removed(self):
        fixer = AgentFixer()
        node = _make_node(node_id="n1", block_id=AGENT_EXECUTOR_BLOCK_ID)
        link = _make_link(source_id="src", sink_id="n1", sink_name="inputs_#_url")
        agent = _make_agent(nodes=[node], links=[link])

        fixer.fix_dynamic_block_sink_names(agent)

        assert agent["links"][0]["sink_name"] == "url"
        assert len(fixer.fixes_applied) == 1

    def test_bare_sink_name_unchanged(self):
        fixer = AgentFixer()
        node = _make_node(node_id="n1", block_id=MCP_TOOL_BLOCK_ID)
        link = _make_link(source_id="src", sink_id="n1", sink_name="query")
        agent = _make_agent(nodes=[node], links=[link])

        fixer.fix_dynamic_block_sink_names(agent)

        assert agent["links"][0]["sink_name"] == "query"
        assert len(fixer.fixes_applied) == 0

    def test_non_dynamic_block_unchanged(self):
        fixer = AgentFixer()
        node = _make_node(node_id="n1", block_id="some-other-block-id")
        link = _make_link(source_id="src", sink_id="n1", sink_name="values_#_key")
        agent = _make_agent(nodes=[node], links=[link])

        fixer.fix_dynamic_block_sink_names(agent)

        assert agent["links"][0]["sink_name"] == "values_#_key"
        assert len(fixer.fixes_applied) == 0


class TestFixDataTypeMismatch:
    """Tests for fix_data_type_mismatch."""

    @staticmethod
    def _make_block(
        block_id: str,
        name: str = "TestBlock",
        input_schema: dict | None = None,
        output_schema: dict | None = None,
    ) -> dict:
        return {
            "id": block_id,
            "name": name,
            "inputSchema": input_schema or {"properties": {}},
            "outputSchema": output_schema or {"properties": {}},
        }

    def test_inserts_converter_for_incompatible_types(self):
        fixer = AgentFixer()
        src_block_id = generate_uuid()
        sink_block_id = generate_uuid()

        src_node = _make_node(node_id="src", block_id=src_block_id)
        sink_node = _make_node(node_id="sink", block_id=sink_block_id)
        link = _make_link(
            source_id="src",
            source_name="result",
            sink_id="sink",
            sink_name="count",
        )
        agent = _make_agent(nodes=[src_node, sink_node], links=[link])

        blocks = [
            self._make_block(
                src_block_id,
                name="Source",
                output_schema={"properties": {"result": {"type": "string"}}},
            ),
            self._make_block(
                sink_block_id,
                name="Sink",
                input_schema={"properties": {"count": {"type": "integer"}}},
            ),
        ]

        result = fixer.fix_data_type_mismatch(agent, blocks)

        # A converter node should have been inserted
        converter_nodes = [
            n
            for n in result["nodes"]
            if n["block_id"] == _UNIVERSAL_TYPE_CONVERTER_BLOCK_ID
        ]
        assert len(converter_nodes) == 1
        assert converter_nodes[0]["input_default"]["type"] == "number"

        # Original link replaced by two new links through the converter
        assert len(result["links"]) == 2
        src_to_converter = result["links"][0]
        converter_to_sink = result["links"][1]

        assert src_to_converter["source_id"] == "src"
        assert src_to_converter["sink_id"] == converter_nodes[0]["id"]
        assert src_to_converter["sink_name"] == "value"

        assert converter_to_sink["source_id"] == converter_nodes[0]["id"]
        assert converter_to_sink["source_name"] == "value"
        assert converter_to_sink["sink_id"] == "sink"
        assert converter_to_sink["sink_name"] == "count"

        assert len(fixer.fixes_applied) == 1

    def test_compatible_types_unchanged(self):
        fixer = AgentFixer()
        block_id = generate_uuid()

        src_node = _make_node(node_id="src", block_id=block_id)
        sink_node = _make_node(node_id="sink", block_id=block_id)
        link = _make_link(
            source_id="src",
            source_name="output",
            sink_id="sink",
            sink_name="input",
        )
        agent = _make_agent(nodes=[src_node, sink_node], links=[link])

        blocks = [
            self._make_block(
                block_id,
                input_schema={"properties": {"input": {"type": "string"}}},
                output_schema={"properties": {"output": {"type": "string"}}},
            ),
        ]

        result = fixer.fix_data_type_mismatch(agent, blocks)

        # No converter inserted, original link kept
        assert len(result["nodes"]) == 2
        assert len(result["links"]) == 1
        assert result["links"][0] is link
        assert fixer.fixes_applied == []

    def test_missing_block_keeps_link(self):
        """Links referencing unknown blocks are kept unchanged."""
        fixer = AgentFixer()
        src_node = _make_node(node_id="src", block_id="unknown-block")
        sink_node = _make_node(node_id="sink", block_id="unknown-block")
        link = _make_link(source_id="src", sink_id="sink")
        agent = _make_agent(nodes=[src_node, sink_node], links=[link])

        result = fixer.fix_data_type_mismatch(agent, blocks=[])

        assert len(result["links"]) == 1
        assert result["links"][0] is link

    def test_missing_type_info_keeps_link(self):
        """Links where source/sink type is not defined are kept unchanged."""
        fixer = AgentFixer()
        block_id = generate_uuid()
        src_node = _make_node(node_id="src", block_id=block_id)
        sink_node = _make_node(node_id="sink", block_id=block_id)
        link = _make_link(
            source_id="src",
            source_name="output",
            sink_id="sink",
            sink_name="input",
        )
        agent = _make_agent(nodes=[src_node, sink_node], links=[link])

        # Block has no properties defined for the linked fields
        blocks = [self._make_block(block_id)]

        result = fixer.fix_data_type_mismatch(agent, blocks)

        assert len(result["links"]) == 1
        assert fixer.fixes_applied == []

    def test_multiple_mismatches_insert_multiple_converters(self):
        """Each incompatible link gets its own converter node."""
        fixer = AgentFixer()
        src_block_id = generate_uuid()
        sink_block_id = generate_uuid()

        src_node = _make_node(node_id="src", block_id=src_block_id)
        sink1 = _make_node(node_id="sink1", block_id=sink_block_id)
        sink2 = _make_node(node_id="sink2", block_id=sink_block_id)
        link1 = _make_link(
            source_id="src", source_name="out", sink_id="sink1", sink_name="count"
        )
        link2 = _make_link(
            source_id="src", source_name="out", sink_id="sink2", sink_name="count"
        )
        agent = _make_agent(nodes=[src_node, sink1, sink2], links=[link1, link2])

        blocks = [
            self._make_block(
                src_block_id,
                output_schema={"properties": {"out": {"type": "string"}}},
            ),
            self._make_block(
                sink_block_id,
                input_schema={"properties": {"count": {"type": "integer"}}},
            ),
        ]

        result = fixer.fix_data_type_mismatch(agent, blocks)

        converter_nodes = [
            n
            for n in result["nodes"]
            if n["block_id"] == _UNIVERSAL_TYPE_CONVERTER_BLOCK_ID
        ]
        assert len(converter_nodes) == 2
        # Each original link becomes two links through its own converter
        assert len(result["links"]) == 4
        assert len(fixer.fixes_applied) == 2
