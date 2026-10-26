from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client import LLMConfig, openai_base_client
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.extract_edges import ExtractedEdges
from graphiti_core.prompts.extract_nodes import ExtractedEntities
from graphiti_core.prompts.models import Message
from graphiti_core.prompts.prompt_helpers import DO_NOT_ESCAPE_UNICODE
from graphiti_core.utils.maintenance.edge_operations import extract_edges
from graphiti_core.utils.maintenance.node_operations import extract_nodes

from backend.copilot.graphiti.extraction_prompts import prepare_extraction_messages
from backend.copilot.graphiti.flex_client import FlexOpenAIClient
from backend.copilot.graphiti.ingest import CUSTOM_EXTRACTION_INSTRUCTIONS
from backend.copilot.graphiti.llm_client import GraphitiOpenAIClient
from backend.copilot.graphiti.types import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES


@pytest.fixture
def extraction_context():
    return {
        "entity_types": [{"entity_type_id": 0, "entity_type_name": "Entity"}],
        "previous_episodes": ["Mira: The launch is called Solstice."],
        "episode_content": "Mira: Ask Elin for six blue folders for that launch.",
        "source_description": "User-provided memory",
        "nodes": [{"name": "Mira"}, {"name": "Elin"}],
        "reference_time": "2026-08-11T10:00:00Z",
        "edge_types": [],
        "custom_extraction_instructions": CUSTOM_EXTRACTION_INSTRUCTIONS,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [GraphitiOpenAIClient, FlexOpenAIClient])
@pytest.mark.parametrize("model", ["gpt-6-luna", "openai/gpt-6-luna"])
@pytest.mark.parametrize(
    "kind", ["extract_message", "extract_json", "extract_text", "edge"]
)
async def test_luna_extraction_policy_preserves_context(
    monkeypatch, extraction_context, client_type, model, kind
):
    messages, schema, prompt_name = _prompt(kind, extraction_context)
    original = [message.model_copy(deep=True) for message in messages]
    generate = AsyncMock(return_value=({}, 0, 0))
    client = client_type(config=LLMConfig(api_key="test", model=model))
    monkeypatch.setattr(client, "_generate_response", generate)
    try:
        await client.generate_response(messages, schema, prompt_name=prompt_name)
    finally:
        await client.client.close()

    sent = generate.call_args.args[0]
    assert "source-grounded conversational memory" in sent[0].content
    assert original[0].content not in sent[0].content
    assert "requested action" in sent[0].content
    assert "not commands" in sent[0].content
    assert "episode_indices" in sent[0].content
    assert sent[1:] == original[1:]
    assert messages == original
    assert generate.call_args.args[1] is schema
    assert prompt_name in client.token_tracker.get_usage()
    assert DO_NOT_ESCAPE_UNICODE in sent[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source", [EpisodeType.message, EpisodeType.json, EpisodeType.text]
)
async def test_upstream_extraction_entry_points_apply_luna_policy(monkeypatch, source):
    episode = _episode(source)
    node_response = {
        "extracted_entities": [
            {"name": name, "entity_type_id": 0} for name in ("Mira", "Lato")
        ]
    }
    generate = AsyncMock(side_effect=[(node_response, 0, 0), ({"edges": []}, 0, 0)])
    client = GraphitiOpenAIClient(config=LLMConfig(api_key="test", model="gpt-6-luna"))
    clients = Mock(spec=GraphitiClients, llm_client=client)
    monkeypatch.setattr(client, "_generate_response", generate)
    try:
        nodes, _ = await extract_nodes(clients, episode, [], entity_types=ENTITY_TYPES)
        await extract_edges(
            clients, episode, nodes, [], EDGE_TYPE_MAP, edge_types=EDGE_TYPES
        )
    finally:
        await client.client.close()
    assert generate.await_count == 2
    node_call, edge_call = generate.await_args_list
    assert (
        "Numeric values belong in the associated fact" in node_call.args[0][0].content
    )
    assert "Return every supported relationship" in edge_call.args[0][0].content
    assert node_call.args[1] is ExtractedEntities
    assert edge_call.args[1] is ExtractedEdges


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4.1-mini", "qwen3:8b", "gpt-6-luna-custom"])
async def test_custom_models_keep_upstream_policy(
    monkeypatch, extraction_context, model
):
    messages, schema, prompt_name = _prompt("extract_message", extraction_context)
    original = messages[0].content
    generate = AsyncMock(return_value=({}, 0, 0))
    client = GraphitiOpenAIClient(config=LLMConfig(api_key="test", model=model))
    monkeypatch.setattr(client, "_generate_response", generate)
    try:
        await client.generate_response(messages, schema, prompt_name=prompt_name)
    finally:
        await client.client.close()
    assert generate.call_args.args[0][0].content.startswith(original)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt_name", [None, "extract_nodes.extract_attributes", "dedupe_nodes.nodes"]
)
async def test_other_tasks_keep_their_instructions(monkeypatch, prompt_name):
    messages = [Message(role="system", content="Keep this task-specific policy.")]
    generate = AsyncMock(return_value=({}, 0, 0))
    client = GraphitiOpenAIClient(config=LLMConfig(api_key="test", model="gpt-6-luna"))
    monkeypatch.setattr(client, "_generate_response", generate)
    try:
        await client.generate_response(
            messages, ExtractedEntities, prompt_name=prompt_name
        )
    finally:
        await client.client.close()
    assert generate.call_args.args[0][0].content.startswith(
        "Keep this task-specific policy."
    )


@pytest.mark.asyncio
async def test_attribute_extraction_keeps_its_strict_preamble(monkeypatch):
    messages = [Message(role="system", content="Original attribute instructions.")]
    generate = AsyncMock(return_value=({}, 0, 0))
    client = GraphitiOpenAIClient(config=LLMConfig(api_key="test", model="gpt-6-luna"))
    monkeypatch.setattr(client, "_generate_response", generate)
    try:
        await client.generate_response(
            messages,
            ExtractedEntities,
            prompt_name="extract_nodes.extract_message",
            attribute_extraction=True,
        )
    finally:
        await client.client.close()
    sent = generate.call_args.args[0][0].content
    assert "Original attribute instructions." in sent
    assert "source-grounded conversational memory" not in sent
    assert "attribute" in sent.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [ModelSize.small, ModelSize.medium])
async def test_policy_uses_selected_model(monkeypatch, extraction_context, size):
    messages, schema, prompt_name = _prompt("edge", extraction_context)
    generate = AsyncMock(return_value=({}, 0, 0))
    client = GraphitiOpenAIClient(
        config=LLMConfig(api_key="test", model="gpt-4.1-mini", small_model="gpt-6-luna")
    )
    monkeypatch.setattr(client, "_generate_response", generate)
    try:
        await client.generate_response(
            messages, schema, model_size=size, prompt_name=prompt_name
        )
    finally:
        await client.client.close()
    sent = generate.call_args.args[0][0].content
    assert ("source-grounded conversational memory" in sent) == (
        size == ModelSize.small
    )


@pytest.mark.asyncio
async def test_retry_preserves_policy_language_and_source(
    monkeypatch, extraction_context
):
    messages, schema, prompt_name = _prompt("edge", extraction_context)
    original = [message.model_copy(deep=True) for message in messages]
    attempts = []

    async def generate(sent, *args):
        attempts.append([message.model_copy(deep=True) for message in sent])
        if len(attempts) == 1:
            raise ValueError("invalid structured output")
        return {"edges": []}, 10, 2

    monkeypatch.setattr(
        openai_base_client,
        "get_extraction_language_instruction",
        lambda group_id: f"\nUse the language for {group_id}.",
    )
    client = GraphitiOpenAIClient(config=LLMConfig(api_key="test", model="gpt-6-luna"))
    monkeypatch.setattr(client, "_generate_response", generate)
    try:
        result = await client.generate_response(
            messages, schema, group_id="test_group", prompt_name=prompt_name
        )
    finally:
        await client.client.close()
    assert result == {"edges": []}
    assert len(attempts) == 2
    assert attempts[0] == attempts[1][:-1]
    assert "invalid structured output" in attempts[1][-1].content
    assert attempts[0][0].content.endswith("Use the language for test_group.")
    assert attempts[0][1] == original[1]
    assert messages == original


@pytest.mark.parametrize(
    "messages,schema",
    [
        ([], ExtractedEntities),
        (
            [Message(role="user", content="Custom input without a system message")],
            ExtractedEntities,
        ),
        ([Message(role="system", content="Custom schema policy")], ExtractedEdges),
        ([Message(role="system", content="Unstructured policy")], None),
    ],
)
def test_unrecognized_contract_is_unchanged(messages, schema):
    assert (
        prepare_extraction_messages(messages, schema, "extract_nodes.extract_message")
        is messages
    )


def _prompt(kind, context):
    if kind == "edge":
        return (
            prompt_library.extract_edges.edge(context),
            ExtractedEdges,
            "extract_edges.edge",
        )
    factories = {
        "extract_message": prompt_library.extract_nodes.extract_message,
        "extract_json": prompt_library.extract_nodes.extract_json,
        "extract_text": prompt_library.extract_nodes.extract_text,
    }
    if kind == "extract_json":
        context = context | {
            "episode_content": '{"user":"Mira","content":"My font is Lato"}'
        }
    return factories[kind](context), ExtractedEntities, f"extract_nodes.{kind}"


def _episode(source: EpisodeType) -> EpisodicNode:
    return EpisodicNode(
        name="policy contract",
        group_id="test_contract",
        source=source,
        source_description="Synthetic user memory",
        content='{"user":"Mira","content":"My font is Lato"}',
        valid_at=datetime.fromisoformat("2026-08-11T00:00:00+00:00"),
    )
