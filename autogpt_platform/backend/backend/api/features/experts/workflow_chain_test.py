import json

import prisma

from backend.api.features.experts.workflow_chain import (
    CHAIN_LENGTH,
    build_workflow_chain,
    integration_providers,
)

AGENT_INPUT_BLOCK = "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b"
AGENT_OUTPUT_BLOCK = "363ae599-353e-4804-937e-b2ee3cef3da4"
GMAIL_READ_BLOCK = "25310c70-b89b-43ba-b25c-4dfa7e2a481c"
AI_TEXT_GENERATOR_BLOCK = "1f292d4a-41a4-4977-9684-7c8d560b9f91"
# Four blocks whose providers a user must connect themselves — the platform
# holds credentials for neither of these, unlike Gmail's google above.
AIRTABLE_BLOCK = "f59b88a8-54ce-4676-a508-fd614b4e8dce"
AGENT_MAIL_BLOCK = "a283ffc4-8087-4c3d-9135-8f26b86742ec"
BANNERBEAR_BLOCK = "c7d3a5c2-05fc-450e-8dce-3b0e04626009"


def _node(block_id: str, constant_input: dict | None = None) -> prisma.models.AgentNode:
    return prisma.models.AgentNode(
        id=f"node-{block_id[:8]}-{json.dumps(constant_input or {})}",
        agentBlockId=block_id,
        agentGraphId="graph-1",
        agentGraphVersion=1,
        constantInput=json.dumps(constant_input or {}),
        metadata=json.dumps({}),
    )


def test_chain_puts_integrations_first_and_reads_input_to_output():
    nodes = [
        _node(AGENT_INPUT_BLOCK),
        _node(GMAIL_READ_BLOCK),
        _node(GMAIL_READ_BLOCK),
        _node(AI_TEXT_GENERATOR_BLOCK, {"model": "Llama-3.3-70B-Instruct"}),
        _node(AGENT_OUTPUT_BLOCK),
    ]

    chain = build_workflow_chain(nodes)

    assert [(item.kind, item.provider) for item in chain] == [
        ("input", None),
        ("integration", "google"),
        ("integration", "llama_api"),
    ]


def test_chain_falls_back_to_block_kinds_without_integrations():
    chain = build_workflow_chain([_node(AGENT_INPUT_BLOCK), _node(AGENT_OUTPUT_BLOCK)])

    assert [(item.kind, item.provider) for item in chain] == [
        ("input", None),
        ("output", None),
    ]


def test_chain_skips_unknown_blocks_and_unresolved_llm_models():
    chain = build_workflow_chain(
        [_node("not-a-block"), _node(AI_TEXT_GENERATOR_BLOCK, {"model": "mystery"})]
    )

    assert [(item.kind, item.provider) for item in chain] == [("ai", None)]


def test_integration_providers_keeps_what_the_display_cut_drops():
    """The chain is capped at three items, so a graph with four integrations
    loses one — the access list on the expert profile must not."""
    nodes = [
        _node(GMAIL_READ_BLOCK),
        _node(AIRTABLE_BLOCK),
        _node(AGENT_MAIL_BLOCK),
        _node(BANNERBEAR_BLOCK),
    ]

    chain = build_workflow_chain(nodes)
    providers = integration_providers(nodes)

    assert len(chain) == CHAIN_LENGTH
    assert len({item.provider for item in chain}) < len(providers)
    assert providers == ["agent_mail", "airtable", "bannerbear", "google"]


def test_integration_providers_ignores_non_credentialed_steps():
    nodes = [
        _node(AGENT_INPUT_BLOCK),
        _node(AGENT_OUTPUT_BLOCK),
        _node(AI_TEXT_GENERATOR_BLOCK, {"model": "mystery"}),
        _node(AIRTABLE_BLOCK),
    ]

    assert integration_providers(nodes) == ["airtable"]
