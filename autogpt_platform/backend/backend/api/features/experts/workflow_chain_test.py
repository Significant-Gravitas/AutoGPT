import json

import prisma

from backend.api.features.experts.workflow_chain import build_workflow_chain

AGENT_INPUT_BLOCK = "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b"
AGENT_OUTPUT_BLOCK = "363ae599-353e-4804-937e-b2ee3cef3da4"
GMAIL_READ_BLOCK = "25310c70-b89b-43ba-b25c-4dfa7e2a481c"
AI_TEXT_GENERATOR_BLOCK = "1f292d4a-41a4-4977-9684-7c8d560b9f91"


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
