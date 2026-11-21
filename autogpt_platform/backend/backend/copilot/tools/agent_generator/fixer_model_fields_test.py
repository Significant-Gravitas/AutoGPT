"""The fixer's check of an AI block's ``model`` input, on the real pipeline and
the real block schemas: an LLM model field holds a model the block offers,
written as the block lists it, while a free-form model field (a Jina embedding
model, say) keeps what the generator wrote."""

import pytest

from backend.blocks.io import AgentInputBlock, AgentOutputBlock
from backend.blocks.jina.embeddings import JinaEmbeddingBlock
from backend.blocks.llm import AITextGeneratorBlock

from .fixer_switched_off_model_test import (
    AI_NODE,
    ENABLED_OPENAI,
    INPUT_NODE,
    OUTPUT_NODE,
    WORKFLOW_ID,
    link,
    model_on,
    node,
    preview,
    workflow,
)

JINA_DEFAULT = JinaEmbeddingBlock().input_schema.jsonschema()["properties"]["model"][
    "default"
]


def embedding_workflow(jina_inputs: dict) -> dict:
    """Input -> Jina Embedding -> output, as Otto writes one."""
    return {
        "id": WORKFLOW_ID,
        "name": "Embed Notes",
        "description": "Embeds a note with Jina.",
        "version": 1,
        "is_active": True,
        "nodes": [
            node(INPUT_NODE, AgentInputBlock().id, {"name": "note"}),
            node(AI_NODE, JinaEmbeddingBlock().id, jina_inputs),
            node(OUTPUT_NODE, AgentOutputBlock().id, {"name": "embeddings"}),
        ],
        "links": [
            link(INPUT_NODE, "result", AI_NODE, "texts"),
            link(AI_NODE, "embeddings", OUTPUT_NODE, "value"),
        ],
    }


@pytest.mark.asyncio
async def test_a_jina_embedding_node_keeps_the_model_the_generator_wrote():
    """Jina's model is free text the LLM lists don't know; replacing it with an
    LLM made the saved node send gpt-4o to Jina's embeddings API."""
    result = await preview(embedding_workflow({"model": "jina-embeddings-v3"}))

    assert model_on(result, AI_NODE) == "jina-embeddings-v3"


@pytest.mark.asyncio
async def test_a_jina_embedding_node_that_names_no_model_gets_the_blocks_default():
    result = await preview(embedding_workflow({}))

    assert model_on(result, AI_NODE) == JINA_DEFAULT


@pytest.mark.asyncio
async def test_a_new_node_on_a_prefixed_spelling_of_an_offered_model_keeps_it():
    """The enum takes a provider prefix, so the value names a model the block
    offers. It is written as the block lists it instead of being replaced by
    the block's default."""
    result = await preview(workflow(AITextGeneratorBlock, f"openai/{ENABLED_OPENAI}"))

    assert model_on(result, AI_NODE) == ENABLED_OPENAI
