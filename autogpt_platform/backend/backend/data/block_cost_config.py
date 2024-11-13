from typing import Type

from autogpt_libs.supabase_integration_credentials_store.store import (
    anthropic_credentials,
    did_credentials,
    groq_credentials,
    ideogram_credentials,
    jina_credentials,
    openai_credentials,
    replicate_credentials,
    revid_credentials,
    unreal_credentials,
)

from backend.blocks.ai_music_generator import AIMusicGeneratorBlock
from backend.blocks.ai_shortform_video_block import AIShortformVideoCreatorBlock
from backend.blocks.ideogram import IdeogramModelBlock
from backend.blocks.jina.embeddings import JinaEmbeddingBlock
from backend.blocks.jina.search import SearchTheWebBlock
from backend.blocks.llm import (
    MODEL_METADATA,
    AIConversationBlock,
    AIListGeneratorBlock,
    AIStructuredResponseGeneratorBlock,
    AITextGeneratorBlock,
    AITextSummarizerBlock,
    LlmModel,
)
from backend.blocks.replicate_flux_advanced import ReplicateFluxAdvancedModelBlock
from backend.blocks.search import ExtractWebsiteContentBlock
from backend.blocks.talking_head import CreateTalkingAvatarVideoBlock
from backend.blocks.text_to_speech_block import UnrealTextToSpeechBlock
from backend.data.block import Block
from backend.data.cost import BlockCost, BlockCostType

# =============== Configure the cost for each LLM Model call =============== #

MODEL_COST: dict[LlmModel, int] = {
    LlmModel.O1_PREVIEW: 16,
    LlmModel.O1_MINI: 4,
    LlmModel.GPT4O_MINI: 1,
    LlmModel.GPT4O: 3,
    LlmModel.GPT4_TURBO: 10,
    LlmModel.GPT3_5_TURBO: 1,
    LlmModel.CLAUDE_3_5_SONNET: 4,
    LlmModel.CLAUDE_3_HAIKU: 1,
    LlmModel.LLAMA3_8B: 1,
    LlmModel.LLAMA3_70B: 1,
    LlmModel.MIXTRAL_8X7B: 1,
    LlmModel.GEMMA_7B: 1,
    LlmModel.GEMMA2_9B: 1,
    LlmModel.LLAMA3_1_405B: 1,
    LlmModel.LLAMA3_1_70B: 1,
    LlmModel.LLAMA3_1_8B: 1,
    LlmModel.OLLAMA_LLAMA3_8B: 1,
    LlmModel.OLLAMA_LLAMA3_405B: 1,
}

for model in LlmModel:
    if model not in MODEL_COST:
        raise ValueError(f"Missing MODEL_COST for model: {model}")


LLM_COST = (
    [
        BlockCost(
            cost_type=BlockCostType.RUN,
            cost_filter={
                "model": model,
                "api_key": None,  # Running LLM with user own API key is free.
            },
            cost_amount=cost,
        )
        for model, cost in MODEL_COST.items()
    ]
    + [
        BlockCost(
            cost_type=BlockCostType.RUN,
            cost_filter={
                "model": model,
                "credentials": {
                    "id": anthropic_credentials.id,
                    "provider": anthropic_credentials.provider,
                    "type": anthropic_credentials.type,
                },
            },
            cost_amount=cost,
        )
        for model, cost in MODEL_COST.items()
        if MODEL_METADATA[model].provider == "anthropic"
    ]
    + [
        BlockCost(
            cost_type=BlockCostType.RUN,
            cost_filter={
                "model": model,
                "credentials": {
                    "id": openai_credentials.id,
                    "provider": openai_credentials.provider,
                    "type": openai_credentials.type,
                },
            },
            cost_amount=cost,
        )
        for model, cost in MODEL_COST.items()
        if MODEL_METADATA[model].provider == "openai"
    ]
    + [
        BlockCost(
            cost_type=BlockCostType.RUN,
            cost_filter={
                "model": model,
                "credentials": {"id": groq_credentials.id},
            },
            cost_amount=cost,
        )
        for model, cost in MODEL_COST.items()
        if MODEL_METADATA[model].provider == "groq"
    ]
    + [
        BlockCost(
            # Default cost is running LlmModel.GPT4O.
            cost_amount=MODEL_COST[LlmModel.GPT4O],
            cost_filter={"api_key": None},
        ),
    ]
)

# =============== This is the exhaustive list of cost for each Block =============== #

BLOCK_COSTS: dict[Type[Block], list[BlockCost]] = {
    AIConversationBlock: LLM_COST,
    AITextGeneratorBlock: LLM_COST,
    AIStructuredResponseGeneratorBlock: LLM_COST,
    AITextSummarizerBlock: LLM_COST,
    AIListGeneratorBlock: LLM_COST,
    CreateTalkingAvatarVideoBlock: [
        BlockCost(
            cost_amount=15,
            cost_filter={
                "credentials": {
                    "id": did_credentials.id,
                    "provider": did_credentials.provider,
                    "type": did_credentials.type,
                }
            },
        )
    ],
    SearchTheWebBlock: [
        BlockCost(
            cost_amount=1,
            cost_filter={
                "credentials": {
                    "id": jina_credentials.id,
                    "provider": jina_credentials.provider,
                    "type": jina_credentials.type,
                }
            },
        )
    ],
    ExtractWebsiteContentBlock: [
        BlockCost(cost_amount=1, cost_filter={"raw_content": False})
    ],
    IdeogramModelBlock: [
        BlockCost(
            cost_amount=16,
            cost_filter={
                "credentials": {
                    "id": ideogram_credentials.id,
                    "provider": ideogram_credentials.provider,
                    "type": ideogram_credentials.type,
                }
            },
        )
    ],
    AIShortformVideoCreatorBlock: [
        BlockCost(
            cost_amount=50,
            cost_filter={
                "credentials": {
                    "id": revid_credentials.id,
                    "provider": revid_credentials.provider,
                    "type": revid_credentials.type,
                }
            },
        )
    ],
    ReplicateFluxAdvancedModelBlock: [
        BlockCost(
            cost_amount=10,
            cost_filter={
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                }
            },
        )
    ],
    AIMusicGeneratorBlock: [
        BlockCost(
            cost_amount=11,
            cost_filter={
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                }
            },
        )
    ],
    JinaEmbeddingBlock: [
        BlockCost(
            cost_amount=12,
            cost_filter={
                "credentials": {
                    "id": jina_credentials.id,
                    "provider": jina_credentials.provider,
                    "type": jina_credentials.type,
                }
            },
        )
    ],
    UnrealTextToSpeechBlock: [
        BlockCost(
            cost_amount=5,
            cost_filter={
                "credentials": {
                    "id": unreal_credentials.id,
                    "provider": unreal_credentials.provider,
                    "type": unreal_credentials.type,
                }
            },
        )
    ],
}
