from typing import Type

from backend.blocks.ai_music_generator import AIMusicGeneratorBlock
from backend.blocks.ai_shortform_video_block import AIShortformVideoCreatorBlock
from backend.blocks.apollo.organization import SearchOrganizationsBlock
from backend.blocks.apollo.people import SearchPeopleBlock
from backend.blocks.apollo.person import GetPersonDetailBlock
from backend.blocks.enrichlayer.linkedin import (
    GetLinkedinProfileBlock,
    GetLinkedinProfilePictureBlock,
    LinkedinPersonLookupBlock,
    LinkedinRoleLookupBlock,
)
from backend.blocks.flux_kontext import AIImageEditorBlock, FluxKontextModelName
from backend.blocks.ideogram import IdeogramModelBlock
from backend.blocks.jina.embeddings import JinaEmbeddingBlock
from backend.blocks.jina.search import ExtractWebsiteContentBlock, SearchTheWebBlock
from backend.blocks.llm import (
    MODEL_METADATA,
    AIConversationBlock,
    AIListGeneratorBlock,
    AIStructuredResponseGeneratorBlock,
    AITextGeneratorBlock,
    AITextSummarizerBlock,
    LlmModel,
)
from backend.blocks.replicate.flux_advanced import ReplicateFluxAdvancedModelBlock
from backend.blocks.replicate.replicate_block import ReplicateModelBlock
from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock
from backend.blocks.talking_head import CreateTalkingAvatarVideoBlock
from backend.blocks.text_to_speech_block import UnrealTextToSpeechBlock
from backend.data.block import Block, BlockCost, BlockCostType
from backend.integrations.credentials_store import (
    aiml_api_credentials,
    anthropic_credentials,
    apollo_credentials,
    did_credentials,
    enrichlayer_credentials,
    groq_credentials,
    ideogram_credentials,
    jina_credentials,
    llama_api_credentials,
    open_router_credentials,
    openai_credentials,
    replicate_credentials,
    revid_credentials,
    unreal_credentials,
    v0_credentials,
)

# =============== Configure the cost for each LLM Model call =============== #

MODEL_COST: dict[LlmModel, int] = {
    LlmModel.O3: 4,
    LlmModel.O3_MINI: 2,  # $1.10 / $4.40
    LlmModel.O1: 16,  # $15 / $60
    LlmModel.O1_MINI: 4,
    # GPT-5 models
    LlmModel.GPT5: 2,
    LlmModel.GPT5_MINI: 1,
    LlmModel.GPT5_NANO: 1,
    LlmModel.GPT5_CHAT: 2,
    LlmModel.GPT41: 2,
    LlmModel.GPT41_MINI: 1,
    LlmModel.GPT4O_MINI: 1,
    LlmModel.GPT4O: 3,
    LlmModel.GPT4_TURBO: 10,
    LlmModel.GPT3_5_TURBO: 1,
    LlmModel.CLAUDE_4_1_OPUS: 21,
    LlmModel.CLAUDE_4_OPUS: 21,
    LlmModel.CLAUDE_4_SONNET: 5,
    LlmModel.CLAUDE_4_5_SONNET: 9,
    LlmModel.CLAUDE_3_7_SONNET: 5,
    LlmModel.CLAUDE_3_5_SONNET: 4,
    LlmModel.CLAUDE_3_5_HAIKU: 1,  # $0.80 / $4.00
    LlmModel.CLAUDE_3_HAIKU: 1,
    LlmModel.AIML_API_QWEN2_5_72B: 1,
    LlmModel.AIML_API_LLAMA3_1_70B: 1,
    LlmModel.AIML_API_LLAMA3_3_70B: 1,
    LlmModel.AIML_API_META_LLAMA_3_1_70B: 1,
    LlmModel.AIML_API_LLAMA_3_2_3B: 1,
    LlmModel.LLAMA3_8B: 1,
    LlmModel.LLAMA3_70B: 1,
    LlmModel.GEMMA2_9B: 1,
    LlmModel.LLAMA3_3_70B: 1,  # $0.59 / $0.79
    LlmModel.LLAMA3_1_8B: 1,
    LlmModel.OLLAMA_LLAMA3_3: 1,
    LlmModel.OLLAMA_LLAMA3_2: 1,
    LlmModel.OLLAMA_LLAMA3_8B: 1,
    LlmModel.OLLAMA_LLAMA3_405B: 1,
    LlmModel.DEEPSEEK_LLAMA_70B: 1,  # ? / ?
    LlmModel.OLLAMA_DOLPHIN: 1,
    LlmModel.OPENAI_GPT_OSS_120B: 1,
    LlmModel.OPENAI_GPT_OSS_20B: 1,
    LlmModel.GEMINI_FLASH_1_5: 1,
    LlmModel.GEMINI_2_5_PRO: 4,
    LlmModel.MISTRAL_NEMO: 1,
    LlmModel.COHERE_COMMAND_R_08_2024: 1,
    LlmModel.COHERE_COMMAND_R_PLUS_08_2024: 3,
    LlmModel.DEEPSEEK_CHAT: 2,
    LlmModel.PERPLEXITY_SONAR: 1,
    LlmModel.PERPLEXITY_SONAR_PRO: 5,
    LlmModel.PERPLEXITY_SONAR_DEEP_RESEARCH: 10,
    LlmModel.NOUSRESEARCH_HERMES_3_LLAMA_3_1_405B: 1,
    LlmModel.NOUSRESEARCH_HERMES_3_LLAMA_3_1_70B: 1,
    LlmModel.AMAZON_NOVA_LITE_V1: 1,
    LlmModel.AMAZON_NOVA_MICRO_V1: 1,
    LlmModel.AMAZON_NOVA_PRO_V1: 1,
    LlmModel.MICROSOFT_WIZARDLM_2_8X22B: 1,
    LlmModel.GRYPHE_MYTHOMAX_L2_13B: 1,
    LlmModel.META_LLAMA_4_SCOUT: 1,
    LlmModel.META_LLAMA_4_MAVERICK: 1,
    LlmModel.LLAMA_API_LLAMA_4_SCOUT: 1,
    LlmModel.LLAMA_API_LLAMA4_MAVERICK: 1,
    LlmModel.LLAMA_API_LLAMA3_3_8B: 1,
    LlmModel.LLAMA_API_LLAMA3_3_70B: 1,
    LlmModel.GROK_4: 9,
    LlmModel.KIMI_K2: 1,
    LlmModel.QWEN3_235B_A22B_THINKING: 1,
    LlmModel.QWEN3_CODER: 9,
    LlmModel.GEMINI_2_5_FLASH: 1,
    LlmModel.GEMINI_2_0_FLASH: 1,
    LlmModel.GEMINI_2_5_FLASH_LITE_PREVIEW: 1,
    LlmModel.GEMINI_2_0_FLASH_LITE: 1,
    LlmModel.DEEPSEEK_R1_0528: 1,
    # v0 by Vercel models
    LlmModel.V0_1_5_MD: 1,
    LlmModel.V0_1_5_LG: 2,
    LlmModel.V0_1_0_MD: 1,
}

for model in LlmModel:
    if model not in MODEL_COST:
        raise ValueError(f"Missing MODEL_COST for model: {model}")


LLM_COST = (
    # Anthropic Models
    [
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
    # OpenAI Models
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
    # Groq Models
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
    # Open Router Models
    + [
        BlockCost(
            cost_type=BlockCostType.RUN,
            cost_filter={
                "model": model,
                "credentials": {
                    "id": open_router_credentials.id,
                    "provider": open_router_credentials.provider,
                    "type": open_router_credentials.type,
                },
            },
            cost_amount=cost,
        )
        for model, cost in MODEL_COST.items()
        if MODEL_METADATA[model].provider == "open_router"
    ]
    # Llama API Models
    + [
        BlockCost(
            cost_type=BlockCostType.RUN,
            cost_filter={
                "model": model,
                "credentials": {
                    "id": llama_api_credentials.id,
                    "provider": llama_api_credentials.provider,
                    "type": llama_api_credentials.type,
                },
            },
            cost_amount=cost,
        )
        for model, cost in MODEL_COST.items()
        if MODEL_METADATA[model].provider == "llama_api"
    ]
    # v0 by Vercel Models
    + [
        BlockCost(
            cost_type=BlockCostType.RUN,
            cost_filter={
                "model": model,
                "credentials": {
                    "id": v0_credentials.id,
                    "provider": v0_credentials.provider,
                    "type": v0_credentials.type,
                },
            },
            cost_amount=cost,
        )
        for model, cost in MODEL_COST.items()
        if MODEL_METADATA[model].provider == "v0"
    ]
    # AI/ML Api Models
    + [
        BlockCost(
            cost_type=BlockCostType.RUN,
            cost_filter={
                "model": model,
                "credentials": {
                    "id": aiml_api_credentials.id,
                    "provider": aiml_api_credentials.provider,
                    "type": aiml_api_credentials.type,
                },
            },
            cost_amount=cost,
        )
        for model, cost in MODEL_COST.items()
        if MODEL_METADATA[model].provider == "aiml_api"
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
        BlockCost(
            cost_amount=1,
            cost_filter={
                "raw_content": False,
                "credentials": {
                    "id": jina_credentials.id,
                    "provider": jina_credentials.provider,
                    "type": jina_credentials.type,
                },
            },
        )
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
        ),
        BlockCost(
            cost_amount=18,
            cost_filter={
                "ideogram_model_name": "V_3",
                "credentials": {
                    "id": ideogram_credentials.id,
                    "provider": ideogram_credentials.provider,
                    "type": ideogram_credentials.type,
                },
            },
        ),
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
    ReplicateModelBlock: [
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
    AIImageEditorBlock: [
        BlockCost(
            cost_amount=10,
            cost_filter={
                "model": FluxKontextModelName.PRO.api_name,
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                },
            },
        ),
        BlockCost(
            cost_amount=20,
            cost_filter={
                "model": FluxKontextModelName.MAX.api_name,
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                },
            },
        ),
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
    GetLinkedinProfileBlock: [
        BlockCost(
            cost_amount=1,
            cost_filter={
                "credentials": {
                    "id": enrichlayer_credentials.id,
                    "provider": enrichlayer_credentials.provider,
                    "type": enrichlayer_credentials.type,
                }
            },
        )
    ],
    LinkedinPersonLookupBlock: [
        BlockCost(
            cost_amount=2,
            cost_filter={
                "credentials": {
                    "id": enrichlayer_credentials.id,
                    "provider": enrichlayer_credentials.provider,
                    "type": enrichlayer_credentials.type,
                }
            },
        )
    ],
    LinkedinRoleLookupBlock: [
        BlockCost(
            cost_amount=3,
            cost_filter={
                "credentials": {
                    "id": enrichlayer_credentials.id,
                    "provider": enrichlayer_credentials.provider,
                    "type": enrichlayer_credentials.type,
                }
            },
        )
    ],
    GetLinkedinProfilePictureBlock: [
        BlockCost(
            cost_amount=3,
            cost_filter={
                "credentials": {
                    "id": enrichlayer_credentials.id,
                    "provider": enrichlayer_credentials.provider,
                    "type": enrichlayer_credentials.type,
                }
            },
        )
    ],
    SmartDecisionMakerBlock: LLM_COST,
    SearchOrganizationsBlock: [
        BlockCost(
            cost_amount=2,
            cost_filter={
                "credentials": {
                    "id": apollo_credentials.id,
                    "provider": apollo_credentials.provider,
                    "type": apollo_credentials.type,
                }
            },
        )
    ],
    SearchPeopleBlock: [
        BlockCost(
            cost_amount=10,
            cost_filter={
                "enrich_info": False,
                "credentials": {
                    "id": apollo_credentials.id,
                    "provider": apollo_credentials.provider,
                    "type": apollo_credentials.type,
                },
            },
        ),
        BlockCost(
            cost_amount=20,
            cost_filter={
                "enrich_info": True,
                "credentials": {
                    "id": apollo_credentials.id,
                    "provider": apollo_credentials.provider,
                    "type": apollo_credentials.type,
                },
            },
        ),
    ],
    GetPersonDetailBlock: [
        BlockCost(
            cost_amount=1,
            cost_filter={
                "credentials": {
                    "id": apollo_credentials.id,
                    "provider": apollo_credentials.provider,
                    "type": apollo_credentials.type,
                }
            },
        )
    ],
}
