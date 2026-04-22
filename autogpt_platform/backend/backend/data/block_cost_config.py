from typing import Type

from backend.blocks._base import Block, BlockCost, BlockCostType
from backend.blocks.ai_image_customizer import AIImageCustomizerBlock, GeminiImageModel
from backend.blocks.ai_image_generator_block import AIImageGeneratorBlock, ImageGenModel
from backend.blocks.ai_music_generator import AIMusicGeneratorBlock
from backend.blocks.ai_shortform_video_block import (
    AIAdMakerVideoCreatorBlock,
    AIScreenshotToVideoAdBlock,
    AIShortformVideoCreatorBlock,
)
from backend.blocks.apollo.organization import SearchOrganizationsBlock
from backend.blocks.apollo.people import SearchPeopleBlock
from backend.blocks.apollo.person import GetPersonDetailBlock
from backend.blocks.claude_code import ClaudeCodeBlock
from backend.blocks.codex import CodeGenerationBlock, CodexModel
from backend.blocks.enrichlayer.linkedin import (
    GetLinkedinProfileBlock,
    GetLinkedinProfilePictureBlock,
    LinkedinPersonLookupBlock,
    LinkedinRoleLookupBlock,
)
from backend.blocks.flux_kontext import AIImageEditorBlock, FluxKontextModelName
from backend.blocks.ideogram import IdeogramModelBlock
from backend.blocks.jina.embeddings import JinaEmbeddingBlock
from backend.blocks.jina.fact_checker import FactCheckerBlock
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
from backend.blocks.mem0 import (
    AddMemoryBlock,
    GetAllMemoriesBlock,
    GetLatestMemoryBlock,
    SearchMemoryBlock,
)
from backend.blocks.nvidia.deepfake import NvidiaDeepfakeDetectBlock
from backend.blocks.orchestrator import OrchestratorBlock
from backend.blocks.perplexity import PerplexityBlock, PerplexityModel
from backend.blocks.replicate.flux_advanced import ReplicateFluxAdvancedModelBlock
from backend.blocks.replicate.replicate_block import ReplicateModelBlock
from backend.blocks.screenshotone import ScreenshotWebPageBlock
from backend.blocks.smartlead.campaign import (
    AddLeadToCampaignBlock,
    CreateCampaignBlock,
    SaveCampaignSequencesBlock,
)
from backend.blocks.talking_head import CreateTalkingAvatarVideoBlock
from backend.blocks.text_to_speech_block import UnrealTextToSpeechBlock
from backend.blocks.video.narration import VideoNarrationBlock
from backend.blocks.zerobounce.validate_emails import ValidateEmailsBlock
from backend.integrations.credentials_store import (
    aiml_api_credentials,
    anthropic_credentials,
    apollo_credentials,
    did_credentials,
    e2b_credentials,
    elevenlabs_credentials,
    enrichlayer_credentials,
    groq_credentials,
    ideogram_credentials,
    jina_credentials,
    llama_api_credentials,
    mem0_credentials,
    nvidia_credentials,
    open_router_credentials,
    openai_credentials,
    replicate_credentials,
    revid_credentials,
    screenshotone_credentials,
    smartlead_credentials,
    unreal_credentials,
    v0_credentials,
    zerobounce_credentials,
)

# =============== Configure the cost for each LLM Model call =============== #

MODEL_COST: dict[LlmModel, int] = {
    LlmModel.O3: 4,
    LlmModel.O3_MINI: 2,
    LlmModel.O1: 16,
    LlmModel.O1_MINI: 4,
    # GPT-5 models
    LlmModel.GPT5_2: 6,
    LlmModel.GPT5_1: 5,
    LlmModel.GPT5: 2,
    LlmModel.GPT5_MINI: 1,
    LlmModel.GPT5_NANO: 1,
    LlmModel.GPT5_CHAT: 5,
    LlmModel.GPT41: 2,
    LlmModel.GPT41_MINI: 1,
    LlmModel.GPT4O_MINI: 1,
    LlmModel.GPT4O: 3,
    LlmModel.GPT4_TURBO: 10,
    LlmModel.CLAUDE_4_1_OPUS: 21,
    LlmModel.CLAUDE_4_OPUS: 21,
    LlmModel.CLAUDE_4_SONNET: 5,
    LlmModel.CLAUDE_4_6_OPUS: 14,
    LlmModel.CLAUDE_4_6_SONNET: 9,
    LlmModel.CLAUDE_4_5_HAIKU: 4,
    LlmModel.CLAUDE_4_5_OPUS: 14,
    LlmModel.CLAUDE_4_5_SONNET: 9,
    LlmModel.CLAUDE_3_HAIKU: 1,
    LlmModel.AIML_API_QWEN2_5_72B: 1,
    LlmModel.AIML_API_LLAMA3_1_70B: 1,
    LlmModel.AIML_API_LLAMA3_3_70B: 1,
    LlmModel.AIML_API_META_LLAMA_3_1_70B: 1,
    LlmModel.AIML_API_LLAMA_3_2_3B: 1,
    LlmModel.LLAMA3_3_70B: 1,
    LlmModel.LLAMA3_1_8B: 1,
    LlmModel.OLLAMA_LLAMA3_3: 1,
    LlmModel.OLLAMA_LLAMA3_2: 1,
    LlmModel.OLLAMA_LLAMA3_8B: 1,
    LlmModel.OLLAMA_LLAMA3_405B: 1,
    LlmModel.OLLAMA_DOLPHIN: 1,
    LlmModel.OPENAI_GPT_OSS_120B: 1,
    LlmModel.OPENAI_GPT_OSS_20B: 1,
    LlmModel.GEMINI_2_5_PRO_PREVIEW: 4,
    LlmModel.GEMINI_2_5_PRO: 4,
    LlmModel.GEMINI_3_1_PRO_PREVIEW: 5,
    LlmModel.GEMINI_3_FLASH_PREVIEW: 2,
    LlmModel.GEMINI_2_5_FLASH: 1,
    LlmModel.GEMINI_2_0_FLASH: 1,
    LlmModel.GEMINI_3_1_FLASH_LITE_PREVIEW: 1,
    LlmModel.GEMINI_2_5_FLASH_LITE_PREVIEW: 1,
    LlmModel.GEMINI_2_0_FLASH_LITE: 1,
    LlmModel.MISTRAL_NEMO: 1,
    LlmModel.MISTRAL_LARGE_3: 2,
    LlmModel.MISTRAL_MEDIUM_3_1: 2,
    LlmModel.MISTRAL_SMALL_3_2: 1,
    LlmModel.CODESTRAL: 1,
    LlmModel.COHERE_COMMAND_R_08_2024: 1,
    LlmModel.COHERE_COMMAND_R_PLUS_08_2024: 3,
    LlmModel.COHERE_COMMAND_A_03_2025: 3,
    LlmModel.COHERE_COMMAND_A_TRANSLATE_08_2025: 3,
    LlmModel.COHERE_COMMAND_A_REASONING_08_2025: 6,
    LlmModel.COHERE_COMMAND_A_VISION_07_2025: 3,
    LlmModel.DEEPSEEK_CHAT: 2,
    LlmModel.DEEPSEEK_R1_0528: 1,
    LlmModel.PERPLEXITY_SONAR: 1,
    LlmModel.PERPLEXITY_SONAR_PRO: 5,
    LlmModel.PERPLEXITY_SONAR_REASONING_PRO: 5,
    LlmModel.PERPLEXITY_SONAR_DEEP_RESEARCH: 10,
    LlmModel.NOUSRESEARCH_HERMES_3_LLAMA_3_1_405B: 1,
    LlmModel.NOUSRESEARCH_HERMES_3_LLAMA_3_1_70B: 1,
    LlmModel.AMAZON_NOVA_LITE_V1: 1,
    LlmModel.AMAZON_NOVA_MICRO_V1: 1,
    LlmModel.AMAZON_NOVA_PRO_V1: 1,
    LlmModel.MICROSOFT_WIZARDLM_2_8X22B: 1,
    LlmModel.MICROSOFT_PHI_4: 1,
    LlmModel.GRYPHE_MYTHOMAX_L2_13B: 1,
    LlmModel.META_LLAMA_4_SCOUT: 1,
    LlmModel.META_LLAMA_4_MAVERICK: 1,
    LlmModel.LLAMA_API_LLAMA_4_SCOUT: 1,
    LlmModel.LLAMA_API_LLAMA4_MAVERICK: 1,
    LlmModel.LLAMA_API_LLAMA3_3_8B: 1,
    LlmModel.LLAMA_API_LLAMA3_3_70B: 1,
    LlmModel.GROK_3: 3,
    LlmModel.GROK_4: 9,
    LlmModel.GROK_4_FAST: 1,
    LlmModel.GROK_4_1_FAST: 1,
    LlmModel.GROK_4_20: 5,
    LlmModel.GROK_4_20_MULTI_AGENT: 5,
    LlmModel.GROK_CODE_FAST_1: 1,
    LlmModel.KIMI_K2: 1,
    LlmModel.QWEN3_235B_A22B_THINKING: 1,
    LlmModel.QWEN3_CODER: 9,
    # Z.ai (Zhipu) models
    LlmModel.ZAI_GLM_4_32B: 1,
    LlmModel.ZAI_GLM_4_5: 2,
    LlmModel.ZAI_GLM_4_5_AIR: 1,
    LlmModel.ZAI_GLM_4_5_AIR_FREE: 1,
    LlmModel.ZAI_GLM_4_5V: 2,
    LlmModel.ZAI_GLM_4_6: 1,
    LlmModel.ZAI_GLM_4_6V: 1,
    LlmModel.ZAI_GLM_4_7: 1,
    LlmModel.ZAI_GLM_4_7_FLASH: 1,
    LlmModel.ZAI_GLM_5: 2,
    LlmModel.ZAI_GLM_5_TURBO: 4,
    LlmModel.ZAI_GLM_5V_TURBO: 4,
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
#
# BLOCK_COSTS drives the **credit wallet** — the user-facing balance that funds
# block executions regardless of where they run (builder, graph execution,
# copilot ``run_block`` tool). A missing entry here makes the block run for
# free from the wallet's perspective, even when the upstream provider charges
# real USD. See ``backend.executor.utils::block_usage_cost`` for the lookup
# and ``backend.copilot.tools.helpers::execute_block`` for the copilot-side
# charge path.
#
# Credits are **not** the same as copilot microdollar rate-limit counters
# (``backend.copilot.rate_limit``). Microdollars track AutoGPT's infra cost
# (OpenRouter / Anthropic inference spend) and gate the chat loop; credits
# track the user's prepaid balance. A block running inside copilot ``run_block``
# decrements only the credit wallet via this table — microdollars stay scoped
# to copilot LLM turns and are not double-charged from block execution.
# See the module docstring on ``backend.copilot.rate_limit`` for the full
# boundary.

BLOCK_COSTS: dict[Type[Block], list[BlockCost]] = {
    AIConversationBlock: LLM_COST,
    AITextGeneratorBlock: LLM_COST,
    AIStructuredResponseGeneratorBlock: LLM_COST,
    AITextSummarizerBlock: LLM_COST,
    AIListGeneratorBlock: LLM_COST,
    CodeGenerationBlock: [
        BlockCost(
            cost_type=BlockCostType.RUN,
            cost_filter={
                "model": CodexModel.GPT5_1_CODEX,
                "credentials": {
                    "id": openai_credentials.id,
                    "provider": openai_credentials.provider,
                    "type": openai_credentials.type,
                },
            },
            cost_amount=5,
        )
    ],
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
            cost_amount=307,
            cost_filter={
                "credentials": {
                    "id": revid_credentials.id,
                    "provider": revid_credentials.provider,
                    "type": revid_credentials.type,
                }
            },
        )
    ],
    AIAdMakerVideoCreatorBlock: [
        BlockCost(
            cost_amount=714,
            cost_filter={
                "credentials": {
                    "id": revid_credentials.id,
                    "provider": revid_credentials.provider,
                    "type": revid_credentials.type,
                }
            },
        )
    ],
    AIScreenshotToVideoAdBlock: [
        BlockCost(
            cost_amount=612,
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
                "model": FluxKontextModelName.FLUX_KONTEXT_PRO,
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
                "model": FluxKontextModelName.FLUX_KONTEXT_MAX,
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                },
            },
        ),
        BlockCost(
            cost_amount=14,  # Nano Banana Pro
            cost_filter={
                "model": FluxKontextModelName.NANO_BANANA_PRO,
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                },
            },
        ),
        BlockCost(
            cost_amount=14,  # Nano Banana 2
            cost_filter={
                "model": FluxKontextModelName.NANO_BANANA_2,
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
    AIImageGeneratorBlock: [
        BlockCost(
            cost_amount=5,  # SD3.5 Medium: ~$0.035 per image
            cost_filter={
                "model": ImageGenModel.SD3_5,
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                },
            },
        ),
        BlockCost(
            cost_amount=6,  # Flux 1.1 Pro: ~$0.04 per image
            cost_filter={
                "model": ImageGenModel.FLUX,
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                },
            },
        ),
        BlockCost(
            cost_amount=10,  # Flux 1.1 Pro Ultra: ~$0.08 per image
            cost_filter={
                "model": ImageGenModel.FLUX_ULTRA,
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                },
            },
        ),
        BlockCost(
            cost_amount=7,  # Recraft v3: ~$0.05 per image
            cost_filter={
                "model": ImageGenModel.RECRAFT,
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                },
            },
        ),
        BlockCost(
            cost_amount=14,  # Nano Banana Pro: $0.14 per image at 2K
            cost_filter={
                "model": ImageGenModel.NANO_BANANA_PRO,
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                },
            },
        ),
        BlockCost(
            cost_amount=14,  # Nano Banana 2: same pricing tier as Pro
            cost_filter={
                "model": ImageGenModel.NANO_BANANA_2,
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                },
            },
        ),
    ],
    AIImageCustomizerBlock: [
        BlockCost(
            cost_amount=10,  # Nano Banana (original)
            cost_filter={
                "model": GeminiImageModel.NANO_BANANA,
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                },
            },
        ),
        BlockCost(
            cost_amount=14,  # Nano Banana Pro: $0.14 per image at 2K
            cost_filter={
                "model": GeminiImageModel.NANO_BANANA_PRO,
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                },
            },
        ),
        BlockCost(
            cost_amount=14,  # Nano Banana 2: same pricing tier as Pro
            cost_filter={
                "model": GeminiImageModel.NANO_BANANA_2,
                "credentials": {
                    "id": replicate_credentials.id,
                    "provider": replicate_credentials.provider,
                    "type": replicate_credentials.type,
                },
            },
        ),
    ],
    PerplexityBlock: [
        # Sonar Deep Research: up to $5/1K searches + $8/1M reasoning tokens.
        # Flat-charge 10 credits mirrors the LLM table's SONAR_DEEP_RESEARCH
        # entry. Block execution decrements only the user credit wallet via
        # spend_credits(); the microdollar rate-limit counter is not touched
        # for run_block invocations. The actual per-run provider spend is
        # recorded separately as provider_cost on PlatformCostLog when
        # OpenRouter reports usage.
        BlockCost(
            cost_amount=10,
            cost_filter={
                "model": PerplexityModel.SONAR_DEEP_RESEARCH,
                "credentials": {
                    "id": open_router_credentials.id,
                    "provider": open_router_credentials.provider,
                    "type": open_router_credentials.type,
                },
            },
        ),
        # Sonar Pro: $1/1M input + $1/1M output + $0.005/search.
        BlockCost(
            cost_amount=5,
            cost_filter={
                "model": PerplexityModel.SONAR_PRO,
                "credentials": {
                    "id": open_router_credentials.id,
                    "provider": open_router_credentials.provider,
                    "type": open_router_credentials.type,
                },
            },
        ),
        # Sonar (default): $0.2/1M input + $0.2/1M output + $0.005/search.
        BlockCost(
            cost_amount=1,
            cost_filter={
                "model": PerplexityModel.SONAR,
                "credentials": {
                    "id": open_router_credentials.id,
                    "provider": open_router_credentials.provider,
                    "type": open_router_credentials.type,
                },
            },
        ),
    ],
    FactCheckerBlock: [
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
    OrchestratorBlock: LLM_COST,
    VideoNarrationBlock: [
        BlockCost(
            cost_amount=5,  # ElevenLabs TTS cost
            cost_filter={
                "credentials": {
                    "id": elevenlabs_credentials.id,
                    "provider": elevenlabs_credentials.provider,
                    "type": elevenlabs_credentials.type,
                }
            },
        )
    ],
    # Mem0: Starter $19/mo for 50K adds + 5K retrievals → $0.0004/add,
    # $0.004/retrieval. Floor at 1 credit covers raw cost with margin.
    AddMemoryBlock: [
        BlockCost(
            cost_amount=1,
            cost_filter={
                "credentials": {
                    "id": mem0_credentials.id,
                    "provider": mem0_credentials.provider,
                    "type": mem0_credentials.type,
                }
            },
        )
    ],
    SearchMemoryBlock: [
        BlockCost(
            cost_amount=1,
            cost_filter={
                "credentials": {
                    "id": mem0_credentials.id,
                    "provider": mem0_credentials.provider,
                    "type": mem0_credentials.type,
                }
            },
        )
    ],
    GetAllMemoriesBlock: [
        BlockCost(
            cost_amount=1,
            cost_filter={
                "credentials": {
                    "id": mem0_credentials.id,
                    "provider": mem0_credentials.provider,
                    "type": mem0_credentials.type,
                }
            },
        )
    ],
    GetLatestMemoryBlock: [
        BlockCost(
            cost_amount=1,
            cost_filter={
                "credentials": {
                    "id": mem0_credentials.id,
                    "provider": mem0_credentials.provider,
                    "type": mem0_credentials.type,
                }
            },
        )
    ],
    # ScreenshotOne: $17 / 2K screenshots = $0.0085/call (Basic tier).
    ScreenshotWebPageBlock: [
        BlockCost(
            cost_amount=2,
            cost_filter={
                "credentials": {
                    "id": screenshotone_credentials.id,
                    "provider": screenshotone_credentials.provider,
                    "type": screenshotone_credentials.type,
                }
            },
        )
    ],
    # NVIDIA NIM hosted endpoints: no public per-call SKU; estimate based on
    # peer deepfake APIs (Hive/Sightengine ~$0.005-0.01/call).
    NvidiaDeepfakeDetectBlock: [
        BlockCost(
            cost_amount=2,
            cost_filter={
                "credentials": {
                    "id": nvidia_credentials.id,
                    "provider": nvidia_credentials.provider,
                    "type": nvidia_credentials.type,
                }
            },
        )
    ],
    # Smartlead: $39/mo Basic = $0.0065 per email-equivalent. Campaign
    # creation touches multiple records → 2 credits; per-lead and config
    # writes are lighter → 1 credit.
    CreateCampaignBlock: [
        BlockCost(
            cost_amount=2,
            cost_filter={
                "credentials": {
                    "id": smartlead_credentials.id,
                    "provider": smartlead_credentials.provider,
                    "type": smartlead_credentials.type,
                }
            },
        )
    ],
    AddLeadToCampaignBlock: [
        BlockCost(
            cost_amount=1,
            cost_filter={
                "credentials": {
                    "id": smartlead_credentials.id,
                    "provider": smartlead_credentials.provider,
                    "type": smartlead_credentials.type,
                }
            },
        )
    ],
    SaveCampaignSequencesBlock: [
        BlockCost(
            cost_amount=1,
            cost_filter={
                "credentials": {
                    "id": smartlead_credentials.id,
                    "provider": smartlead_credentials.provider,
                    "type": smartlead_credentials.type,
                }
            },
        )
    ],
    # ZeroBounce: $16 / 2K validations = $0.008 per email. One email per call.
    ValidateEmailsBlock: [
        BlockCost(
            cost_amount=2,
            cost_filter={
                "credentials": {
                    "id": zerobounce_credentials.id,
                    "provider": zerobounce_credentials.provider,
                    "type": zerobounce_credentials.type,
                }
            },
        )
    ],
    # ClaudeCodeBlock runs an E2B sandbox (~$0.00003/sec compute) AND
    # executes Claude Sonnet inside it. Real session cost is dominated by
    # the LLM and varies $0.50–$2 per typical run. Flat 100 credits ($1.00)
    # is a conservative-but-fair estimate; revisit once we expose the
    # x-total-cost header from the in-sandbox Claude calls back to
    # NodeExecutionStats.provider_cost.
    ClaudeCodeBlock: [
        BlockCost(
            cost_amount=100,
            cost_filter={
                "e2b_credentials": {
                    "id": e2b_credentials.id,
                    "provider": e2b_credentials.provider,
                    "type": e2b_credentials.type,
                }
            },
        )
    ],
}
