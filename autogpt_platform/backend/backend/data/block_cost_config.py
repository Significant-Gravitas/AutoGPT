import math
from typing import TYPE_CHECKING, Type

from pydantic import BaseModel

from backend.blocks._base import Block, BlockCost, BlockCostType
from backend.data.block import BlockInput

if TYPE_CHECKING:
    from backend.data.model import NodeExecutionStats
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
from backend.blocks.code_executor import (
    ExecuteCodeBlock,
    ExecuteCodeStepBlock,
    InstantiateCodeSandboxBlock,
)
from backend.blocks.codex import CodeGenerationBlock, CodexModel
from backend.blocks.enrichlayer.linkedin import (
    GetLinkedinProfileBlock,
    GetLinkedinProfilePictureBlock,
    LinkedinPersonLookupBlock,
    LinkedinRoleLookupBlock,
)
from backend.blocks.fal.ai_video_generator import AIVideoGeneratorBlock
from backend.blocks.flux_kontext import AIImageEditorBlock, FluxKontextModelName
from backend.blocks.ideogram import IdeogramModelBlock
from backend.blocks.jina.chunking import JinaChunkingBlock
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
from backend.blocks.youtube import TranscribeYoutubeVideoBlock
from backend.blocks.zerobounce.validate_emails import ValidateEmailsBlock
from backend.integrations.credentials_store import (
    aiml_api_credentials,
    anthropic_credentials,
    apollo_credentials,
    did_credentials,
    e2b_credentials,
    elevenlabs_credentials,
    enrichlayer_credentials,
    fal_credentials,
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
    webshare_proxy_credentials,
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
    LlmModel.KIMI_K2_0905: 1,
    LlmModel.KIMI_K2_5: 1,
    LlmModel.KIMI_K2_6: 2,
    LlmModel.KIMI_K2_THINKING: 2,
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


class TokenRate(BaseModel):
    """Per-token credit rates for a specific model.

    Each field is credits per 1,000,000 tokens of the corresponding kind.
    Cache-read and cache-write are 0 by default for providers that don't
    surface them (most non-Anthropic). Amounts use float so small rates
    (e.g. 0.2 credits / 1M Gemini Flash input) don't round away.
    """

    input: float
    output: float
    cache_read: float = 0.0
    cache_creation: float = 0.0


# TOKEN_COST populates gradually as we migrate LLM blocks to the TOKENS
# cost type. Entries not yet listed fall back to the flat MODEL_COST tier
# via the RUN-based LLM_COST list. Rates below are credits/1M tokens at the
# current credit-to-USD conversion (1 credit ≈ $0.01), with a 1.5x margin
# over the published provider price.
TOKEN_COST: dict[LlmModel, TokenRate] = {
    # Anthropic Opus ($15/$75/$1.50/$18.75 per 1M).
    LlmModel.CLAUDE_4_1_OPUS: TokenRate(
        input=2250, output=11250, cache_read=225, cache_creation=2812
    ),
    LlmModel.CLAUDE_4_OPUS: TokenRate(
        input=2250, output=11250, cache_read=225, cache_creation=2812
    ),
    LlmModel.CLAUDE_4_6_OPUS: TokenRate(
        input=2250, output=11250, cache_read=225, cache_creation=2812
    ),
    LlmModel.CLAUDE_4_5_OPUS: TokenRate(
        input=2250, output=11250, cache_read=225, cache_creation=2812
    ),
    # Anthropic Sonnet ($3/$15/$0.30/$3.75).
    LlmModel.CLAUDE_4_SONNET: TokenRate(
        input=450, output=2250, cache_read=45, cache_creation=562
    ),
    LlmModel.CLAUDE_4_6_SONNET: TokenRate(
        input=450, output=2250, cache_read=45, cache_creation=562
    ),
    LlmModel.CLAUDE_4_5_SONNET: TokenRate(
        input=450, output=2250, cache_read=45, cache_creation=562
    ),
    # Anthropic Haiku ($0.80/$4/$0.08/$1).
    LlmModel.CLAUDE_4_5_HAIKU: TokenRate(
        input=120, output=600, cache_read=12, cache_creation=150
    ),
    LlmModel.CLAUDE_3_HAIKU: TokenRate(input=37, output=187),
    # OpenAI
    LlmModel.GPT5_2: TokenRate(input=600, output=2400),
    LlmModel.GPT5_1: TokenRate(input=450, output=1800),
    LlmModel.GPT5: TokenRate(input=375, output=1500),
    LlmModel.GPT5_MINI: TokenRate(input=22, output=90),
    LlmModel.GPT5_NANO: TokenRate(input=7, output=30),
    LlmModel.GPT5_CHAT: TokenRate(input=375, output=1500),
    LlmModel.GPT4O: TokenRate(input=375, output=1500),
    LlmModel.GPT4O_MINI: TokenRate(input=22, output=90),
    LlmModel.GPT41: TokenRate(input=300, output=1200),
    LlmModel.GPT41_MINI: TokenRate(input=60, output=240),
    LlmModel.GPT4_TURBO: TokenRate(input=1500, output=4500),
    LlmModel.O3: TokenRate(input=1500, output=6000),
    LlmModel.O3_MINI: TokenRate(input=165, output=660),
    LlmModel.O1: TokenRate(input=2250, output=9000),
    LlmModel.O1_MINI: TokenRate(input=165, output=660),
    # Google Gemini
    LlmModel.GEMINI_2_5_PRO: TokenRate(input=187, output=750),
    LlmModel.GEMINI_2_5_PRO_PREVIEW: TokenRate(input=187, output=750),
    LlmModel.GEMINI_2_5_FLASH: TokenRate(input=11, output=45),
    LlmModel.GEMINI_2_5_FLASH_LITE_PREVIEW: TokenRate(input=5, output=22),
    LlmModel.GEMINI_2_0_FLASH: TokenRate(input=11, output=45),
    LlmModel.GEMINI_2_0_FLASH_LITE: TokenRate(input=5, output=22),
    LlmModel.GEMINI_3_1_PRO_PREVIEW: TokenRate(input=750, output=3000),
    LlmModel.GEMINI_3_FLASH_PREVIEW: TokenRate(input=15, output=60),
    LlmModel.GEMINI_3_1_FLASH_LITE_PREVIEW: TokenRate(input=5, output=22),
    # xAI Grok
    LlmModel.GROK_3: TokenRate(input=450, output=2250),
    LlmModel.GROK_4: TokenRate(input=2250, output=11250),
    LlmModel.GROK_4_FAST: TokenRate(input=37, output=150),
    LlmModel.GROK_4_1_FAST: TokenRate(input=37, output=150),
    LlmModel.GROK_4_20: TokenRate(input=750, output=3000),
    LlmModel.GROK_CODE_FAST_1: TokenRate(input=37, output=150),
    # DeepSeek
    LlmModel.DEEPSEEK_CHAT: TokenRate(input=40, output=165),
    LlmModel.DEEPSEEK_R1_0528: TokenRate(input=82, output=328),
    # Mistral
    LlmModel.MISTRAL_LARGE_3: TokenRate(input=300, output=900),
    LlmModel.MISTRAL_MEDIUM_3_1: TokenRate(input=405, output=1215),
    LlmModel.MISTRAL_SMALL_3_2: TokenRate(input=15, output=45),
    LlmModel.MISTRAL_NEMO: TokenRate(input=15, output=45),
    LlmModel.CODESTRAL: TokenRate(input=22, output=67),
    # Cohere
    LlmModel.COHERE_COMMAND_R_08_2024: TokenRate(input=22, output=90),
    LlmModel.COHERE_COMMAND_R_PLUS_08_2024: TokenRate(input=375, output=1500),
    LlmModel.COHERE_COMMAND_A_03_2025: TokenRate(input=187, output=750),
    # Moonshot Kimi
    LlmModel.KIMI_K2: TokenRate(input=90, output=375),
    LlmModel.KIMI_K2_0905: TokenRate(input=90, output=375),
    LlmModel.KIMI_K2_5: TokenRate(input=90, output=375),
    LlmModel.KIMI_K2_6: TokenRate(input=225, output=900),
    LlmModel.KIMI_K2_THINKING: TokenRate(input=225, output=900),
    # Perplexity Sonar
    LlmModel.PERPLEXITY_SONAR: TokenRate(input=30, output=30),
    LlmModel.PERPLEXITY_SONAR_PRO: TokenRate(input=150, output=150),
    LlmModel.PERPLEXITY_SONAR_REASONING_PRO: TokenRate(input=150, output=750),
    LlmModel.PERPLEXITY_SONAR_DEEP_RESEARCH: TokenRate(input=750, output=3750),
    # Groq (LLama + OpenAI OSS)
    LlmModel.LLAMA3_3_70B: TokenRate(input=88, output=118),
    LlmModel.LLAMA3_1_8B: TokenRate(input=7, output=12),
    LlmModel.META_LLAMA_4_SCOUT: TokenRate(input=22, output=75),
    LlmModel.META_LLAMA_4_MAVERICK: TokenRate(input=45, output=97),
    LlmModel.OPENAI_GPT_OSS_120B: TokenRate(input=22, output=67),
    LlmModel.OPENAI_GPT_OSS_20B: TokenRate(input=15, output=45),
}


def compute_token_credits(
    input_data: BlockInput, stats: "NodeExecutionStats | None"
) -> int:
    """Compute the credit charge for a TOKENS-billed LLM call from stats.

    Falls back to MODEL_COST[model] (the per-model flat tier) when the
    model has no TOKEN_COST entry or stats haven't been populated yet
    (pre-flight). Callers in block_usage_cost handle the TOKENS branch.
    """
    if stats is None:
        # Pre-flight — use the flat MODEL_COST entry as an estimate.
        raw_model = input_data.get("model")
        model = _lookup_llm_model(raw_model)
        return MODEL_COST.get(model, 0) if model else 0

    raw_model = input_data.get("model")
    model = _lookup_llm_model(raw_model)
    rate = TOKEN_COST.get(model) if model else None
    if rate is None:
        # Unmapped model — charge the per-call flat tier so we don't under-bill.
        return MODEL_COST.get(model, 0) if model else 0

    total = (
        stats.input_token_count * rate.input
        + stats.output_token_count * rate.output
        + stats.cache_read_token_count * rate.cache_read
        + stats.cache_creation_token_count * rate.cache_creation
    )
    return max(0, math.ceil(total / 1_000_000))


def _lookup_llm_model(raw: "str | LlmModel | None") -> "LlmModel | None":
    if raw is None:
        return None
    if isinstance(raw, LlmModel):
        return raw
    try:
        return LlmModel(raw)
    except ValueError:
        return None


LLM_COST = (
    # Anthropic Models
    [
        BlockCost(
            cost_type=BlockCostType.TOKENS,
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
            cost_type=BlockCostType.TOKENS,
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
            cost_type=BlockCostType.TOKENS,
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
            cost_type=BlockCostType.TOKENS,
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
            cost_type=BlockCostType.TOKENS,
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
            cost_type=BlockCostType.TOKENS,
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
            cost_type=BlockCostType.TOKENS,
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
    # Jina Reader Search: $0.01/query on the paid tier. Block emits
    # merge_stats(provider_cost=0.01, cost_usd) so the COST_USD resolver
    # bills 1 platform credit per call AND the platform cost telemetry
    # captures real USD spend (RUN wouldn't populate costMicrodollars).
    SearchTheWebBlock: [
        BlockCost(
            cost_amount=100,
            cost_type=BlockCostType.COST_USD,
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
    # Apollo Search blocks: bill per returned record. Blocks already emit
    # provider_cost=float(len(people/organizations)) with
    # provider_cost_type="items" via merge_stats, so ITEMS multiplies the
    # count by cost_amount post-flight. Pre-flight returns 0 (unknown
    # result count). enrich_info=True doubles the provider-side unit cost
    # (email enrichment), so we bill 2cr/person vs 1cr/person.
    SearchOrganizationsBlock: [
        BlockCost(
            cost_amount=1,
            cost_type=BlockCostType.ITEMS,
            cost_divisor=2,
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
            cost_amount=2,
            cost_type=BlockCostType.ITEMS,
            cost_filter={
                "enrich_info": True,
                "credentials": {
                    "id": apollo_credentials.id,
                    "provider": apollo_credentials.provider,
                    "type": apollo_credentials.type,
                },
            },
        ),
        BlockCost(
            cost_amount=1,
            cost_type=BlockCostType.ITEMS,
            cost_filter={
                "enrich_info": False,
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
    # Block emits merge_stats(provider_cost=0.008, cost_usd) so the platform
    # cost telemetry records real USD; resolver bills 2 credits per call
    # (ceil(0.008 * 250)).
    ValidateEmailsBlock: [
        BlockCost(
            cost_amount=250,
            cost_type=BlockCostType.COST_USD,
            cost_filter={
                "credentials": {
                    "id": zerobounce_credentials.id,
                    "provider": zerobounce_credentials.provider,
                    "type": zerobounce_credentials.type,
                }
            },
        )
    ],
    # E2B code-execution blocks: Hobby tier ~$0.000014/vCPU-s × 2 vCPU ≈
    # $0.000028/s. Charge 1 credit per 10 seconds of walltime (~$0.0003)
    # — recovers infra cost with margin and scales with session length.
    # Pre-flight returns 0 (walltime unknown); reconciliation charges the
    # true walltime after the block finishes (manager.py calls
    # billing.charge_reconciled_usage on completion).
    ExecuteCodeBlock: [
        BlockCost(
            cost_amount=1,
            cost_type=BlockCostType.SECOND,
            cost_divisor=10,
            cost_filter={
                "credentials": {
                    "id": e2b_credentials.id,
                    "provider": e2b_credentials.provider,
                    "type": e2b_credentials.type,
                }
            },
        )
    ],
    InstantiateCodeSandboxBlock: [
        BlockCost(
            cost_amount=1,
            cost_type=BlockCostType.SECOND,
            cost_divisor=10,
            cost_filter={
                "credentials": {
                    "id": e2b_credentials.id,
                    "provider": e2b_credentials.provider,
                    "type": e2b_credentials.type,
                }
            },
        )
    ],
    ExecuteCodeStepBlock: [
        BlockCost(
            cost_amount=1,
            cost_type=BlockCostType.SECOND,
            cost_divisor=10,
            cost_filter={
                "credentials": {
                    "id": e2b_credentials.id,
                    "provider": e2b_credentials.provider,
                    "type": e2b_credentials.type,
                }
            },
        )
    ],
    # FAL video generation: $0.001–$0.02 per output second. Charge 3 credits
    # per walltime second (~$0.03) — covers the median tier with margin and
    # scales fairly across short clips vs long renders.
    AIVideoGeneratorBlock: [
        BlockCost(
            cost_amount=3,
            cost_type=BlockCostType.SECOND,
            cost_filter={
                "credentials": {
                    "id": fal_credentials.id,
                    "provider": fal_credentials.provider,
                    "type": fal_credentials.type,
                }
            },
        )
    ],
    # Webshare is a flat monthly proxy subscription — the per-call cost to us
    # is effectively zero, but the transcription step itself consumes compute
    # time we haven't otherwise charged for. 1 credit is a tooling-tax floor.
    TranscribeYoutubeVideoBlock: [
        BlockCost(
            cost_amount=1,
            cost_filter={
                "credentials": {
                    "id": webshare_proxy_credentials.id,
                    "provider": webshare_proxy_credentials.provider,
                    "type": webshare_proxy_credentials.type,
                }
            },
        )
    ],
    # ClaudeCodeBlock runs an E2B sandbox AND executes Claude Sonnet inside it.
    # Real cost $0.50-$2/run; flat 100 credits is conservative until we pipe
    # x-total-cost from the in-sandbox Claude calls into provider_cost.
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
    # Ayrshare post blocks use the @cost(...) decorator directly on each block
    # class (see backend/blocks/ayrshare/_cost.py). They can't be listed here
    # because post_to_*.py imports from backend.sdk, which imports from this
    # module — registering via decorator avoids the circular import.
    # Jina chunking: $0.02/1M tokens. Flat 1 credit floor so the block is not
    # wallet-free; embedding/search already have their own entries.
    JinaChunkingBlock: [
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
}
