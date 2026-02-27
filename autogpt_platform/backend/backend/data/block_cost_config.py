import logging
from typing import Type

import prisma.models

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
from backend.blocks.jina.search import ExtractWebsiteContentBlock, SearchTheWebBlock
from backend.blocks.llm import (
    AIConversationBlock,
    AIListGeneratorBlock,
    AIStructuredResponseGeneratorBlock,
    AITextGeneratorBlock,
    AITextSummarizerBlock,
)
from backend.blocks.replicate.flux_advanced import ReplicateFluxAdvancedModelBlock
from backend.blocks.replicate.replicate_block import ReplicateModelBlock
from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock
from backend.blocks.talking_head import CreateTalkingAvatarVideoBlock
from backend.blocks.text_to_speech_block import UnrealTextToSpeechBlock
from backend.blocks.video.narration import VideoNarrationBlock
from backend.data import llm_registry
from backend.integrations.credentials_store import (
    aiml_api_credentials,
    anthropic_credentials,
    apollo_credentials,
    did_credentials,
    elevenlabs_credentials,
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

logger = logging.getLogger(__name__)

PROVIDER_CREDENTIALS = {
    "openai": openai_credentials,
    "anthropic": anthropic_credentials,
    "groq": groq_credentials,
    "open_router": open_router_credentials,
    "llama_api": llama_api_credentials,
    "aiml_api": aiml_api_credentials,
    "v0": v0_credentials,
}

# =============== Configure the cost for each LLM Model call =============== #
# All LLM costs now come from the database via llm_registry

LLM_COST: list[BlockCost] = []


async def _build_llm_costs_from_registry() -> list[BlockCost]:
    """
    Build BlockCost list from all models in the LLM registry.

    This function checks for active model migrations with customCreditCost overrides.
    When a model has been migrated with a custom price, that price is used instead
    of the target model's default cost.
    """
    # Query active migrations with custom pricing overrides.
    # Note: LlmModelMigration is system-level data (no userId field) and this function
    # is only called during app startup and admin operations, so no user ID filter needed.
    migration_overrides: dict[str, int] = {}
    try:
        active_migrations = await prisma.models.LlmModelMigration.prisma().find_many(
            where={
                "isReverted": False,
                "customCreditCost": {"not": None},
            }
        )
        # Key by targetModelSlug since that's the model nodes are now using
        # after migration. The custom cost applies to the target model.
        migration_overrides = {
            migration.targetModelSlug: migration.customCreditCost
            for migration in active_migrations
            if migration.customCreditCost is not None
        }
        if migration_overrides:
            logger.info(
                "Found %d active model migrations with custom pricing overrides",
                len(migration_overrides),
            )
    except Exception as exc:
        logger.warning(
            "Failed to query model migration overrides: %s. Proceeding with default costs.",
            exc,
            exc_info=True,
        )

    costs: list[BlockCost] = []
    for model in llm_registry.iter_dynamic_models():
        for cost in model.costs:
            credentials = PROVIDER_CREDENTIALS.get(cost.credential_provider)
            if not credentials:
                logger.warning(
                    "Skipping cost entry for %s due to unknown credentials provider %s",
                    model.slug,
                    cost.credential_provider,
                )
                continue

            # Check if this model has a custom cost override from migration
            cost_amount = migration_overrides.get(model.slug, cost.credit_cost)

            if model.slug in migration_overrides:
                logger.debug(
                    "Applying custom cost override for model %s: %d credits (default: %d)",
                    model.slug,
                    cost_amount,
                    cost.credit_cost,
                )

            cost_filter = {
                "model": model.slug,
                "credentials": {
                    "id": credentials.id,
                    "provider": credentials.provider,
                    "type": credentials.type,
                },
            }
            costs.append(
                BlockCost(
                    cost_type=BlockCostType.RUN,
                    cost_filter=cost_filter,
                    cost_amount=cost_amount,
                )
            )
    return costs


async def refresh_llm_costs() -> None:
    """
    Refresh LLM costs from the registry. All costs now come from the database.

    This function also checks for active model migrations with custom pricing overrides
    and applies them to ensure accurate billing.
    """
    # Build new costs first, then swap atomically to avoid race condition
    # where concurrent readers see an empty list during the await
    new_costs = await _build_llm_costs_from_registry()
    LLM_COST.clear()
    LLM_COST.extend(new_costs)


# Initial load will happen after registry is refreshed at startup
# Don't call refresh_llm_costs() here - it will be called after registry refresh

# =============== This is the exhaustive list of cost for each Block =============== #

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
    ],
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
}
