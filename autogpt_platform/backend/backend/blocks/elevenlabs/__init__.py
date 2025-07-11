"""
ElevenLabs integration blocks for AutoGPT Platform.
"""

# Speech generation blocks
from .speech import (
    ElevenLabsGenerateSpeechBlock,
    ElevenLabsGenerateSpeechWithTimestampsBlock,
)

# Speech-to-text blocks
from .transcription import (
    ElevenLabsTranscribeAudioAsyncBlock,
    ElevenLabsTranscribeAudioSyncBlock,
)

# Webhook trigger blocks
from .triggers import ElevenLabsWebhookTriggerBlock

# Utility blocks
from .utility import ElevenLabsGetUsageStatsBlock, ElevenLabsListModelsBlock

# Voice management blocks
from .voices import (
    ElevenLabsCreateVoiceCloneBlock,
    ElevenLabsDeleteVoiceBlock,
    ElevenLabsGetVoiceDetailsBlock,
    ElevenLabsListVoicesBlock,
)

__all__ = [
    # Voice management
    "ElevenLabsListVoicesBlock",
    "ElevenLabsGetVoiceDetailsBlock",
    "ElevenLabsCreateVoiceCloneBlock",
    "ElevenLabsDeleteVoiceBlock",
    # Speech generation
    "ElevenLabsGenerateSpeechBlock",
    "ElevenLabsGenerateSpeechWithTimestampsBlock",
    # Speech-to-text
    "ElevenLabsTranscribeAudioSyncBlock",
    "ElevenLabsTranscribeAudioAsyncBlock",
    # Utility
    "ElevenLabsListModelsBlock",
    "ElevenLabsGetUsageStatsBlock",
    # Webhook triggers
    "ElevenLabsWebhookTriggerBlock",
]
