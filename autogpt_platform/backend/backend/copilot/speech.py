"""Text-to-speech for AutoPilot voice mode, metered like a chat turn.

A spoken reply debits the same per-user microdollar counter and writes the
same ``PlatformCostLog`` row as the LLM turn that produced it, so voice is
in-plan usage rather than a free side channel.
"""

import logging

from openai import AsyncOpenAI

from backend.copilot.config import ChatConfig
from backend.copilot.token_tracking import persist_and_record_usage
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

config = ChatConfig()
settings = Settings()

# OpenAI rejects longer inputs outright; the client chunks at sentence
# boundaries well below this, so hitting it means a malformed request.
MAX_SPEECH_CHARS = 4096

ALLOWED_VOICES = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "nova",
        "onyx",
        "sage",
        "shimmer",
        "verse",
    }
)

AUDIO_MEDIA_TYPE = "audio/mpeg"

TTS_BLOCK_NAME = "copilot:tts"


class SpeechUnavailable(Exception):
    """No OpenAI key is configured, so voice mode cannot synthesise."""


async def synthesize_speech(
    *,
    user_id: str,
    text: str,
    session_id: str | None = None,
    voice: str | None = None,
) -> bytes:
    """Speak *text* as MP3 and meter its cost against the user's plan.

    Raises ``SpeechUnavailable`` when no key is configured and ``ValueError``
    for an empty/oversized input or an unknown voice.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("text is empty")
    if len(cleaned) > MAX_SPEECH_CHARS:
        raise ValueError(f"text exceeds {MAX_SPEECH_CHARS} characters")

    resolved_voice = voice or config.voice_tts_voice
    if resolved_voice not in ALLOWED_VOICES:
        raise ValueError(f"unknown voice {resolved_voice!r}")

    response = await _speech_client().audio.speech.create(
        model=config.voice_tts_model,
        voice=resolved_voice,
        input=cleaned,
        response_format="mp3",
    )
    audio = await response.aread()

    await _meter_speech(
        user_id=user_id,
        characters=len(cleaned),
        session_id=session_id,
        voice=resolved_voice,
        audio_bytes=len(audio),
    )
    return audio


async def _meter_speech(
    *,
    user_id: str,
    characters: int,
    session_id: str | None,
    voice: str,
    audio_bytes: int,
) -> None:
    """Charge the turn's TTS to the same counters a text turn writes to."""
    cost_usd = speech_cost_usd(characters)
    await persist_and_record_usage(
        session=None,
        user_id=user_id,
        prompt_tokens=0,
        completion_tokens=0,
        log_prefix="[Voice]",
        cost_usd=cost_usd,
        model=config.voice_tts_model,
        provider="openai",
        block_name_override=TTS_BLOCK_NAME,
        graph_exec_id_override=session_id,
        extra_metadata={
            "surface": "voice_mode",
            "characters": characters,
            "voice": voice,
            "audio_bytes": audio_bytes,
        },
    )


def speech_cost_usd(characters: int) -> float:
    """Our USD cost for synthesising *characters*.

    The speech endpoint reports no usage, so this is a rate estimate rather
    than a billed figure — see ``ChatConfig.voice_tts_usd_per_1k_chars``.
    """
    return characters / 1000 * config.voice_tts_usd_per_1k_chars


def _speech_client() -> AsyncOpenAI:
    """Direct OpenAI client — OpenRouter does not proxy the speech endpoint."""
    api_key = (
        settings.secrets.openai_internal_api_key or settings.secrets.openai_api_key
    )
    if not api_key:
        raise SpeechUnavailable("no OpenAI API key configured")
    return AsyncOpenAI(api_key=api_key)
