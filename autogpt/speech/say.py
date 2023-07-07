""" Text to speech module """
from __future__ import annotations

import threading
from threading import Semaphore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogpt.config import Config

from .base import VoiceBase
from .eleven_labs import ElevenLabsSpeech
from .gtts import GTTSVoice
from .macos_tts import MacOSTTS
from .stream_elements_speech import StreamElementsSpeech

_QUEUE_SEMAPHORE = Semaphore(
    1
)  # The amount of sounds to queue before blocking the main thread


def say_text(text: str, config: Config, voice_index: int = 0) -> None:
    """Speak the given text using the given voice index"""
    default_voice_engine, voice_engine = _get_voice_engine(config)

    def speak() -> None:
        success = voice_engine.say(text, voice_index)
        if not success:
            default_voice_engine.say(text)

        _QUEUE_SEMAPHORE.release()

    _QUEUE_SEMAPHORE.acquire(True)
    thread = threading.Thread(target=speak)
    thread.start()


def _get_voice_engine(config: Config) -> tuple[VoiceBase, VoiceBase]:
    """Get the voice engine to use for the given configuration"""
    tts_provider = config.text_to_speech_provider
    if tts_provider == "elevenlabs":
        voice_engine = ElevenLabsSpeech(config)
    elif tts_provider == "macos":
        voice_engine = MacOSTTS(config)
    elif tts_provider == "streamelements":
        voice_engine = StreamElementsSpeech(config)
    else:
        voice_engine = GTTSVoice(config)

    return GTTSVoice(config), voice_engine
