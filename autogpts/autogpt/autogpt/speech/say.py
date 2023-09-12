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


class TextToSpeechProvider:
    def __init__(self, config: Config):
        self._config = config
        self._default_voice_engine, self._voice_engine = self._get_voice_engine(config)

    def say(self, text, voice_index: int = 0) -> None:
        def _speak() -> None:
            success = self._voice_engine.say(text, voice_index)
            if not success:
                self._default_voice_engine.say(text, voice_index)
            _QUEUE_SEMAPHORE.release()

        if self._config.speak_mode:
            _QUEUE_SEMAPHORE.acquire(True)
            thread = threading.Thread(target=_speak)
            thread.start()

    def __repr__(self):
        return f"{self.__class__.__name__}(enabled={self._config.speak_mode}, provider={self._voice_engine.__class__.__name__})"

    @staticmethod
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
