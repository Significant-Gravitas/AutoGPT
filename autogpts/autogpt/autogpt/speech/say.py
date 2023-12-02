""" Text to speech module """
from __future__ import annotations

import threading
from threading import Semaphore
from typing import Literal, Optional

from autogpt.core.configuration.schema import SystemConfiguration, UserConfigurable

from .base import VoiceBase
from .eleven_labs import ElevenLabsConfig, ElevenLabsSpeech
from .gtts import GTTSVoice
from .macos_tts import MacOSTTS
from .stream_elements_speech import StreamElementsConfig, StreamElementsSpeech

_QUEUE_SEMAPHORE = Semaphore(
    1
)  # The amount of sounds to queue before blocking the main thread


class TTSConfig(SystemConfiguration):
    speak_mode: bool = False
    provider: Literal[
        "elevenlabs", "gtts", "macos", "streamelements"
    ] = UserConfigurable(default="gtts")
    elevenlabs: Optional[ElevenLabsConfig] = None
    streamelements: Optional[StreamElementsConfig] = None


class TextToSpeechProvider:
    def __init__(self, config: TTSConfig):
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
        return "{class_name}(provider={voice_engine_name})".format(
            class_name=self.__class__.__name__,
            voice_engine_name=self._voice_engine.__class__.__name__,
        )

    @staticmethod
    def _get_voice_engine(config: TTSConfig) -> tuple[VoiceBase, VoiceBase]:
        """Get the voice engine to use for the given configuration"""
        tts_provider = config.provider
        if tts_provider == "elevenlabs":
            voice_engine = ElevenLabsSpeech(config.elevenlabs)
        elif tts_provider == "macos":
            voice_engine = MacOSTTS()
        elif tts_provider == "streamelements":
            voice_engine = StreamElementsSpeech(config.streamelements)
        else:
            voice_engine = GTTSVoice()

        return GTTSVoice(), voice_engine
