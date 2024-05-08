from __future__ import annotations

import logging
import os

import requests
from playsound import playsound

from forge.config.schema import SystemConfiguration, UserConfigurable

from .base import VoiceBase

logger = logging.getLogger(__name__)


class StreamElementsConfig(SystemConfiguration):
    voice: str = UserConfigurable(default="Brian", from_env="STREAMELEMENTS_VOICE")


class StreamElementsSpeech(VoiceBase):
    """Streamelements speech module for autogpt"""

    def _setup(self, config: StreamElementsConfig) -> None:
        """Setup the voices, API key, etc."""
        self.config = config

    def _speech(self, text: str, voice: str, _: int = 0) -> bool:
        voice = self.config.voice
        """Speak text using the streamelements API

        Args:
            text (str): The text to speak
            voice (str): The voice to use

        Returns:
            bool: True if the request was successful, False otherwise
        """
        tts_url = (
            f"https://api.streamelements.com/kappa/v2/speech?voice={voice}&text={text}"
        )
        response = requests.get(tts_url)

        if response.status_code == 200:
            with open("speech.mp3", "wb") as f:
                f.write(response.content)
            playsound("speech.mp3")
            os.remove("speech.mp3")
            return True
        else:
            logger.error(
                "Request failed with status code: %s, response content: %s",
                response.status_code,
                response.content,
            )
            return False
