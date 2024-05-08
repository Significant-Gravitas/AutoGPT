"""ElevenLabs speech module"""
from __future__ import annotations

import logging
import os

import requests
from playsound import playsound

from forge.config.schema import SystemConfiguration, UserConfigurable

from .base import VoiceBase

logger = logging.getLogger(__name__)

PLACEHOLDERS = {"your-voice-id"}


class ElevenLabsConfig(SystemConfiguration):
    api_key: str = UserConfigurable(from_env="ELEVENLABS_API_KEY")
    voice_id: str = UserConfigurable(from_env="ELEVENLABS_VOICE_ID")


class ElevenLabsSpeech(VoiceBase):
    """ElevenLabs speech class"""

    def _setup(self, config: ElevenLabsConfig) -> None:
        """Set up the voices, API key, etc.

        Returns:
            None: None
        """

        default_voices = ["ErXwobaYiN019PkySvjV", "EXAVITQu4vr4xnSDxMaL"]
        voice_options = {
            "Rachel": "21m00Tcm4TlvDq8ikWAM",
            "Domi": "AZnzlk1XvdvUeBnXmlld",
            "Bella": "EXAVITQu4vr4xnSDxMaL",
            "Antoni": "ErXwobaYiN019PkySvjV",
            "Elli": "MF3mGyEYCl7XYWbV9V6O",
            "Josh": "TxGEqnHWrfWFTfGW9XjX",
            "Arnold": "VR6AewLTigWG4xSOukaG",
            "Adam": "pNInz6obpgDQGcFmaJgB",
            "Sam": "yoZ06aMxZJJ28mfd3POQ",
        }
        self._headers = {
            "Content-Type": "application/json",
            "xi-api-key": config.api_key,
        }
        self._voices = default_voices.copy()
        if config.voice_id in voice_options:
            config.voice_id = voice_options[config.voice_id]
        self._use_custom_voice(config.voice_id, 0)

    def _use_custom_voice(self, voice, voice_index) -> None:
        """Use a custom voice if provided and not a placeholder

        Args:
            voice (str): The voice ID
            voice_index (int): The voice index

        Returns:
            None: None
        """
        # Placeholder values that should be treated as empty
        if voice and voice not in PLACEHOLDERS:
            self._voices[voice_index] = voice

    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """Speak text using elevenlabs.io's API

        Args:
            text (str): The text to speak
            voice_index (int, optional): The voice to use. Defaults to 0.

        Returns:
            bool: True if the request was successful, False otherwise
        """
        tts_url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{self._voices[voice_index]}"
        )
        response = requests.post(tts_url, headers=self._headers, json={"text": text})

        if response.status_code == 200:
            with open("speech.mpeg", "wb") as f:
                f.write(response.content)
            playsound("speech.mpeg", True)
            os.remove("speech.mpeg")
            return True
        else:
            logger.warning("Request failed with status code:", response.status_code)
            logger.info("Response content:", response.content)
            return False
