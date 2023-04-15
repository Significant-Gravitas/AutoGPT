"""ElevenLabs speech module"""
import os
from playsound import playsound

import requests

from autogpt.config import Config
from autogpt.speech.base import VoiceBase

PLACEHOLDERS = {"your-voice-id"}


class ElevenLabsSpeech(VoiceBase):
    """ElevenLabs speech class"""

    def _setup(self) -> None:
        """Setup the voices, API key, etc.

        Returns:
            None: None
        """

        cfg = Config()
        default_voices = ["ErXwobaYiN019PkySvjV", "EXAVITQu4vr4xnSDxMaL"]
        self._headers = {
            "Content-Type": "application/json",
            "xi-api-key": cfg.elevenlabs_api_key,
        }
        self._voices = default_voices.copy()
        self._use_custom_voice(cfg.elevenlabs_voice_1_id, 0)
        self._use_custom_voice(cfg.elevenlabs_voice_2_id, 1)

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
            print("Request failed with status code:", response.status_code)
            print("Response content:", response.content)
            return False
