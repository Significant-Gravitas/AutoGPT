import logging
import os

import requests
from playsound import playsound

from autogpt.speech.base import VoiceBase


class StreamElementsSpeech(VoiceBase):
    """StreamElements speech module for autogpt"""

    def _setup(self) -> None:
        """Setup the voices, API key, etc."""
        pass

    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """Speak text using the StreamElements API

        Args:
            text (str): The text to speak

        Returns:
            bool: True if the request was successful, False otherwise
        """
        voice1 = os.getenv("SE_VOICE_1") or "Emma"
        voice2 = os.getenv("SE_VOICE_2") or "Brian"
        voice3 = os.getenv("SE_VOICE_3") or "Amy"

        if voice_index == 0:
            tts_url = (
                f"https://api.streamelements.com/kappa/v2/speech?voice={voice1}&text={text}"
            )
        elif voice_index == 1:
            tts_url = (
                f"https://api.streamelements.com/kappa/v2/speech?voice={voice2}&text={text}"
            )
        else:
            tts_url = (
                f"https://api.streamelements.com/kappa/v2/speech?voice={voice3}&text={text}"
            )

        response = requests.get(tts_url)

        if response.status_code == 200:
            with open("speech.mp3", "wb") as f:
                f.write(response.content)
            playsound("speech.mp3")
            os.remove("speech.mp3")
            return True
        else:
            logging.error(
                "Request failed with status code: %s, response content: %s",
                response.status_code,
                response.content,
            )
            return False
