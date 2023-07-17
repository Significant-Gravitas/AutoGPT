""" MacOS TTS Voice. """
import os
import subprocess

from autogpt.config import Config
from autogpt.speech.base import VoiceBase


class MacOSTTS(VoiceBase):
    """MacOS TTS Voice."""

    def _setup(self, config: Config) -> None:
        pass

    def _speech(self, text: str, voice: int = 0) -> bool:
        voice1 = os.getenv("MAC_OS_VOICE_1", "Allison")
        voice2 = os.getenv("MAC_OS_VOICE_2", "Ava")
        voice3 = os.getenv("MAC_OS_VOICE_3", "Samantha")
        """Play the given text."""
        if voice == 2:
            os.system(f'say -v {voice3} "{text}"')
        elif voice == 1:
            os.system(f'say -v {voice2} "{text}"')
        else:
            os.system(f'say -v {voice1} "{text}"')
        return True
