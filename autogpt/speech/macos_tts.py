""" MacOS TTS Voice. """
import os

from autogpt.speech.base import VoiceBase


class MacOSTTS(VoiceBase):
    """MacOS TTS Voice."""

    def _setup(self) -> None:
        pass

    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """Play the given text."""
        voice1 = os.getenv("MAC_OS_VOICE_1") or "Allison"
        voice2 = os.getenv("MAC_OS_VOICE_2") or "Ava"
        voice3 = os.getenv("MAC_OS_VOICE_3") or "Samantha"
        if voice_index == 0:
            os.system(f'say -v {voice1} "{text}"')
        elif voice_index == 1:
            os.system(f'say -v {voice2} "{text}"')
        else:
            os.system(f'say -v {voice3} "{text}"')
        return True