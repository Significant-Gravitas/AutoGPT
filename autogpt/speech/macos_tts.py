""" MacOS TTS Voice. """
import os

from autogpt.speech.base import VoiceBase


class MacOSTTS(VoiceBase):
    """MacOS TTS Voice."""

    def _setup(self) -> None:
        pass

    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """Play the given text."""
        if voice_index == 0:
            os.system(f'say "{text}"')
        elif voice_index == 1:
            os.system(f'say -v "Ava (Premium)" "{text}"')
        else:
            os.system(f'say -v Samantha "{text}"')
        return True
