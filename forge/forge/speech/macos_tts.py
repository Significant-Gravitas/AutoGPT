""" MacOS TTS Voice. """
from __future__ import annotations

import subprocess

from .base import VoiceBase


class MacOSTTS(VoiceBase):
    """MacOS TTS Voice."""

    def _setup(self) -> None:
        pass

    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """Play the given text."""
        if voice_index == 0:
            subprocess.run(["say", text], shell=False)
        elif voice_index == 1:
            subprocess.run(["say", "-v", "Ava (Premium)", text], shell=False)
        else:
            subprocess.run(["say", "-v", "Samantha", text], shell=False)
        return True
