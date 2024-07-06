""" GTTS Voice. """
from __future__ import annotations

import os

import gtts
from playsound import playsound

from .base import VoiceBase


class GTTSVoice(VoiceBase):
    """GTTS Voice."""

    def _setup(self) -> None:
        pass

    def _speech(self, text: str, voice_id: int = 0) -> bool:
        """Play the given text."""
        tts = gtts.gTTS(text)
        tts.save("speech.mp3")
        playsound("speech.mp3", True)
        os.remove("speech.mp3")
        return True
