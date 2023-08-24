""" GTTS Voice. """
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import gtts
from playsound import playsound

if TYPE_CHECKING:
    from autogpt.config import Config

from autogpt.speech.base import VoiceBase


class GTTSVoice(VoiceBase):
    """GTTS Voice."""

    def _setup(self, config: Config) -> None:
        pass

    def _speech(self, text: str, _: int = 0) -> bool:
        """Play the given text."""
        tts = gtts.gTTS(text)
        tts.save("speech.mp3")
        playsound("speech.mp3", True)
        os.remove("speech.mp3")
        return True
