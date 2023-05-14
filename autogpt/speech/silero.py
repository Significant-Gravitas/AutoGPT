"""ElevenLabs speech module"""
import os

# V3
import os
import torch
from playsound import playsound

from autogpt.config import Config
from autogpt.speech.base import VoiceBase

PLACEHOLDERS = {"your-voice-id"}


class Silero(VoiceBase):
    """ElevenLabs speech class"""

    def _setup(self) -> None:
        """Set up the voices, API key, etc.

        Returns:
            None: None
        """

        device = torch.device("cpu")
        torch.set_num_threads(4)
        local_file = "model.pt"

        if not os.path.isfile(local_file):
            torch.hub.download_url_to_file(
                "https://models.silero.ai/models/tts/en/v3_en.pt", local_file
            )

        self.model = torch.package.PackageImporter(local_file).load_pickle(
            "tts_models", "model"
        )
        self.model.to(device)

    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """Speak text using elevenlabs.io's API

        Args:
            text (str): The text to speak
            voice_index (int, optional): The voice to use. Defaults to 0.

        Returns:
            bool: True if the request was successful, False otherwise
        """
        sample_rate = 48000
        speaker = Config().silero_tts_voice

        put_accent = True
        put_yo = True

        audio = self.model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=sample_rate,
            put_accent=put_accent,
            put_yo=put_yo,
        )

        playsound(audio, True)
        os.remove(audio)

        return True
