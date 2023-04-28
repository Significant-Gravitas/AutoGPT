""" MacOS TTS Voice. """
import os
import subprocess

from autogpt.config import Config
from autogpt.speech.base import VoiceBase


class MacOSTTS(VoiceBase):
    """MacOS TTS Voice."""

    def _setup(self) -> None:
        """Set up the voices for MacOSTTS.

        Returns:
            None: None
        """
        cfg = Config()
        available_voices = self.get_voice_names()
        default_voices = ["Alex", "Victoria"]
        custom_voices = [cfg.macos_voice_1, cfg.macos_voice_2]

        self._voices = [
            voice for voice in custom_voices if voice in available_voices
        ] + default_voices
        self._voices = self._voices[:2]

    def get_voice_names(self):
        """Get the available voice names in the system.

        Returns:
            list[str]: A list of available voice names.
        """
        cmd = ["say", "-v", "?"]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        lines = result.stdout.split("\n")

        voice_names = []
        for line in lines:
            if line.strip():
                name = line.split()[0]
                voice_names.append(name)

        return voice_names

    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """Speak the given text using the specified voice in the macOS TTS system.

        Args:
            text (str): The text to be spoken.
            voice_index (int, optional): The index of the voice to use from the _voices list. Defaults to 0.

        Returns:
            bool: True if the text was spoken successfully, False otherwise.
        """
        if voice_index == 0:
            os.system(f'say "{text}"')
        else:
            voice = self._voices[voice_index % len(self._voices)]
            os.system(f'say -v "{voice}" "{text}"')

        return True
