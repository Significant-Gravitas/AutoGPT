""" GTTS Voice. """
import gtts

from autogpt.speech.playback import play_audio
from autogpt.speech.base import VoiceBase


class GTTSVoice(VoiceBase):
    """GTTS Voice."""

    def _setup(self) -> None:
        pass

    def _speech(self, text: str, _: int = 0) -> bool:
        """Play the given text."""
        tts = gtts.gTTS(text)
        audio_data = b''.join(tts.stream())
        play_audio(audio_data)
        return True
