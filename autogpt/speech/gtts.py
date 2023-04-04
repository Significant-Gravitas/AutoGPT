""" GTTS Voice. """

from autogpt.speech.base import VoiceBase
from autogpt.speech.vocode_base import VocodeVoiceBase
from vocode.turn_based.synthesizer.gtts_synthesizer import GTTSSynthesizer


class GTTSVoice(VocodeVoiceBase):
    """GTTS Voice."""

    def _create_synthesizer(self) -> GTTSSynthesizer:
        return GTTSSynthesizer()
