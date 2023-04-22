""" Brian speech module for autogpt """
from vocode.turn_based.output_device.speaker_output import SpeakerOutput
from vocode.turn_based.synthesizer.stream_elements_synthesizer import (
    StreamElementsSynthesizer,
)

from autogpt.speech.vocode_base import VocodeVoiceBase


class BrianSpeech(VocodeVoiceBase):
    """Speak text using Brian with the streamelements API"""

    def _create_synthesizer(self) -> StreamElementsSynthesizer:
        return StreamElementsSynthesizer(voice="Brian")