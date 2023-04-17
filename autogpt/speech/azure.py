from vocode.turn_based.synthesizer.azure_synthesizer import AzureSynthesizer
from autogpt.config import Config
from autogpt.speech.vocode_base import VocodeVoiceBase


class AzureSpeech(VocodeVoiceBase):
    """Azure speech module for autogpt: https://azure.microsoft.com/en-us/products/cognitive-services/text-to-speech/"""

    def _create_synthesizer(self) -> AzureSynthesizer:
        cfg = Config()
        return AzureSynthesizer(
            api_key=cfg.azure_speech_key,
            region=cfg.azure_speech_region,
        )
