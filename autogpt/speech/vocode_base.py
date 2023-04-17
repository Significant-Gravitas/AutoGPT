import abc
from autogpt.speech.base import VoiceBase
from vocode.turn_based.synthesizer.base_synthesizer import BaseSynthesizer
from vocode.turn_based.output_device.speaker_output import SpeakerOutput


class VocodeVoiceBase(VoiceBase):
    def _setup(self) -> None:
        self.output_device = SpeakerOutput.from_default_device()
        self.synthesizer = self._create_synthesizer()

    @abc.abstractmethod
    def _create_synthesizer(self) -> BaseSynthesizer:
        pass

    def _speech(self, text: str, voice_index: int = 0) -> bool:
        try:
            audio_segment = self.synthesizer.synthesize(text)
        except Exception as e:
            print("Request failed")
            print("Response content:", str(e))
            return False
        self.output_device.send_audio(audio_segment)
        return True
