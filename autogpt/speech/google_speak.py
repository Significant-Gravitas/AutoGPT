from google.cloud import texttospeech
import io, os
from autogpt.speech.base import VoiceBase
from autogpt.config import Config
from playsound import playsound

PATH = os.path.dirname(os.path.abspath(__file__))

PLACEHOLDERS = {"your-voice-id"}

class GoogleSpeech(VoiceBase):
    """Google speech class"""
    # Instantiates a client

    def _setup(self) -> None:
        """Set up the voices, API key, etc.

        Returns:
            None: None
        """
        cfg = Config()
        self.client = texttospeech.TextToSpeechClient.from_service_account_json(cfg.google_speak_key_path)
        default_voices = ["a", "b"]
        voice_options = {
            "a": "en-US-Neural2-J",
            "b": "en-US-Neural2-A",
            "c": "en-US-Neural2-B",
            "d": "en-US-Neural2-C",
            "e": "en-US-Neural2-D",
            "f": "en-US-Neural2-E",
            "g": "en-US-Neural2-F",
            "h": "en-US-Neural2-G",
            "i": "en-US-Neural2-H"
        }
        self._voices = default_voices.copy()
        self._voice_options = voice_options.copy()
        if cfg.google_voice_1_id in voice_options:
            cfg.google_voice_1_id = voice_options[cfg.google_voice_1_id]
        if cfg.google_voice_2_id in voice_options:
            cfg.google_voice_2_id = voice_options[cfg.google_voice_2_id]
        
        self._use_custom_voice(cfg.google_voice_1_id, 0)
        self._use_custom_voice(cfg.google_voice_2_id, 1)
        
    def _use_custom_voice(self, voice, voice_index) -> None:
        """Use a custom voice if provided and not a placeholder

        Args:
            voice (str): The voice ID
            voice_index (int): The voice index

        Returns:
            None: None
        """
        # Placeholder values that should be treated as empty
        if voice and voice not in PLACEHOLDERS:
            self._voices[voice_index] = voice


    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """
        Speak text using Google's API
        Args:
            text (str): The text to speak
            voice_name (str): The voice name default: "en-US-Neural2-J"
        Returns:
            bool: True if successful, False otherwise
        """
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Build the voice request, select the language code ("en-US") and the ssml
        # voice gender ("neutral")


        # voice = texttospeech.VoiceSelectionParams(
        #     language_code="en-US",
        #     ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        # )

        voice_name = self._voice_options[self._voices[voice_index]]
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", name=voice_name
        )

        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        try:
            # Perform the text-to-speech request on the text input with the selected
            # voice parameters and audio file type
            response = self.client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            with open(PATH+"/output.mp3", "wb") as f:
                f.write(response.audio_content)
            playsound(PATH+"/output.mp3", True)
            os.remove(PATH+"/output.mp3")
            return True
        except Exception as e:
            print("Google tts request failed: "+ str(e))
            return False
