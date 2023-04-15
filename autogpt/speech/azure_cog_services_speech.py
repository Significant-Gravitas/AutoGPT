"""Azure Cog Services speech module"""
import os
import requests

from playsound import playsound
import azure.cognitiveservices.speech as speechsdk

from autogpt.config import Config
from autogpt.speech.base import VoiceBase

VOICE_PLACEHOLDERS = { "your-voice-id-1","your-voice-id-2" }
VOICE_STYLE_PLACEHOLDERS = { "your-voice-1-style", "your-voice-2-style" }

class AzureCogServicesSpeech(VoiceBase):
    """Azure Cog Services speech class"""

    def _setup(self) -> None:
        """Setup the voices, API key, etc.

        Returns:
            None: None
        """
        cfg = Config()
        self._region = cfg.azure_cs_tts_region
        default_voices = [ "en-US-JennyMultilingualNeural", "en-US-DavisNeural" ]
        default_style = [ "Default", "Default" ]
        self._headers = {
            "Ocp-Apim-Subscription-Key": cfg.azure_cs_tts_apikey,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
        }
        self._voices = default_voices.copy()
        self._voicesStyle = default_style.copy()
        self._use_custom_voice(cfg.azure_cs_tts_voice_1_id, cfg.azure_cs_tts_voice_1_style, 0)
        self._use_custom_voice(cfg.azure_cs_tts_voice_2_id, cfg.azure_cs_tts_voice_2_style, 1)

    def _use_custom_voice(self, voice, style, voice_index) -> None:
        """Use a custom voice if provided and not a placeholder

        Args:
            voice (str): The voice ID
            voice_index (int): The voice index

        Returns:
            None: None
        """
        # Placeholder values that should be treated as empty
        if voice and voice not in VOICE_PLACEHOLDERS:
            self._voices[voice_index] = voice

        if style and style not in VOICE_STYLE_PLACEHOLDERS:
            self._voicesStyle[voice_index] = style

    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """Speak text using Azure Cog Services

        Args:
            text (str): The text to speak
            voice_index (int, optional): The voice to use. Defaults to 0.

        Returns:
            bool: True if the request was successful, False otherwise
        """
        speech_config = speechsdk.SpeechConfig(subscription=self._headers["Ocp-Apim-Subscription-Key"], region=self._region)
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat["Audio24Khz48KBitRateMonoMp3"])

        speech_config.speech_synthesis_voice_name = self._voices[voice_index]
        ssml_text = "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><voice name='" + self._voices[voice_index] + "' style='" + self._voicesStyle[voice_index] + "'>" + text + "</voice></speak>"

        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        result = synthesizer.speak_ssml_async(ssml_text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data = result.audio_data
            with open("speech.mpeg", "wb") as f:
                f.write(audio_data)
            playsound("speech.mpeg", True)
            os.remove("speech.mpeg")
            return True
        else:
            print("Speech synthesis failed: {}".format(result.reason))
            return False
