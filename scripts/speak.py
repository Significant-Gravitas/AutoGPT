import os
from playsound import playsound
import requests
from config import Config
cfg = Config()

# Remove the import of gtts
# import gtts

# Change voices to macOS voice identifiers
voices = ["com.apple.speech.synthesis.voice.siri_female", "com.apple.speech.synthesis.voice.siri_male"]

tts_headers = {
    "Content-Type": "application/json",
    "xi-api-key": cfg.elevenlabs_api_key
}

def eleven_labs_speech(text, voice_index=0):
    tts_url = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}".format(
        voice_id=voices[voice_index])
    formatted_message = {"text": text}
    response = requests.post(
        tts_url, headers=tts_headers, json=formatted_message)

    if response.status_code == 200:
        with open("speech.mpeg", "wb") as f:
            f.write(response.content)
        playsound("speech.mpeg")
        os.remove("speech.mpeg")
        return True
    else:
        print("Request failed with status code:", response.status_code)
        print("Response content:", response.content)
        return False

# Use macOS built-in TTS instead of gtts
def macos_tts_speech(text, voice_index=1):
    os.system(f'say "{text}"')

def say_text(text, voice_index=0):
    if not cfg.elevenlabs_api_key:
        macos_tts_speech(text, voice_index)
    else:
        success = eleven_labs_speech(text, voice_index)
        if not success:
            macos_tts_speech(text, voice_index)
