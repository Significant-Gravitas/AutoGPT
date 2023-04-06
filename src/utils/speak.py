import gtts
import os
from playsound import playsound
import requests
from colorama import Fore, Style
from config import Config
import utils.console_log as console_log
cfg = Config()

# TODO: Nicer names for these ids
voices = ["ErXwobaYiN019PkySvjV", "EXAVITQu4vr4xnSDxMaL"]

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

    if response.status_code != 200:
        return False, f"Error {response.status_code}: {response.content}"
    
    with open("speech.mpeg", "wb") as f:
        f.write(response.content)
    playsound("speech.mpeg")
    os.remove("speech.mpeg")
    return True, None

def gtts_speech(text):
    tts = gtts.gTTS(text)
    tts.save("speech.mp3")
    playsound("speech.mp3")
    os.remove("speech.mp3")

def say_text(text, voice_index=0):
    if not cfg.elevenlabs_api_key or cfg.elevenlabs_api_key == "your-elevenlabs-api-key":
        gtts_speech(text)
    else:
        success, error = eleven_labs_speech(text, voice_index)
        if not success:
            gtts_speech(text)
            console_log.print_error(f"ElevenLabs has run into an error ({error})")