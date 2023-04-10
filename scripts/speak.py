from os import remove
from playsound import playsound
from requests import post
from config import Config
cfg = Config()
from gtts import gTTS


# TODO: Nicer names for these ids
voices = ["ErXwobaYiN019PkySvjV", "EXAVITQu4vr4xnSDxMaL"]

tts_headers = {
    "Content-Type": "application/json",
    "xi-api-key": cfg.elevenlabs_api_key
}

def eleven_labs_speech(text, voice_index=0):
    """Speak text using elevenlabs.io's API"""
    tts_url = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}".format(
        voice_id=voices[voice_index])
    formatted_message = {"text": text}
    response = post(
        tts_url, headers=tts_headers, json=formatted_message)

    if response.status_code == 200:
        with open("speech.mpeg", "wb") as f:
            f.write(response.content)
        playsound("speech.mpeg")
        remove("speech.mpeg")
        return True
    else:
        print("Request failed with status code:", response.status_code)
        print("Response content:", response.content)
        return False

def gtts_speech(text):
    tts = gTTS(text)
    tts.save("speech.mp3")
    playsound("speech.mp3")
    remove("speech.mp3")

def say_text(text, voice_index=0):
    if not cfg.elevenlabs_api_key:
        gtts_speech(text)
    else:
        success = eleven_labs_speech(text, voice_index)
        if not success:
            gtts_speech(text)

