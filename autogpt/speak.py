import os

import requests
from playsound import playsound

from autogpt.config import Config

import threading
from threading import Lock, Semaphore

import gtts

cfg = Config()

# Default voice IDs
default_voices = ["ErXwobaYiN019PkySvjV", "EXAVITQu4vr4xnSDxMaL"]

# Retrieve custom voice IDs from the Config class
custom_voice_1 = cfg.elevenlabs_voice_1_id
custom_voice_2 = cfg.elevenlabs_voice_2_id

# Placeholder values that should be treated as empty
placeholders = {"your-voice-id"}

# Use custom voice IDs if provided and not placeholders, otherwise use default voice IDs
voices = [
    custom_voice_1
    if custom_voice_1 and custom_voice_1 not in placeholders
    else default_voices[0],
    custom_voice_2
    if custom_voice_2 and custom_voice_2 not in placeholders
    else default_voices[1],
]

tts_headers = {"Content-Type": "application/json", "xi-api-key": cfg.elevenlabs_api_key}

mutex_lock = Lock()  # Ensure only one sound is played at a time
queue_semaphore = Semaphore(
    1
)  # The amount of sounds to queue before blocking the main thread


def eleven_labs_speech(text, voice_index=0):
    """Speak text using elevenlabs.io's API"""
    tts_url = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}".format(
        voice_id=voices[voice_index]
    )
    formatted_message = {"text": text}
    response = requests.post(tts_url, headers=tts_headers, json=formatted_message)

    if response.status_code == 200:
        with mutex_lock:
            with open("speech.mpeg", "wb") as f:
                f.write(response.content)
            playsound("speech.mpeg", True)
            os.remove("speech.mpeg")
        return True
    else:
        print("Request failed with status code:", response.status_code)
        print("Response content:", response.content)
        return False


def brian_speech(text):
    """Speak text using Brian with the streamelements API"""
    tts_url = f"https://api.streamelements.com/kappa/v2/speech?voice=Brian&text={text}"
    response = requests.get(tts_url)

    if response.status_code == 200:
        with mutex_lock:
            with open("speech.mp3", "wb") as f:
                f.write(response.content)
            playsound("speech.mp3")
            os.remove("speech.mp3")
        return True
    else:
        print("Request failed with status code:", response.status_code)
        print("Response content:", response.content)
        return False


def gtts_speech(text):
    tts = gtts.gTTS(text)
    with mutex_lock:
        tts.save("speech.mp3")
        playsound("speech.mp3", True)
        os.remove("speech.mp3")


def macos_tts_speech(text, voice_index=0):
    if voice_index == 0:
        os.system(f'say "{text}"')
    else:
        if voice_index == 1:
            os.system(f'say -v "Ava (Premium)" "{text}"')
        else:
            os.system(f'say -v Samantha "{text}"')


def say_text(text, voice_index=0):
    def speak():
        if not cfg.elevenlabs_api_key:
            if cfg.use_mac_os_tts == "True":
                macos_tts_speech(text)
            elif cfg.use_brian_tts == "True":
                success = brian_speech(text)
                if not success:
                    gtts_speech(text)
            else:
                gtts_speech(text)
        else:
            success = eleven_labs_speech(text, voice_index)
            if not success:
                gtts_speech(text)

        queue_semaphore.release()

    queue_semaphore.acquire(True)
    thread = threading.Thread(target=speak)
    thread.start()
