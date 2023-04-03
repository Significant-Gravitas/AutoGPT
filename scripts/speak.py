import os
from playsound import playsound
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

voices = ["ErXwobaYiN019PkySvjV", "EXAVITQu4vr4xnSDxMaL"]
tts_headers = {
    "Content-Type": "application/json",
    "xi-api-key": os.getenv("ELEVENLABS_API_KEY")
}


def say_text(text, voice_index=0):
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voices[voice_index]}"
    formatted_message = {"text": text}
    response = requests.post(tts_url, headers=tts_headers, json=formatted_message)

    if response.status_code == 200:
        audio_file = Path("speech.mpeg")
        with audio_file.open(mode="wb") as f:
            f.write(response.content)

        playsound(str(audio_file))
        audio_file.unlink()
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(f"Response content: {response.content}")
