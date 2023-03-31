import os
from playsound import playsound
import requests
import keys

voice_id = "ErXwobaYiN019PkySvjV"
tts_url = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}".format(voice_id=voice_id)

tts_headers = {
    "Content-Type": "application/json",
    "xi-api-key": keys.ELEVENLABS_API_KEY
}


def say_text(text):
    formatted_message = {"text": text}
    response = requests.post(
        tts_url, headers=tts_headers, json=formatted_message)

    if response.status_code == 200:
        with open("speech.mpeg", "wb") as f:
            f.write(response.content)
        playsound("speech.mpeg")
        # Delete audio file
        os.remove("speech.mpeg")
    else:
        print("Request failed with status code:", response.status_code)
        print("Response content:", response.content)

