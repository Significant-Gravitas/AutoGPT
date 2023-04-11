import os
import requests
from config import Config
cfg = Config()
import gtts
import threading
from threading import Lock, Semaphore
from pydub import AudioSegment
from pydub.playback import play
import os
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play

load_dotenv()
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

# TODO: Nicer names for these ids
voices = ["ErXwobaYiN019PkySvjV", "TxGEqnHWrfWFTfGW9XjX"]

tts_headers = {
    "Content-Type": "application/json",
    "xi-api-key": cfg.elevenlabs_api_key
}

mutex_lock = Lock() # Ensure only one sound is played at a time
queue_semaphore = Semaphore(1) # The amount of sounds to queue before blocking the main thread

from pydub import AudioSegment
from pydub.playback import play

def eleven_labs_speech(text, voice_index):
    url = f'https://api.elevenlabs.io/v1/synthesize?voice={voice_index}'
    headers = {
        'Authorization': f'Bearer {ELEVENLABS_API_KEY}',
    }
    data = {
        'text': text,
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        with open('speech.mpeg', 'wb') as f:
            f.write(response.content)

        # Convert MPEG to WAV using FFmpeg
        os.system(f'ffmpeg -i speech.mpeg -acodec pcm_s16le -ac 1 -ar 16000 speech.wav')

        sound = AudioSegment.from_file("speech.wav", format="wav")
        sound.export("output.wav", format="wav")
        return True
    else:
        print(f'Eleven Labs API request failed with status code: {response.status_code}')
        return False

def gtts_speech(text):
    tts = gtts.gTTS(text)
    tts.save("speech.mp3")
    sound = AudioSegment.from_mp3("speech.mp3")
    play(sound)
    os.remove("speech.mp3")

def macos_tts_speech(text):
    os.system(f'say "{text}"')

def say_text(text, voice_index=0):

    def speak():
        if not cfg.elevenlabs_api_key:
            if cfg.use_mac_os_tts == 'True':
                macos_tts_speech(text)
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
