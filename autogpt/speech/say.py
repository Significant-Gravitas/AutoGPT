""" Text to speech module """
import threading
from threading import Semaphore

from autogpt.config import Config
from autogpt.speech.brian import BrianSpeech
from autogpt.speech.eleven_labs import ElevenLabsSpeech
from autogpt.speech.gtts import GTTSVoice
from autogpt.speech.macos_tts import MacOSTTS

CFG = Config()
DEFAULT_VOICE_ENGINE = GTTSVoice()
VOICE_ENGINE = None
if CFG.elevenlabs_api_key:
    VOICE_ENGINE = ElevenLabsSpeech()
elif CFG.use_mac_os_tts == "True":
    VOICE_ENGINE = MacOSTTS()
elif CFG.use_brian_tts == "True":
    VOICE_ENGINE = BrianSpeech()
else:
    VOICE_ENGINE = GTTSVoice()


QUEUE_SEMAPHORE = Semaphore(
    1
)  # The amount of sounds to queue before blocking the main thread


def say_text(text: str, voice_index: int = 0) -> None:
    """Speak the given text using the given voice index"""

    def speak() -> None:
        success = VOICE_ENGINE.say(text, voice_index)
        if not success:
            DEFAULT_VOICE_ENGINE.say(text)

        QUEUE_SEMAPHORE.release()

    QUEUE_SEMAPHORE.acquire(True)
    thread = threading.Thread(target=speak)
    thread.start()
