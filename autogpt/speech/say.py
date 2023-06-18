""" Text to speech module """
import threading
from threading import Semaphore

from autogpt.config.config import Config
from autogpt.speech.base import VoiceBase
from autogpt.speech.eleven_labs import ElevenLabsSpeech
from autogpt.speech.gtts import GTTSVoice
from autogpt.speech.macos_tts import MacOSTTS
from autogpt.speech.stream_elements_speech import StreamElementsSpeech

_QUEUE_SEMAPHORE = Semaphore(
    1
)  # The amount of sounds to queue before blocking the main thread


def say_text(text: str, description: str = "", voice_index: int = 0) -> None:
    """Speak the given text using the given voice index"""
    cfg = Config()
    default_voice_engine, voice_engine = _get_voice_engine(cfg)
    plain_speech = _get_plain_speech(text, description)

    def speak() -> None:
        success = voice_engine.say(plain_speech, voice_index)
        if not success:
            default_voice_engine.say(plain_speech)

        _QUEUE_SEMAPHORE.release()

    _QUEUE_SEMAPHORE.acquire(True)
    thread = threading.Thread(target=speak)
    thread.start()


def _get_voice_engine(config: Config) -> tuple[VoiceBase, VoiceBase]:
    """Get the voice engine to use for the given configuration"""
    tts_provider = config.text_to_speech_provider
    if tts_provider == "elevenlabs":
        voice_engine = ElevenLabsSpeech()
    elif tts_provider == "macos":
        voice_engine = MacOSTTS()
    elif tts_provider == "streamelements":
        voice_engine = StreamElementsSpeech()
    else:
        voice_engine = GTTSVoice()

    return GTTSVoice(), voice_engine


def _get_plain_speech(text: str, description: str = ""):
    # When executing agent commands, the "text" parameter is the command name, e.g. "read_file".
    # Many commands have descriptions that are suitable for natural-sounding TTS, but not all.
    # If you think a command's description could be better phrased for TTS, add an override below.

    preamble = "I'll need permission to"

    try:
        if text.lower() == "error:":
            return "An unexpected error occurred"
        elif text == "do_nothing" or text.lower() == "none":
            return "I'll need some time to think about this"

        # commands/execute_code.py
        elif text == "execute_python_file":
            description = "execute a Python file"
        elif text == "execute_shell":
            description = "execute a non-interactive shell command"
        elif text == "execute_shell_popen":
            description = "execute a non-interactive shell command with P-open"

        # commands/file_operations.py
        elif text == "download_file":
            description = "download files"
        elif text == "read_file":
            description = "read files"
        elif text == "write_to_file":
            description = "write to files"
        elif text == "append_to_file":
            description = "append to files"
        elif text == "delete_file":
            description = "delete files"
        elif text == "list_file":
            description = "list files in a directory"

        # commands/git_operations.py
        elif text == "clone_repository":
            description = "clone Git repositories"

        # commands/google_search.py
        elif text == "google":
            description = "search Google"

        # commands/image_gen.py
        elif text == "generate_image":
            description = "generate images"

        # commands/task_statuses.py
        elif text == "task_complete":
            description = "shut down"

        # commands/web_selenium.py
        elif text == "browse_website":
            description = "browse websites with Selenium"

        if description:
            return f"{preamble} {description}"
        else:
            return text

    # All errors, return "Error: + error message"
    except Exception as e:
        return f"Error: {str(e)}"
