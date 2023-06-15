"""Base class for all voice classes."""
import abc
from threading import Lock

from autogpt.singleton import AbstractSingleton


class VoiceBase(AbstractSingleton):
    """
    Base class for all voice classes.
    """

    def __init__(self):
        """
        Initialize the voice class.
        """
        self._url = None
        self._headers = None
        self._api_key = None
        self._voices = []
        self._mutex = Lock()
        self._setup()

    def say(self, text: str, voice_index: int = 0) -> bool:
        """
        Say the given text.

        Args:
            text (str): The text to say.
            voice_index (int): The index of the voice to use.
        """
        with self._mutex:
            return self._speech(text, voice_index)

    @abc.abstractmethod
    def _setup(self) -> None:
        """
        Setup the voices, API key, etc.
        """

    @abc.abstractmethod
    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """
        Play the given text.

        Args:
            text (str): The text to play.
        """
