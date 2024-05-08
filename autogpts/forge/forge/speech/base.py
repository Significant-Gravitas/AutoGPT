"""Base class for all voice classes."""
from __future__ import annotations

import abc
import re
from threading import Lock


class VoiceBase:
    """
    Base class for all voice classes.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the voice class.
        """
        self._url = None
        self._headers = None
        self._api_key = None
        self._voices = []
        self._mutex = Lock()
        self._setup(*args, **kwargs)

    def say(self, text: str, voice_index: int = 0) -> bool:
        """
        Say the given text.

        Args:
            text (str): The text to say.
            voice_index (int): The index of the voice to use.
        """
        text = re.sub(
            r"\b(?:https?://[-\w_.]+/?\w[-\w_.]*\.(?:[-\w_.]+/?\w[-\w_.]*\.)?[a-z]+(?:/[-\w_.%]+)*\b(?!\.))",  # noqa: E501
            "",
            text,
        )
        with self._mutex:
            return self._speech(text, voice_index)

    @abc.abstractmethod
    def _setup(self, *args, **kwargs) -> None:
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
