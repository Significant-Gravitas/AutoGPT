# -*- coding: utf-8 -*-
from .version import __version__  # noqa: F401
from .tts import gTTS, gTTSError

__all__ = ["gTTS", "gTTSError"]
