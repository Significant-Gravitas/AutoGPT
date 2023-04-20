from io import BytesIO
import os
from typing import Union

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame


def play_audio(audio: Union[bytes, BytesIO]):
    """
    Play audio using pygame.

    This function takes in audio data in the form of bytes or a BytesIO object
    and plays it using the pygame mixer.

    :param audio: Audio data in the form of bytes or a BytesIO object.
    :type audio: Union[bytes, BytesIO]
    """
    if not isinstance(audio, (bytes, BytesIO)):
        return
    if isinstance(audio, bytes):
        audio = BytesIO(audio)

    # play audio
    pygame.mixer.init()
    pygame.mixer.music.load(audio)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(10)
