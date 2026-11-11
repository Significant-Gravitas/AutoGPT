"""Otto's voice coming back through the person's microphone.

A speaker's sound reaches a microphone, and a browser's echo canceller does
not always take it out (a Bluetooth or monitor speaker's delay is more than
it can follow). The live model then hears Otto's sentences as the person's
and cuts him off with them. So the relay stops trusting the canceller and
decides, per call, whether the microphone carries Otto:

- Every call starts closed (``probing``): during Otto's first turn the model
  and the recording get silence in place of the microphone. From
  ``Timing.echo_probe_skip_s`` after his first audio (a working canceller
  leaks until it settles) until ``Timing.echo_tail_s`` after the turn played
  out, the share of microphone frames the voice detector calls speech is
  counted (:class:`Probe`). At ``HALF_DUPLEX_SHARE`` or more the call is
  ``half``: the microphone is silence while Otto is audible, until
  ``echo_tail_s`` after his audio. Otherwise it is ``full``.
- On a ``full`` call echo can still start (headphones unplugged, the volume
  up). A caption of the person's of two or more words found as a run in
  Otto's current or previous turn is his own voice (:func:`is_echo`): the
  call goes ``half`` for good, the caption is dropped, and the model is told
  once (``ECHO_NOTE``, sent as a server note).
- The transcript of record cannot take Otto's words for the person's: what
  the model was given is what was recorded, and the post-call job drops a
  clip whose words are at least ``ECHO_CLIP_SHARE`` Otto's own from the
  turns either side (:func:`echoed_clip`).

Nothing here uses a LiveKit Cloud feature, so hosted and self-hosted calls
behave the same.
"""

import re
from typing import Literal

from pydantic import BaseModel

Duplex = Literal["probing", "full", "half"]
EchoBy = Literal["probe", "captions"]

# The share of the probe's frames heard as speech from which the call is
# half-duplex: a clean microphone is near none, one carrying Otto over half.
HALF_DUPLEX_SHARE = 0.20
# A caption this many words long, or longer, found in Otto's words is his.
MIN_ECHO_WORDS = 2
# A clip of the person's whose words are this much Otto's own is his.
ECHO_CLIP_SHARE = 0.80

ECHO_NOTE = (
    "What you just heard was your own voice coming back through the user's "
    "speakers, not the user. Carry on from your last question."
)

_WORDS = re.compile(r"[\w']+")


class Probe(BaseModel):
    """The microphone while Otto's first turn plays: frames counted, and those
    the voice detector called speech."""

    frames: int = 0
    heard: int = 0

    def count(self, *, heard: bool) -> None:
        self.frames += 1
        if heard:
            self.heard += 1

    @property
    def share(self) -> float:
        """The share of frames heard as speech; none for no frames."""
        return self.heard / self.frames if self.frames else 0.0

    @property
    def carries_otto(self) -> bool:
        return self.share >= HALF_DUPLEX_SHARE


def words_of(text: str) -> list[str]:
    """*text*'s words, lowercased, without punctuation."""
    return _WORDS.findall(text.lower())


def is_echo(caption: str, *otto_turns: str) -> bool:
    """Whether *caption*, the person's words so far in a turn of theirs, is
    ``MIN_ECHO_WORDS`` or more words found in order, as one run, in one of
    *otto_turns*."""
    said = words_of(caption)
    if len(said) < MIN_ECHO_WORDS:
        return False
    return any(_holds_run(words_of(turn), said) for turn in otto_turns)


def echoed_clip(text: str, *otto_turns: str) -> bool:
    """Whether a transcribed clip of the person's is at least
    ``ECHO_CLIP_SHARE`` words Otto said in *otto_turns*, the turns either side
    of it."""
    said = words_of(text)
    if not said:
        return False
    otto = {word for turn in otto_turns for word in words_of(turn)}
    return sum(word in otto for word in said) / len(said) >= ECHO_CLIP_SHARE


def _holds_run(words: list[str], run: list[str]) -> bool:
    size = len(run)
    return any(words[i : i + size] == run for i in range(len(words) - size + 1))
