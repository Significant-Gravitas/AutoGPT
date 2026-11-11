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
  ``echo_tail_s`` after his audio. Otherwise, and when the probe heard less
  than ``PROBE_MIN_S`` of microphone to judge by, it is ``full``.
- On a ``full`` call echo can still start (headphones unplugged, the volume
  up), and what tells an echo from an answer is when it was heard: an echo
  reaches the microphone while Otto is audible, an answer after he stops.
  Speech heard while he is audible opens an overheard stretch
  (:class:`Overheard`), which keeps the person's captions until Otto's next
  turn begins. By then their turn is closed and its caption whole, and the
  stretch is judged once: it is Otto's when most of it was heard over him and
  its whole caption is a run of his words. Then the call goes ``half`` for
  good, what it made count as an answer is dropped, and the model is told
  once (``ECHO_NOTE``, sent as a server note), unless the call is already
  saying goodbye. Otherwise its caption is the person's.
- The transcript of record cannot take a caught echo for the person's words:
  the recording is silenced where it was caught (the stretch's span, on the
  call's record), before the post-call job cuts it into clips.

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
# Less microphone than this counted by the probe is too little to judge by.
PROBE_MIN_S = 1.0
# A caption this many words long, or longer, found in Otto's words may be his.
MIN_ECHO_WORDS = 2
# An overheard stretch is Otto's only with at least this much speech heard
# while he was audible, and that at least this many times what was heard
# while he was not: someone who starts in his last moment and carries on is
# mostly in the clear.
OVERHEARD_MIN_S = 0.3
OVERHEARD_OVER_CLEAR = 2.0

ECHO_NOTE = (
    "What you just heard was your own voice coming back through the user's "
    "speakers, not the user. Carry on from your last question."
)

_WORDS = re.compile(r"[\w']+")


class Probe(BaseModel):
    """The microphone while Otto's first turn plays: frames and seconds
    counted, and the frames the voice detector called speech."""

    frames: int = 0
    heard: int = 0
    seconds: float = 0.0

    def count(self, *, heard: bool, seconds: float) -> None:
        self.frames += 1
        self.seconds += seconds
        if heard:
            self.heard += 1

    @property
    def judged(self) -> bool:
        """Whether it counted enough microphone to judge by."""
        return self.seconds >= PROBE_MIN_S

    @property
    def share(self) -> float:
        """The share of frames heard as speech; none for no frames."""
        return self.heard / self.frames if self.frames else 0.0

    @property
    def carries_otto(self) -> bool:
        return self.judged and self.share >= HALF_DUPLEX_SHARE


class Overheard(BaseModel):
    """Speech heard on a full-duplex call while Otto was audible, kept until
    his next turn begins: where it starts in the recording, the seconds of
    speech heard while he was audible and while he was not, and the person's
    captions meanwhile."""

    start: int
    over_s: float = 0.0
    clear_s: float = 0.0
    text: str = ""

    def hear(self, seconds: float, *, otto_audible: bool) -> None:
        if otto_audible:
            self.over_s += seconds
        else:
            self.clear_s += seconds

    def is_otto(self, *otto_turns: str) -> bool:
        """Whether the stretch is Otto's voice: mostly heard over him, and its
        whole caption a run of his words in one of *otto_turns*."""
        return (
            self.over_s >= OVERHEARD_MIN_S
            and self.over_s >= OVERHEARD_OVER_CLEAR * self.clear_s
            and is_echo(self.text, *otto_turns)
        )


def words_of(text: str) -> list[str]:
    """*text*'s words, lowercased, without punctuation."""
    return _WORDS.findall(text.lower())


def is_echo(caption: str, *otto_turns: str) -> bool:
    """Whether *caption* is ``MIN_ECHO_WORDS`` or more words found in order,
    as one run, in one of *otto_turns*."""
    said = words_of(caption)
    if len(said) < MIN_ECHO_WORDS:
        return False
    return any(_holds_run(words_of(turn), said) for turn in otto_turns)


def silenced(pcm: bytes, spans: list[tuple[int, int]]) -> bytes:
    """*pcm*, 16-bit samples, with each of *spans* (start and end, in samples)
    made silence, clamped to its length."""
    if not spans:
        return pcm
    audio = bytearray(pcm)
    total = len(pcm) // 2
    for start, end in spans:
        low, high = min(max(start, 0), total), min(max(end, 0), total)
        if high > low:
            audio[low * 2 : high * 2] = bytes((high - low) * 2)
    return bytes(audio)


def _holds_run(words: list[str], run: list[str]) -> bool:
    size = len(run)
    return any(words[i : i + size] == run for i in range(len(words) - size + 1))
