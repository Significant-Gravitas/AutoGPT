"""Replies held open until a journey has done something mid-answer.

A turn with ``stream.hold_after_words`` stops after that many words and waits
until the journey releases its run with ``POST /__scripted__/release?nonce=``,
for example once it has cut the browser's connection. The cut then always
lands mid-answer, however quickly the model writes or slowly the page shows
the first words. A release is kept, so one sent before the reply reaches its
hold lets the reply straight through.
"""

import threading


class Holds:
    def __init__(self) -> None:
        self._released: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def release(self, nonce: str) -> None:
        self._run(nonce).set()

    def wait(self, nonce: str, timeout_s: float) -> bool:
        """Whether the run was released within ``timeout_s``."""
        return self._run(nonce).wait(timeout_s)

    def _run(self, nonce: str) -> threading.Event:
        with self._lock:
            return self._released.setdefault(nonce, threading.Event())
