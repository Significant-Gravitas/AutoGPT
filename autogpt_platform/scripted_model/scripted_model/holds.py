"""Replies held open until a journey has done something mid-answer.

A turn with ``stream.hold_after_words`` stops after that many words and waits
until the journey releases its run with ``POST /__scripted__/release?nonce=``,
for example once it has cut the browser's connection. The cut then always
lands mid-answer, however quickly the model writes or slowly the page shows
the first words.

Each release lets one held reply through. One sent before the reply reaches
its hold lets it straight through, and a second held reply in the same run
waits for a second release.
"""

import threading


class Holds:
    def __init__(self) -> None:
        self._releases: dict[str, threading.Semaphore] = {}
        self._lock = threading.Lock()

    def release(self, nonce: str) -> None:
        self._run(nonce).release()

    def wait(self, nonce: str, timeout_s: float) -> bool:
        """Whether a release for the run came within ``timeout_s``; it is
        used up by this reply."""
        return self._run(nonce).acquire(timeout=timeout_s)

    def _run(self, nonce: str) -> threading.Semaphore:
        with self._lock:
            return self._releases.setdefault(nonce, threading.Semaphore(0))
