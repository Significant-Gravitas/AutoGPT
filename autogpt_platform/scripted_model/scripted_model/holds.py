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
        # Releases sent for each run and not yet used; a run with none left
        # has no entry.
        self._banked: dict[str, int] = {}
        self._changed = threading.Condition()

    def release(self, nonce: str) -> None:
        with self._changed:
            self._banked[nonce] = self._banked.get(nonce, 0) + 1
            self._changed.notify_all()

    def wait(self, nonce: str, timeout_s: float) -> bool:
        """Whether a release for the run came within ``timeout_s``; this reply
        uses it up."""
        with self._changed:
            if not self._changed.wait_for(lambda: nonce in self._banked, timeout_s):
                return False
            if self._banked[nonce] > 1:
                self._banked[nonce] -= 1
            else:
                del self._banked[nonce]
            return True
