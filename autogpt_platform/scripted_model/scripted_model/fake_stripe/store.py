"""What the fake Stripe holds: its objects by id, as Stripe would for the run.

Objects are plain dicts in Stripe's JSON shape. Keys starting with ``_`` are
the fake's own bookkeeping, such as a session's line items, and never leave
it (``public``). Every read is a copy, so a caller never changes the store
without ``update``.
"""

import copy
import secrets
import threading
import time
from collections.abc import Callable
from typing import Any

StripeObject = dict[str, Any]


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(12)}"


def now() -> int:
    return int(time.time())


def public(obj: StripeObject) -> StripeObject:
    """The object as the API shows it."""
    return {key: value for key, value in obj.items() if not key.startswith("_")}


class Store:
    def __init__(self) -> None:
        self._objects: dict[str, StripeObject] = {}
        # Re-entrant, so a change ``update`` applies can read the store,
        # e.g. to number an invoice it adds.
        self._lock = threading.RLock()

    def add(self, obj: StripeObject) -> StripeObject:
        with self._lock:
            self._objects[obj["id"]] = copy.deepcopy(obj)
        return obj

    def get(self, object_id: str, kind: str | None = None) -> StripeObject | None:
        """The object with this id, if it is a ``kind`` (any kind when None)."""
        with self._lock:
            obj = self._objects.get(object_id)
            if obj is None or (kind is not None and obj["object"] != kind):
                return None
            return copy.deepcopy(obj)

    def update(
        self,
        object_id: str,
        change: Callable[[StripeObject], None],
        also_add: tuple[StripeObject, ...] = (),
        committed: Callable[[StripeObject], None] | None = None,
    ) -> StripeObject | None:
        """Applies ``change`` to the object under the lock, so a check it makes
        (a session still open, say) and its write are one step, and adds
        ``also_add`` in the same step. An exception from ``change`` leaves the
        store as it was, without ``also_add``. ``committed`` gets the result
        in that step too, before another change can be made: the events it
        records go out in the order the changes were made."""
        with self._lock:
            obj = self._objects.get(object_id)
            if obj is None:
                return None
            changed = copy.deepcopy(obj)
            change(changed)
            self._objects[object_id] = changed
            for extra in also_add:
                self._objects[extra["id"]] = copy.deepcopy(extra)
            result = copy.deepcopy(changed)
            if committed is not None:
                committed(result)
            return result

    def find(
        self, kind: str, match: Callable[[StripeObject], bool]
    ) -> list[StripeObject]:
        """Every ``kind`` object ``match`` accepts, newest first, as Stripe
        lists them."""
        with self._lock:
            found = [
                copy.deepcopy(obj)
                for obj in self._objects.values()
                if obj["object"] == kind and match(obj)
            ]
        return found[::-1]
