"""Webhook events: what the fake tells the platform, signed as Stripe signs them.

An event is delivered when it happens, unless a journey held its object, and
again whenever a journey asks, as Stripe retries one the platform didn't
acknowledge: the same event, with the same id and body, and a new signature.
Every delivery is kept with the platform's answer.

Deliveries go out one at a time, in the order the events happened. What the
fake's own pages and controls do (paying, cancelling in the billing portal,
ending a period) waits for its events to be delivered, so a journey knows the
platform has heard of it by the time the page answers. What the platform asks
for through the API never waits: as with Stripe, the API answers first and the
webhook follows. It must not wait either way, since the platform's webhook
handler calls the API back, and the call that raised the event may be holding
the idempotency lock that call needs (``StripeApi._idempotent``).

A change to a subscription records its events in the store step that makes it
(``record``, from ``Store.update``'s ``committed``), so two changes made at
once are announced in the order they were made. Waiting for a delivery comes
after that step, never under the store's lock, which the platform's handler
needs to read anything back.
"""

import copy
import hashlib
import hmac
import http.client
import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from .store import StripeObject, new_id, now, public

logger = logging.getLogger("scripted_model.fake_stripe")

# The platform's Stripe SDK (stripe 11) pins this version, and the fake's
# objects take its shapes: a subscription's period on the subscription, an
# invoice's subscription on the invoice.
API_VERSION = "2025-02-24.acacia"
DELIVERY_TIMEOUT_S = 30
BODY_SNIPPET_LENGTH = 300


def signature_header(secret: str, payload: bytes, timestamp: int) -> str:
    """``Stripe-Signature``: an HMAC-SHA256 of ``<timestamp>.<payload>``."""
    signed = f"{timestamp}.".encode() + payload
    digest = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={digest}"


class Events:
    def __init__(self, webhook_url: str, webhook_secret: str) -> None:
        self._webhook_url = webhook_url
        self._secret = webhook_secret
        self._records: dict[str, dict[str, Any]] = {}
        self._held: set[str] = set()
        self._lock = threading.Lock()
        # None tells the sender to stop (``close``).
        self._queue: queue.SimpleQueue[tuple[str, threading.Event] | None] = (
            queue.SimpleQueue()
        )
        self._sender: threading.Thread | None = None
        # The queued events' delivery signals, until each is delivered.
        self._sent: dict[str, threading.Event] = {}

    def close(self, timeout: float = DELIVERY_TIMEOUT_S) -> None:
        """Stops the sender once every queued event is delivered, so a test
        leaves no thread behind.

        The sender stays the only one until it has stopped: an event emitted
        while it closes is queued behind the stop marker, and it delivers
        that too before stopping. Only an event emitted after it stopped
        starts a new one."""
        with self._lock:
            sender = self._sender
            if sender is None or not sender.is_alive():
                return
            self._queue.put(None)
        sender.join(timeout)

    def hold(self, object_id: str) -> None:
        """Keep the events this object's changes raise from now on until a
        journey delivers them, e.g. to let the user back before Stripe's
        webhook arrives."""
        with self._lock:
            self._held.add(object_id)

    def hold_with(self, parent_id: str, *object_ids: str) -> None:
        """Hold ``object_ids`` too if ``parent_id`` is held: a held checkout
        holds the subscription and invoice paying it starts."""
        with self._lock:
            if parent_id in self._held:
                self._held.update(object_ids)

    def emit(
        self,
        event_type: str,
        obj: StripeObject,
        *,
        holdable: bool = True,
        wait: bool = True,
        previous_attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """A new event about ``obj`` as it is now, delivered in turn unless
        ``holdable`` and the object is held. ``wait`` returns only once it
        has been delivered; never wait from an API call (see above)."""
        event_id = self.record(
            event_type, obj, holdable=holdable, previous_attributes=previous_attributes
        )
        if wait:
            self.wait(event_id)
        return self.get(event_id) or {}

    def record(
        self,
        event_type: str,
        obj: StripeObject,
        *,
        holdable: bool = True,
        previous_attributes: dict[str, Any] | None = None,
    ) -> str:
        """``emit`` without the wait: records the event, queues it unless
        ``holdable`` and the object is held, and returns its id."""
        # A copy: the caller may hand ``obj`` on, and the API expands what it
        # returns in place.
        data: dict[str, Any] = {"object": copy.deepcopy(public(obj))}
        if previous_attributes is not None:
            data["previous_attributes"] = previous_attributes
        event = {
            "id": new_id("evt"),
            "object": "event",
            "api_version": API_VERSION,
            "created": now(),
            "data": data,
            "livemode": False,
            "pending_webhooks": 1,
            "request": {"id": None, "idempotency_key": None},
            "type": event_type,
        }
        customer = obj["id"] if obj.get("object") == "customer" else obj.get("customer")
        # Recorded and queued in one step, so concurrent events go out in the
        # order they are recorded.
        with self._lock:
            held = holdable and obj["id"] in self._held
            self._records[event["id"]] = {
                "id": event["id"],
                "type": event_type,
                "object": obj["id"],
                "customer": customer,
                "held": held,
                "event": event,
                "deliveries": [],
            }
            if not held:
                sent = threading.Event()
                self._sent[event["id"]] = sent
                self._queue.put((event["id"], sent))
                self._start_sender()
        return event["id"]

    def wait(self, *event_ids: str) -> None:
        """Returns once each of these events has been delivered; one held, or
        already delivered, doesn't wait. Never call it under the store's
        lock: a delivery waits for the platform, whose handler reads the
        store."""
        for event_id in event_ids:
            with self._lock:
                sent = self._sent.get(event_id)
            if sent is not None:
                sent.wait()

    def deliver(self, event_id: str) -> dict[str, Any] | None:
        """Sends the event (again) and records the platform's answer, or why
        it got none."""
        with self._lock:
            record = self._records.get(event_id)
            if record is None:
                return None
            payload = json.dumps(record["event"]).encode()
        at = time.time()
        try:
            delivery = self._post(payload, at)
        except Exception as error:
            delivery = _failure(at, error)
        with self._lock:
            record["deliveries"].append(delivery)
        return copy.deepcopy(delivery)

    def get(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            return copy.deepcopy(self._records.get(event_id))

    def find(
        self, object_id: str | None = None, customer: str | None = None
    ) -> list[dict[str, Any]]:
        """Every event, or those about ``object_id`` or about the objects of
        ``customer``, oldest first."""
        with self._lock:
            return [
                copy.deepcopy(record)
                for record in self._records.values()
                if (object_id is None or record["object"] == object_id)
                and (customer is None or record["customer"] == customer)
            ]

    def _start_sender(self) -> None:
        """Starts the thread that delivers the queue, or a new one if it
        died. Called under ``self._lock``."""
        if self._sender is None or not self._sender.is_alive():
            self._sender = threading.Thread(
                target=self._send_queued, name="fake-stripe-webhooks", daemon=True
            )
            self._sender.start()

    def _send_queued(self) -> None:
        """Delivers queued events one by one until ``close`` and an empty
        queue. A delivery that fails is recorded as failed and the next one
        goes out, and whoever waits for one is let go either way."""
        stopping = False
        while True:
            with self._lock:
                # Events are queued under this lock, so an empty queue here
                # stays empty until this sender is gone; an emit after that
                # starts a new one.
                if stopping and self._queue.empty():
                    if self._sender is threading.current_thread():
                        self._sender = None
                    return
            item = self._queue.get()
            if item is None:
                stopping = True
                continue
            event_id, sent = item
            try:
                self.deliver(event_id)
            except Exception:
                logger.exception("fake Stripe: delivering %s failed", event_id)
            finally:
                sent.set()
                with self._lock:
                    self._sent.pop(event_id, None)

    def _post(self, payload: bytes, at: float) -> dict[str, Any]:
        if not self._webhook_url or not self._secret:
            return {
                "at": at,
                "status": None,
                "body": "",
                "error": "no webhook endpoint: set FAKE_STRIPE_WEBHOOK_URL and"
                " FAKE_STRIPE_WEBHOOK_SECRET",
            }
        request = urllib.request.Request(
            self._webhook_url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Stripe-Signature": signature_header(
                    self._secret, payload, int(time.time())
                ),
                "User-Agent": "Stripe/1.0 (+https://stripe.com/docs/webhooks)",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=DELIVERY_TIMEOUT_S) as reply:
                return _delivery(at, reply.status, reply.read())
        except urllib.error.HTTPError as error:
            return _delivery(at, error.code, error.read())
        except (OSError, http.client.HTTPException) as error:
            # Refused, unreachable, timed out or cut short (a body shorter
            # than its Content-Length): Stripe would retry later.
            return _failure(at, error)


def _delivery(at: float, status: int, body: bytes) -> dict[str, Any]:
    snippet = body[:BODY_SNIPPET_LENGTH].decode(errors="replace")
    return {"at": at, "status": status, "body": snippet, "error": None}


def _failure(at: float, error: Exception) -> dict[str, Any]:
    """A delivery the platform never answered."""
    return {"at": at, "status": None, "body": "", "error": repr(error)}
