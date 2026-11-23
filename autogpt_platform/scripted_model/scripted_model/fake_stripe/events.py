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
        self._queue: queue.SimpleQueue[tuple[str, threading.Event]] = (
            queue.SimpleQueue()
        )
        self._sender: threading.Thread | None = None

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
        data: dict[str, Any] = {"object": public(obj)}
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
        sent = threading.Event()
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
                self._queue.put((event["id"], sent))
                self._start_sender()
        if not held and wait:
            sent.wait()
        return self.get(event["id"]) or {}

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
        """Delivers queued events one by one for as long as the fake runs. A
        delivery that fails is recorded as failed and the next one goes out,
        and whoever waits for one is let go either way."""
        while True:
            event_id, sent = self._queue.get()
            try:
                self.deliver(event_id)
            except Exception:
                logger.exception("fake Stripe: delivering %s failed", event_id)
            finally:
                sent.set()

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
