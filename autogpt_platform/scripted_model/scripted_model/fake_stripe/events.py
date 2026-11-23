"""Webhook events: what the fake tells the platform, signed as Stripe signs them.

An event is delivered when it happens, unless a journey held its object, and
again whenever a journey asks, as Stripe retries one the platform didn't
acknowledge: the same event, with the same id and body, and a new signature.
Every delivery is kept with the platform's answer.
"""

import copy
import hashlib
import hmac
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from .store import StripeObject, new_id, now, public

API_VERSION = "2025-03-31.basil"
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

    def hold(self, object_id: str) -> None:
        """Keep the events this object's changes raise from now on until a
        journey delivers them, e.g. to let the user back before Stripe's
        webhook arrives."""
        with self._lock:
            self._held.add(object_id)

    def emit(
        self, event_type: str, obj: StripeObject, *, holdable: bool = True
    ) -> dict[str, Any]:
        """A new event about ``obj`` as it is now, delivered at once unless
        ``holdable`` and the object is held."""
        event = {
            "id": new_id("evt"),
            "object": "event",
            "api_version": API_VERSION,
            "created": now(),
            "data": {"object": public(obj)},
            "livemode": False,
            "pending_webhooks": 1,
            "request": {"id": None, "idempotency_key": None},
            "type": event_type,
        }
        with self._lock:
            held = holdable and obj["id"] in self._held
            self._records[event["id"]] = {
                "id": event["id"],
                "type": event_type,
                "object": obj["id"],
                "held": held,
                "event": event,
                "deliveries": [],
            }
        if not held:
            self.deliver(event["id"])
        return self.get(event["id"]) or {}

    def deliver(self, event_id: str) -> dict[str, Any] | None:
        """Sends the event (again) and records the platform's answer."""
        with self._lock:
            record = self._records.get(event_id)
            if record is None:
                return None
            payload = json.dumps(record["event"]).encode()
        delivery = self._post(payload)
        with self._lock:
            record["deliveries"].append(delivery)
        return copy.deepcopy(delivery)

    def get(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            return copy.deepcopy(self._records.get(event_id))

    def find(self, object_id: str | None = None) -> list[dict[str, Any]]:
        """Every event, or those about ``object_id``, oldest first."""
        with self._lock:
            return [
                copy.deepcopy(record)
                for record in self._records.values()
                if object_id is None or record["object"] == object_id
            ]

    def _post(self, payload: bytes) -> dict[str, Any]:
        at = time.time()
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
        except OSError as error:
            # Refused, unreachable or timed out: Stripe would retry later.
            return {"at": at, "status": None, "body": "", "error": repr(error)}


def _delivery(at: float, status: int, body: bytes) -> dict[str, Any]:
    snippet = body[:BODY_SNIPPET_LENGTH].decode(errors="replace")
    return {"at": at, "status": status, "body": snippet, "error": None}
