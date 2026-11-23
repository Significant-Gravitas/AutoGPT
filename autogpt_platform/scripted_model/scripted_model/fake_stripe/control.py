"""The journeys' side of the fake Stripe, under ``/__stripe__/``.

- ``GET /__stripe__/events?object=<id>``: the webhook events about an object
  (a checkout session, say), each with every delivery and the platform's
  answer to it; ``?customer=<id>`` lists those about the customer's objects.
- ``POST /__stripe__/events/<event id>/deliver``: deliver an event again, as
  Stripe retries one, or for the first time if it was held.
- ``POST /__stripe__/events`` with ``{"type": ..., "object": <id>}``: a new
  event about the object as it is now, as Stripe sends a second event type
  for one payment. It is delivered at once, held object or not.
- ``POST /__stripe__/hold?object=<id>``: keep the events the object's changes
  raise from now on (paying for a session) until a journey delivers them; an
  object the fake doesn't hold gets a 404.
- ``GET /__stripe__/requests?unserved=true&after=<n>``: API requests the
  fake doesn't model, numbered above ``n``; without ``unserved``, every
  recent request.
- ``GET /__stripe__/requests/cursor``: the latest request's number, for a
  journey to read at its start and pass as ``after``.
- ``GET /__stripe__/objects/<id>``: an object as the API shows it, for a
  journey to read what Stripe holds (a subscription's period) without a key.
- ``POST /__stripe__/subscriptions/<id>/end_period``: end an active
  subscription's current period now, as if the clock had reached it. One set
  to cancel at the period's end is cancelled (``customer.subscription.deleted``);
  any other renews, with a new paid invoice (``customer.subscription.updated``,
  ``invoice.paid``, ``invoice.payment_succeeded``). Answers with the
  subscription and those events, once they are delivered.
"""

import json
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

from .errors import ApiError
from .forms import as_int
from .responses import Response
from .store import public

if TYPE_CHECKING:
    from .fake import FakeStripe


def handle(
    fake: "FakeStripe", method: str, path: str, query: str, body: bytes
) -> Response:
    params = {key: values[-1] for key, values in parse_qs(query).items()}
    match method, path.removeprefix("/__stripe__/").strip("/").split("/"):
        case "GET", ["events"]:
            events = fake.events.find(params.get("object"), params.get("customer"))
            return Response.json(200, {"events": events})
        case "POST", ["events"]:
            return _new_event(fake, body)
        case "POST", ["events", event_id, "deliver"]:
            delivery = fake.events.deliver(event_id)
            if delivery is None:
                return _missing(f"no event {event_id}")
            return Response.json(200, {"delivery": delivery})
        case "POST", ["hold"]:
            return _hold(fake, params.get("object"))
        case "GET", ["requests"]:
            unserved = params.get("unserved") == "true"
            after = as_int(params.get("after"), 0)
            requests = fake.requests(unserved_only=unserved, after=after)
            return Response.json(200, {"requests": requests})
        case "GET", ["requests", "cursor"]:
            return Response.json(200, {"cursor": fake.last_request()})
        case "GET", ["objects", object_id]:
            obj = fake.store.get(object_id)
            return (
                _missing(f"no object {object_id}")
                if obj is None
                else Response.json(200, public(obj))
            )
        case "POST", ["subscriptions", subscription_id, "end_period"]:
            return _end_period(fake, subscription_id)
    return _missing(f"nothing serves {method} {path}")


def _end_period(fake: "FakeStripe", subscription_id: str) -> Response:
    try:
        subscription, events = fake.subscriptions.end_period(subscription_id)
    except ApiError as error:
        return Response.json(error.status, error.body)
    return Response.json(200, {"subscription": subscription, "events": events})


def _hold(fake: "FakeStripe", object_id: str | None) -> Response:
    """Holds an object's events; one the fake doesn't hold is a typo, and
    fails at once rather than leaving the journey's webhook unheld."""
    if not object_id:
        return Response.json(400, {"error": {"message": "hold needs ?object=<id>"}})
    if fake.store.get(object_id) is None:
        return _missing(f"no object {object_id}")
    fake.events.hold(object_id)
    return Response.json(200, {"held": object_id})


def _new_event(fake: "FakeStripe", body: bytes) -> Response:
    try:
        request = json.loads(body or b"{}")
    except json.JSONDecodeError:
        request = None
    if not isinstance(request, dict) or not request.get("type"):
        message = 'send {"type": "<event type>", "object": "<object id>"}'
        return Response.json(400, {"error": {"message": message}})
    obj = fake.store.get(str(request.get("object")))
    if obj is None:
        return _missing(f"no object {request.get('object')}")
    record = fake.events.emit(str(request["type"]), obj, holdable=False)
    return Response.json(200, record)


def _missing(message: str) -> Response:
    return Response.json(404, {"error": {"message": message}})
