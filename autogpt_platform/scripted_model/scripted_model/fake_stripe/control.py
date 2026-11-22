"""The journeys' side of the fake Stripe, under ``/__stripe__/``.

- ``GET /__stripe__/events?object=<id>``: the webhook events about an object
  (a checkout session, say), each with every delivery and the platform's
  answer to it.
- ``POST /__stripe__/events/<event id>/deliver``: deliver an event again, as
  Stripe retries one, or for the first time if it was held.
- ``POST /__stripe__/events`` with ``{"type": ..., "object": <id>}``: a new
  event about the object as it is now, as Stripe sends a second event type
  for one payment. It is delivered at once, held object or not.
- ``POST /__stripe__/hold?object=<id>``: keep the events the object's changes
  raise from now on (paying for a session) until a journey delivers them.
- ``GET /__stripe__/requests?unserved=true``: API requests the fake doesn't
  model.
"""

import json
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

from .responses import Response

if TYPE_CHECKING:
    from .fake import FakeStripe


def handle(
    fake: "FakeStripe", method: str, path: str, query: str, body: bytes
) -> Response:
    parts = path.removeprefix("/__stripe__/").strip("/").split("/")
    params = {key: values[-1] for key, values in parse_qs(query).items()}
    if method == "GET" and parts == ["events"]:
        return Response.json(200, {"events": fake.events.find(params.get("object"))})
    if method == "POST" and parts == ["events"]:
        return _new_event(fake, body)
    if method == "POST" and len(parts) == 3 and parts[0::2] == ["events", "deliver"]:
        delivery = fake.events.deliver(parts[1])
        if delivery is None:
            return _missing(f"no event {parts[1]}")
        return Response.json(200, {"delivery": delivery})
    if method == "POST" and parts == ["hold"]:
        object_id = params.get("object")
        if not object_id:
            return Response.json(400, {"error": {"message": "hold needs ?object=<id>"}})
        fake.events.hold(object_id)
        return Response.json(200, {"held": object_id})
    if method == "GET" and parts == ["requests"]:
        unserved = params.get("unserved") == "true"
        return Response.json(200, {"requests": fake.requests(unserved_only=unserved)})
    return _missing(f"nothing serves {method} {path}")


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
