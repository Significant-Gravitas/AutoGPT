"""Tests for the Exa webhook manager: registration, signatures and payloads."""

import hashlib
import hmac
import json
import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from pydantic import SecretStr
from starlette.requests import Request

from backend.api.features.library.model import node_input_mask_key
from backend.blocks.exa import _webhook
from backend.blocks.exa._exa_rest import ExaApiError
from backend.blocks.exa._test import TEST_CREDENTIALS
from backend.blocks.exa._webhook import (
    FILTER_FIELD_EVENTS,
    SEARCH_MONITOR_SECRET_INPUT,
    SIGNATURE_TOLERANCE_SECONDS,
    ExaEventType,
    ExaWebhookManager,
    exa_event_names,
    filter_field_for,
)
from backend.blocks.exa.search_monitor_trigger import (
    ExaSearchMonitorTriggerBlock,
    SearchMonitorEventFilter,
)
from backend.blocks.exa.webhook_blocks import ExaWebsetWebhookBlock, WebsetEventFilter

SECRET = "whsec_test_secret"
BODY = b'{"id":"event_1","object":"event","type":"webset.item.created","data":{}}'
TIMESTAMP = "1767225600"
# The trigger node of a preset's graph: the preset stores the trigger block's
# inputs under this node's mask key.
TRIGGER_NODE_ID = "5b0f3c1e-7d2a-4e8f-9a6b-2c4d6e8f0a1b"

# The 19 values of `EventType` in Exa's OpenAPI spec (components/schemas).
SPEC_EVENT_TYPES = {
    "webset.created",
    "webset.deleted",
    "webset.paused",
    "webset.idle",
    "webset.search.created",
    "webset.search.canceled",
    "webset.search.completed",
    "webset.search.updated",
    "import.created",
    "import.completed",
    "webset.item.created",
    "webset.item.enriched",
    "monitor.created",
    "monitor.updated",
    "monitor.deleted",
    "monitor.run.created",
    "monitor.run.completed",
    "webset.export.created",
    "webset.export.completed",
}


def _sign(body: bytes, timestamp: str = TIMESTAMP, secret: str = SECRET) -> str:
    return hmac.new(
        secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256
    ).hexdigest()


def _request(headers: dict[str, str], body: bytes = BODY) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _webset_webhook(config: dict | None = None) -> Any:
    return SimpleNamespace(
        id="wh-1",
        webhook_type="webset",
        config={"exa_secret": SECRET} if config is None else config,
        triggered_nodes=[],
        triggered_presets=[],
    )


def _monitor_webhook(
    nodes: list[dict] | None = None, presets: list[dict] | None = None
) -> Any:
    """A Search Monitor webhook used by trigger nodes and presets with these
    inputs."""
    return SimpleNamespace(
        id="wh-2",
        webhook_type="search_monitor",
        config={},
        triggered_nodes=[SimpleNamespace(input_default=i) for i in nodes or []],
        triggered_presets=[SimpleNamespace(inputs=i) for i in presets or []],
    )


def _nested(trigger_config: dict) -> dict:
    """A preset's inputs as trigger setup stores them: the trigger block's inputs
    under the trigger node's mask key, beside the graph's own inputs."""
    return {"topic": "news", node_input_mask_key(TRIGGER_NODE_ID): trigger_config}


def _node_webhook(*trigger_configs: dict) -> Any:
    return _monitor_webhook(nodes=list(trigger_configs))


def _preset_webhook(*trigger_configs: dict) -> Any:
    return _monitor_webhook(presets=[_nested(c) for c in trigger_configs])


def _flat_preset_webhook(*trigger_configs: dict) -> Any:
    """Presets from before the trigger config was nested; boot migrates them."""
    return _monitor_webhook(presets=list(trigger_configs))


@pytest.fixture(
    params=[_node_webhook, _preset_webhook, _flat_preset_webhook],
    ids=["node", "preset", "flat-preset"],
)
def monitor_webhook(request: pytest.FixtureRequest) -> Callable[..., Any]:
    """Builds a Search Monitor webhook from its triggers' inputs, stored in each
    layout a trigger's inputs can have."""
    return request.param


@pytest.fixture
def pinned_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the verifier's clock to TIMESTAMP, when the test deliveries are signed."""
    monkeypatch.setattr(
        _webhook, "time", SimpleNamespace(time=lambda: float(TIMESTAMP))
    )


async def _refusal(webhook: Any, headers: dict[str, str]) -> HTTPException:
    with pytest.raises(HTTPException) as exc:
        await ExaWebhookManager.verify_signature(webhook, _request(headers))
    assert exc.value.status_code == 403
    return exc.value


class TestEventCatalogue:
    def test_enum_covers_every_spec_event_plus_the_legacy_one(self):
        values = {event.value for event in ExaEventType}
        assert SPEC_EVENT_TYPES <= values
        assert values - SPEC_EVENT_TYPES == {"import.processing"}

    def test_every_filter_field_maps_to_an_event(self):
        assert set(WebsetEventFilter.model_fields) == set(FILTER_FIELD_EVENTS)
        assert set(SearchMonitorEventFilter.model_fields) <= set(FILTER_FIELD_EVENTS)

    def test_new_monitor_filters_default_off(self):
        defaults = WebsetEventFilter()
        assert not any(
            getattr(defaults, name)
            for name in WebsetEventFilter.model_fields
            if name.startswith("monitor_")
        )

    def test_everything_enabled_fits_exas_limit_and_drops_legacy(self):
        events = exa_event_names(list(FILTER_FIELD_EVENTS))
        assert len(events) == 19
        assert set(events) == SPEC_EVENT_TYPES

    def test_unknown_filter_field_is_rejected(self):
        with pytest.raises(ValueError, match="nope"):
            exa_event_names(["nope"])

    def test_event_names_round_trip_to_filter_fields(self):
        assert filter_field_for("webset.search.completed") == "search_completed"
        assert filter_field_for("monitor.run.completed") == "monitor_run_completed"
        assert filter_field_for("something.new") == "something.new"

    def test_trigger_secret_input_name_matches_manager(self):
        fields = ExaSearchMonitorTriggerBlock.Input.model_fields
        assert SEARCH_MONITOR_SECRET_INPUT in fields

    @pytest.mark.parametrize(
        "block_cls, filter_input, event, enabled",
        [
            (ExaWebsetWebhookBlock, "event_filter", "webset.item.created", True),
            (ExaWebsetWebhookBlock, "event_filter", "monitor.run.completed", False),
            (ExaSearchMonitorTriggerBlock, "events", "monitor.run.completed", True),
            (ExaSearchMonitorTriggerBlock, "events", "monitor.created", False),
        ],
    )
    def test_platform_matches_deliveries_against_default_filters(
        self, block_cls, filter_input, event, enabled
    ):
        block = block_cls()
        filter_model = block.input_schema.model_fields[filter_input].annotation
        trigger_config = {filter_input: filter_model().model_dump()}
        assert (
            block.is_triggered_by_event_type(trigger_config, filter_field_for(event))
            is enabled
        )


class TestRegistration:
    async def test_posts_to_the_websets_webhooks_endpoint(self):
        exa = AsyncMock(return_value={"id": "webhook_1", "secret": "s3"})
        with patch("backend.blocks.exa._webhook.exa_request", exa):
            webhook_id, config = await ExaWebhookManager()._register_webhook(
                TEST_CREDENTIALS,
                "webset",
                "webset_1",
                ["item_created", "import_processing", "monitor_run_completed"],
                "https://platform.example/api/integrations/exa/webhooks/x/ingress",
                "platform-secret",
            )

        args, kwargs = exa.call_args
        assert args[1:] == ("POST", "https://api.exa.ai/websets/v0/webhooks")
        assert kwargs["body"] == {
            "url": "https://platform.example/api/integrations/exa/webhooks/x/ingress",
            "events": ["webset.item.created", "monitor.run.completed"],
            "metadata": {"resource": "webset_1", "webhook_type": "webset"},
        }
        assert webhook_id == "webhook_1"
        assert config["exa_secret"] == "s3"
        assert config["events"] == ["webset.item.created", "monitor.run.completed"]

    async def test_only_legacy_events_is_refused_before_calling_exa(self):
        exa = AsyncMock()
        with patch("backend.blocks.exa._webhook.exa_request", exa), pytest.raises(
            ValueError, match="legacy"
        ):
            await ExaWebhookManager()._register_webhook(
                TEST_CREDENTIALS, "webset", "", ["import_processing"], "u", "s"
            )
        exa.assert_not_called()

    @pytest.mark.parametrize("delete_fails", [False, True])
    async def test_missing_secret_in_response_fails_and_removes_the_webhook(
        self, delete_fails
    ):
        created = {"id": "webhook_1", "secret": None}
        exa = AsyncMock(
            side_effect=[created, RuntimeError("down") if delete_fails else {}]
        )
        with patch("backend.blocks.exa._webhook.exa_request", exa), pytest.raises(
            ValueError, match="signing secret"
        ):
            await ExaWebhookManager()._register_webhook(
                TEST_CREDENTIALS, "webset", "", ["item_created"], "u", "s"
            )

        assert exa.call_args.args[1:] == (
            "DELETE",
            "https://api.exa.ai/websets/v0/webhooks/webhook_1",
        )


class TestDeregistration:
    def _webhook(self, provider_webhook_id: str = "webhook_1") -> Any:
        return SimpleNamespace(provider_webhook_id=provider_webhook_id)

    async def test_deletes_at_the_websets_endpoint(self):
        exa = AsyncMock(return_value={"id": "webhook_1", "status": "inactive"})
        with patch("backend.blocks.exa._webhook.exa_request", exa):
            await ExaWebhookManager()._deregister_webhook(
                self._webhook(), TEST_CREDENTIALS
            )
        args, _ = exa.call_args
        assert args[1:] == (
            "DELETE",
            "https://api.exa.ai/websets/v0/webhooks/webhook_1",
        )

    async def test_already_deleted_is_fine(self):
        exa = AsyncMock(side_effect=ExaApiError("gone", 404))
        with patch("backend.blocks.exa._webhook.exa_request", exa):
            await ExaWebhookManager()._deregister_webhook(
                self._webhook(), TEST_CREDENTIALS
            )

    async def test_other_errors_propagate(self):
        exa = AsyncMock(side_effect=ExaApiError("boom", 500))
        with patch("backend.blocks.exa._webhook.exa_request", exa), pytest.raises(
            ExaApiError
        ):
            await ExaWebhookManager()._deregister_webhook(
                self._webhook(), TEST_CREDENTIALS
            )

    async def test_manual_webhook_has_nothing_to_delete(self):
        exa = AsyncMock()
        with patch("backend.blocks.exa._webhook.exa_request", exa):
            await ExaWebhookManager()._deregister_webhook(
                self._webhook(""), TEST_CREDENTIALS
            )
        exa.assert_not_called()


@pytest.mark.usefixtures("pinned_clock")
class TestWebsetSignature:
    async def test_valid_signature_passes(self):
        header = f"t={TIMESTAMP},v1={_sign(BODY)}"
        await ExaWebhookManager.verify_signature(
            _webset_webhook(), _request({"Exa-Signature": header})
        )

    async def test_any_matching_v1_passes(self):
        header = f"t={TIMESTAMP},v1={'0' * 64},v1={_sign(BODY)}"
        await ExaWebhookManager.verify_signature(
            _webset_webhook(), _request({"Exa-Signature": header})
        )

    @pytest.mark.parametrize(
        "header",
        [
            f"t={TIMESTAMP},v1={'0' * 64}",
            f"t=1767225601,v1={_sign(BODY)}",  # timestamp is part of the signature
            f"t={TIMESTAMP},v1={_sign(BODY, secret='other')}",
            f"t={TIMESTAMP},v1=é",
        ],
    )
    async def test_invalid_signature_is_rejected(self, header):
        refusal = await _refusal(_webset_webhook(), {"Exa-Signature": header})
        assert refusal.detail == "Invalid webhook signature"

    async def test_tampered_body_is_rejected(self):
        header = f"t={TIMESTAMP},v1={_sign(BODY)}"
        with pytest.raises(HTTPException, match="Invalid webhook signature"):
            await ExaWebhookManager.verify_signature(
                _webset_webhook(),
                _request({"Exa-Signature": header}, body=BODY + b" "),
            )

    @pytest.mark.parametrize("headers", [{}, {"Exa-Signature": "v1=abc"}])
    async def test_missing_or_malformed_header_is_rejected(self, headers):
        with pytest.raises(HTTPException) as exc:
            await ExaWebhookManager.verify_signature(
                _webset_webhook(), _request(headers)
            )
        assert exc.value.status_code == 403

    async def test_missing_stored_secret_fails_closed(self):
        header = f"t={TIMESTAMP},v1={_sign(BODY)}"
        with pytest.raises(HTTPException, match="signing secret"):
            await ExaWebhookManager.verify_signature(
                _webset_webhook(config={}), _request({"Exa-Signature": header})
            )


@pytest.mark.usefixtures("pinned_clock")
class TestSearchMonitorSignature:
    async def test_valid_signature_with_block_secret_passes(self, monitor_webhook):
        header = f"t={TIMESTAMP},v1={_sign(BODY)}"
        await ExaWebhookManager.verify_signature(
            monitor_webhook({"webhook_secret": SECRET}),
            _request({"Exa-Signature": header}),
        )

    async def test_secret_stored_as_secretstr_works(self, monitor_webhook):
        header = f"t={TIMESTAMP},v1={_sign(BODY)}"
        await ExaWebhookManager.verify_signature(
            monitor_webhook({"webhook_secret": SecretStr(SECRET)}),
            _request({"Exa-Signature": header}),
        )

    async def test_invalid_signature_is_rejected(self, monitor_webhook):
        header = f"t={TIMESTAMP},v1={_sign(BODY, secret='other')}"
        with pytest.raises(HTTPException, match="Invalid webhook signature"):
            await ExaWebhookManager.verify_signature(
                monitor_webhook({"webhook_secret": SECRET}),
                _request({"Exa-Signature": header}),
            )

    async def test_missing_header_is_rejected_once_a_secret_is_set(
        self, monitor_webhook
    ):
        with pytest.raises(HTTPException, match="Missing Exa-Signature header"):
            await ExaWebhookManager.verify_signature(
                monitor_webhook({"webhook_secret": SECRET}), _request({})
            )

    @pytest.mark.parametrize("unreadable", [{"not": "a secret"}, ["a", "b"], 123])
    async def test_an_unreadable_secret_fails_closed(self, monitor_webhook, unreadable):
        # A secret is set but can't be read as one. Treating it as "no secret"
        # would accept unsigned deliveries, so the delivery is refused, even
        # one signed with the real secret.
        header = f"t={TIMESTAMP},v1={_sign(BODY)}"
        refusal = await _refusal(
            monitor_webhook({"webhook_secret": unreadable}), {"Exa-Signature": header}
        )
        assert "could not be read" in refusal.detail

    @pytest.mark.parametrize(
        "inputs", [{}, {"webhook_secret": None}, {"webhook_secret": "  "}]
    )
    async def test_no_secret_yet_accepts_the_delivery(self, monitor_webhook, inputs):
        await ExaWebhookManager.verify_signature(monitor_webhook(inputs), _request({}))

    @pytest.mark.parametrize("signing_secret", ["a", "b"])
    async def test_a_delivery_signed_by_any_monitor_on_the_webhook_passes(
        self, monitor_webhook, signing_secret
    ):
        # Two monitors' triggers share the webhook; each monitor signs with
        # its own secret.
        header = f"t={TIMESTAMP},v1={_sign(BODY, secret=signing_secret)}"
        await ExaWebhookManager.verify_signature(
            monitor_webhook({"webhook_secret": "a"}, {"webhook_secret": "b"}),
            _request({"Exa-Signature": header}),
        )

    async def test_a_delivery_signed_by_no_monitor_on_the_webhook_is_refused(
        self, monitor_webhook
    ):
        header = f"t={TIMESTAMP},v1={_sign(BODY, secret='other')}"
        with pytest.raises(HTTPException, match="Invalid webhook signature"):
            await ExaWebhookManager.verify_signature(
                monitor_webhook({"webhook_secret": "a"}, {"webhook_secret": "b"}),
                _request({"Exa-Signature": header}),
            )

    async def test_a_secret_on_any_trigger_of_the_webhook_is_enforced(self):
        # A preset has its monitor's secret; a node sharing the webhook has
        # none yet. Deliveries must still be signed.
        webhook = _monitor_webhook(
            nodes=[{}], presets=[_nested({"webhook_secret": SECRET})]
        )
        refusal = await _refusal(webhook, {})
        assert refusal.detail == "Missing Exa-Signature header"


@pytest.fixture(params=["webset", "search_monitor"])
def signed_webhook(request: pytest.FixtureRequest) -> Any:
    """A webhook of each Exa type whose deliveries are signed with SECRET."""
    if request.param == "webset":
        return _webset_webhook()
    return _preset_webhook({"webhook_secret": SECRET})


def _signed_at(offset: int) -> dict[str, str]:
    """Headers of a delivery correctly signed `offset` seconds from TIMESTAMP."""
    timestamp = str(int(TIMESTAMP) + offset)
    return {"Exa-Signature": f"t={timestamp},v1={_sign(BODY, timestamp)}"}


@pytest.mark.usefixtures("pinned_clock")
class TestSignatureTimestamp:
    """Both webhook types share the verifier: a delivery signed with the right
    secret is still refused when its timestamp is too far from now."""

    @pytest.mark.parametrize(
        "offset", [0, -SIGNATURE_TOLERANCE_SECONDS, SIGNATURE_TOLERANCE_SECONDS]
    )
    async def test_a_delivery_signed_within_the_window_passes(
        self, signed_webhook, offset
    ):
        await ExaWebhookManager.verify_signature(
            signed_webhook, _request(_signed_at(offset))
        )

    @pytest.mark.parametrize("age", [SIGNATURE_TOLERANCE_SECONDS + 1, 86_400])
    async def test_a_stale_delivery_is_refused(self, signed_webhook, age):
        refusal = await _refusal(signed_webhook, _signed_at(-age))
        assert "outside the accepted window" in refusal.detail

    @pytest.mark.parametrize("ahead", [SIGNATURE_TOLERANCE_SECONDS + 1, 86_400])
    async def test_a_future_dated_delivery_is_refused(self, signed_webhook, ahead):
        refusal = await _refusal(signed_webhook, _signed_at(ahead))
        assert "outside the accepted window" in refusal.detail

    @pytest.mark.parametrize(
        "timestamp",
        [
            pytest.param("soon", id="word"),
            pytest.param("1767225600.5", id="fraction"),
            pytest.param("1e9", id="exponent"),
            pytest.param("9" * 400, id="too-large-for-a-float"),
        ],
    )
    async def test_an_unreadable_timestamp_is_refused(self, signed_webhook, timestamp):
        header = f"t={timestamp},v1={_sign(BODY, timestamp)}"
        refusal = await _refusal(signed_webhook, {"Exa-Signature": header})
        assert refusal.detail == "Malformed Exa-Signature header"

    async def test_a_signed_delivery_refused_for_its_time_is_logged(
        self, signed_webhook, caplog: pytest.LogCaptureFixture
    ):
        # The line to look for when every delivery to a trigger with a secret
        # is refused: it names the webhook and how far off the time was
        with caplog.at_level(logging.WARNING, logger=_webhook.__name__):
            await _refusal(
                signed_webhook, _signed_at(-(SIGNATURE_TOLERANCE_SECONDS + 60))
            )

        assert f"webhook #{signed_webhook.id}" in caplog.text
        assert "signed 360s ago" in caplog.text
        assert SECRET not in caplog.text

    async def test_an_unsigned_delivery_out_of_time_leaves_no_log_line(
        self, signed_webhook, caplog: pytest.LogCaptureFixture
    ):
        timestamp = str(int(TIMESTAMP) - 86_400)
        header = f"t={timestamp},v1={_sign(BODY, timestamp, secret='other')}"
        with caplog.at_level(logging.WARNING, logger=_webhook.__name__):
            refusal = await _refusal(signed_webhook, {"Exa-Signature": header})

        assert refusal.detail == "Invalid webhook signature"
        assert caplog.text == ""


class TestValidatePayload:
    async def test_returns_payload_and_filter_field(self):
        payload = {"id": "e1", "type": "webset.item.enriched", "data": {}}
        result, event = await ExaWebhookManager.validate_payload(
            _webset_webhook(), _request({}, json.dumps(payload).encode()), None
        )
        assert result == payload
        assert event == "item_enriched"

    async def test_non_object_body_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            await ExaWebhookManager.validate_payload(
                _webset_webhook(), _request({}, b"[1, 2]"), None
            )
        assert exc.value.status_code == 400
