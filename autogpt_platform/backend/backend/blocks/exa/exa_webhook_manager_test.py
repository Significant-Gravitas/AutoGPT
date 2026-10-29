"""Tests for the Exa webhook manager: registration, signatures and payloads."""

import hashlib
import hmac
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from pydantic import SecretStr
from starlette.requests import Request

from backend.blocks.exa._exa_rest import ExaApiError
from backend.blocks.exa._test import TEST_CREDENTIALS
from backend.blocks.exa._webhook import (
    FILTER_FIELD_EVENTS,
    SEARCH_MONITOR_SECRET_INPUT,
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


def _monitor_webhook(*node_inputs: dict) -> Any:
    return SimpleNamespace(
        id="wh-2",
        webhook_type="search_monitor",
        config={},
        triggered_nodes=[SimpleNamespace(input_default=i) for i in node_inputs],
        triggered_presets=[],
    )


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

    async def test_missing_secret_in_response_fails(self):
        exa = AsyncMock(return_value={"id": "webhook_1", "secret": None})
        with patch("backend.blocks.exa._webhook.exa_request", exa), pytest.raises(
            ValueError, match="signing secret"
        ):
            await ExaWebhookManager()._register_webhook(
                TEST_CREDENTIALS, "webset", "", ["item_created"], "u", "s"
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
        with pytest.raises(HTTPException) as exc:
            await ExaWebhookManager.verify_signature(
                _webset_webhook(), _request({"Exa-Signature": header})
            )
        assert exc.value.status_code == 403

    async def test_tampered_body_is_rejected(self):
        header = f"t={TIMESTAMP},v1={_sign(BODY)}"
        with pytest.raises(HTTPException):
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


class TestSearchMonitorSignature:
    async def test_valid_signature_with_block_secret_passes(self):
        header = f"t={TIMESTAMP},v1={_sign(BODY)}"
        await ExaWebhookManager.verify_signature(
            _monitor_webhook({"webhook_secret": SECRET}),
            _request({"Exa-Signature": header}),
        )

    async def test_secret_stored_as_secretstr_works(self):
        header = f"t={TIMESTAMP},v1={_sign(BODY)}"
        await ExaWebhookManager.verify_signature(
            _monitor_webhook({"webhook_secret": SecretStr(SECRET)}),
            _request({"Exa-Signature": header}),
        )

    async def test_invalid_signature_is_rejected(self):
        header = f"t={TIMESTAMP},v1={_sign(BODY, secret='other')}"
        with pytest.raises(HTTPException):
            await ExaWebhookManager.verify_signature(
                _monitor_webhook({"webhook_secret": SECRET}),
                _request({"Exa-Signature": header}),
            )

    async def test_missing_header_is_rejected_once_a_secret_is_set(self):
        with pytest.raises(HTTPException):
            await ExaWebhookManager.verify_signature(
                _monitor_webhook({"webhook_secret": SECRET}), _request({})
            )

    @pytest.mark.parametrize(
        "inputs", [{}, {"webhook_secret": None}, {"webhook_secret": "  "}]
    )
    async def test_no_secret_yet_accepts_the_delivery(self, inputs):
        await ExaWebhookManager.verify_signature(_monitor_webhook(inputs), _request({}))

    async def test_conflicting_secrets_are_refused(self):
        with pytest.raises(HTTPException, match="same webhook secret"):
            await ExaWebhookManager.verify_signature(
                _monitor_webhook({"webhook_secret": "a"}, {"webhook_secret": "b"}),
                _request({}),
            )


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
