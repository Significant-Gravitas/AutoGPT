"""Tests for All Quiet webhook signature verification."""

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from fastapi import HTTPException
from pydantic import SecretStr
from starlette.requests import Request

from backend.blocks.allquiet._webhook import (
    AllQuietWebhooksManager,
    AllQuietWebhookType,
)
from backend.blocks.allquiet.triggers import AllQuietIncidentTriggerBlock

SECRET = "s3cr3t-signing-key"
BODY = b'{"id":"81cd20be","title":"RAM above 60%"}'


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _sign(body: bytes, timestamp: str, secret: str = SECRET) -> str:
    return base64.b64encode(
        hmac.new(
            secret.encode("utf-8"),
            msg=f"{timestamp}:".encode("utf-8") + body,
            digestmod=hashlib.sha256,
        ).digest()
    ).decode("utf-8")


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


def _webhook(
    *, node_inputs: list[dict] | None = None, preset_inputs: list[dict] | None = None
):
    return SimpleNamespace(
        id="webhook-1",
        triggered_nodes=[
            SimpleNamespace(input_default=inputs) for inputs in node_inputs or []
        ],
        triggered_presets=[
            SimpleNamespace(inputs=inputs) for inputs in preset_inputs or []
        ],
    )


def _signed_webhook(secret: str = SECRET):
    return _webhook(node_inputs=[{"signing_secret": secret}])


class TestManagerIsConcrete:
    def test_can_be_instantiated(self):
        # Regression: BaseWebhooksManager declares validate_payload abstract, so
        # omitting it still imports and registers fine and only explodes at
        # trigger-setup time with "Can't instantiate abstract class
        # AllQuietWebhooksManager without an implementation for abstract method
        # 'validate_payload'". The block test harness never instantiates the
        # manager, so nothing else catches this.
        AllQuietWebhooksManager()

    def test_implements_every_abstract_method(self):
        assert not getattr(AllQuietWebhooksManager, "__abstractmethods__", frozenset())


class TestValidatePayload:
    async def test_returns_the_body_and_the_incident_event_type(self):
        body = {"id": "inc-1", "title": "RAM above 60%"}
        request = _request({}, body=json.dumps(body).encode())

        payload, event_type = await AllQuietWebhooksManager.validate_payload(
            _webhook(), request
        )

        assert payload == body
        assert event_type == AllQuietWebhookType.INCIDENT

    async def test_rejects_a_body_that_is_not_json(self):
        request = _request({}, body=b"not json at all")

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.validate_payload(_webhook(), request)
        assert exc.value.status_code == 400

    async def test_rejects_a_json_body_that_is_not_an_object(self):
        # A malformed Handlebars template can render a bare array or string.
        request = _request({}, body=b'["not", "an", "object"]')

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.validate_payload(_webhook(), request)
        assert exc.value.status_code == 400
        assert "JSON object" in exc.value.detail


class TestUnsignedWebhooks:
    async def test_accepts_anything_when_no_secret_is_configured(self):
        # Signing is opt-in in All Quiet; the URL is the only credential.
        await AllQuietWebhooksManager.verify_signature(_webhook(), _request({}))

    async def test_treats_a_blank_secret_as_unconfigured(self):
        webhook = _webhook(node_inputs=[{"signing_secret": "   "}])

        await AllQuietWebhooksManager.verify_signature(webhook, _request({}))


class TestSignedWebhooks:
    async def test_accepts_a_valid_allquiet_format_signature(self):
        timestamp = _now()
        request = _request(
            {"x-aq-signature": _sign(BODY, timestamp), "x-aq-timestamp": timestamp}
        )

        await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)

    async def test_accepts_a_valid_aws_format_signature(self):
        timestamp = _now()
        request = _request(
            {
                "x-amzn-event-signature": _sign(BODY, timestamp),
                "x-amzn-event-timestamp": timestamp,
            }
        )

        await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)

    async def test_rejects_a_signature_made_with_the_wrong_secret(self):
        timestamp = _now()
        request = _request(
            {
                "x-aq-signature": _sign(BODY, timestamp, secret="not-the-secret"),
                "x-aq-timestamp": timestamp,
            }
        )

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)
        assert exc.value.status_code == 403

    async def test_rejects_a_tampered_body(self):
        timestamp = _now()
        request = _request(
            {"x-aq-signature": _sign(BODY, timestamp), "x-aq-timestamp": timestamp},
            body=b'{"id":"attacker-controlled"}',
        )

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)
        assert exc.value.status_code == 403

    async def test_rejects_a_signature_bound_to_a_different_timestamp(self):
        # The timestamp is part of the signed material, so swapping it must fail.
        request = _request(
            {
                "x-aq-signature": _sign(BODY, "2020-01-01T00:00:00.000Z"),
                "x-aq-timestamp": _now(),
            }
        )

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)
        assert exc.value.status_code == 403

    async def test_rejects_an_unsigned_request_when_a_secret_is_set(self):
        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(
                _signed_webhook(), _request({})
            )

        assert exc.value.status_code == 403
        assert "x-aq-signature" in exc.value.detail

    async def test_rejects_a_signature_with_no_timestamp_header(self):
        timestamp = _now()
        request = _request({"x-aq-signature": _sign(BODY, timestamp)})

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)
        assert exc.value.status_code == 403


class TestReplayProtection:
    async def test_rejects_a_stale_but_correctly_signed_delivery(self):
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        )
        request = _request({"x-aq-signature": _sign(BODY, old), "x-aq-timestamp": old})

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)
        assert "window" in exc.value.detail


class TestSecretDiscovery:
    async def test_reads_the_secret_from_a_preset(self):
        timestamp = _now()
        request = _request(
            {"x-aq-signature": _sign(BODY, timestamp), "x-aq-timestamp": timestamp}
        )
        webhook = _webhook(preset_inputs=[{"signing_secret": SECRET}])

        await AllQuietWebhooksManager.verify_signature(webhook, request)

    async def test_accepts_a_secret_stored_as_a_secretstr(self):
        timestamp = _now()
        request = _request(
            {"x-aq-signature": _sign(BODY, timestamp), "x-aq-timestamp": timestamp}
        )
        webhook = _webhook(node_inputs=[{"signing_secret": SecretStr(SECRET)}])

        await AllQuietWebhooksManager.verify_signature(webhook, request)

    async def test_ignores_nodes_without_a_secret(self):
        timestamp = _now()
        request = _request(
            {"x-aq-signature": _sign(BODY, timestamp), "x-aq-timestamp": timestamp}
        )
        webhook = _webhook(node_inputs=[{}, {"signing_secret": SECRET}])

        await AllQuietWebhooksManager.verify_signature(webhook, request)


class TestSigningSecretFieldCoupling:
    """`SIGNING_SECRET_INPUT` is matched against the block's field name by string.

    A rename on either side would silently downgrade every signed webhook to
    "accept anything", with no type or test failure anywhere else.
    """

    def test_names_a_real_field_on_the_trigger_block(self):
        fields = AllQuietIncidentTriggerBlock().input_schema.model_fields

        assert AllQuietWebhooksManager.SIGNING_SECRET_INPUT in fields

    def test_that_field_is_marked_secret(self):
        schema = AllQuietIncidentTriggerBlock().input_schema.model_json_schema()
        field = schema["properties"][AllQuietWebhooksManager.SIGNING_SECRET_INPUT]

        assert field.get("secret") is True


class TestTimestampFormats:
    """All Quiet's two header pairs don't share a timestamp format."""

    @pytest.mark.parametrize(
        "format_stamp",
        [
            lambda now: now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            lambda now: now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            lambda now: now.isoformat(),
            lambda now: str(int(now.timestamp())),
            lambda now: str(int(now.timestamp() * 1000)),
        ],
        ids=["aq-millis", "aws-iso", "iso-offset", "epoch-seconds", "epoch-millis"],
    )
    async def test_accepts_each_format_all_quiet_may_send(
        self, format_stamp: Callable[[datetime], str]
    ):
        # Stamped here rather than in the parametrize list. Decorator arguments
        # are evaluated at import time, so on any suite that takes longer than
        # MAX_TIMESTAMP_SKEW to reach this test the stamp is already stale and
        # the replay check rejects it -- a failure that depends on how long the
        # rest of the suite took, not on this code.
        stamp = format_stamp(datetime.now(timezone.utc))
        request = _request(
            {"x-aq-signature": _sign(BODY, stamp), "x-aq-timestamp": stamp}
        )

        await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)

    @pytest.mark.parametrize(
        "millis",
        [
            # Leading digits spell a valid YYYYMMDD (1787-12-16), so
            # `fromisoformat` claims it on Python 3.11+ and returns a date
            # centuries in the past, which then fails the replay window.
            "1787121651526",
            # Leading digits are not a valid date (month 20), so this one
            # always fell through to the epoch branch and passed — which is why
            # the failure came and went with the wall clock.
            "1787207590123",
        ],
    )
    def test_epoch_millis_are_never_read_as_a_basic_format_date(self, millis: str):
        from backend.blocks.allquiet._webhook import _parse_timestamp

        parsed = _parse_timestamp(millis)

        assert parsed is not None
        assert parsed.year >= 2020, f"{millis} parsed as {parsed}"

    async def test_fails_closed_on_an_unrecognized_format(self):
        # Previously this was waved through, which left the replay window
        # permanently open for any sender using an unusual timestamp format.
        odd = "17/12/2023 11:51:08"
        request = _request({"x-aq-signature": _sign(BODY, odd), "x-aq-timestamp": odd})

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)
        assert "recognized format" in exc.value.detail

    async def test_rejects_a_stale_epoch_timestamp(self):
        old = str(int(datetime.now(timezone.utc).timestamp()) - 3600)
        request = _request({"x-aq-signature": _sign(BODY, old), "x-aq-timestamp": old})

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)
        assert "window" in exc.value.detail


class TestAmbiguousSecrets:
    async def test_refuses_when_targets_disagree_on_the_secret(self):
        # Picking a winner would make verification depend on node ordering.
        timestamp = _now()
        request = _request(
            {"x-aq-signature": _sign(BODY, timestamp), "x-aq-timestamp": timestamp}
        )
        webhook = _webhook(
            node_inputs=[{"signing_secret": SECRET}, {"signing_secret": "other"}]
        )

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(webhook, request)
        assert "different signing secrets" in exc.value.detail

    async def test_allows_targets_that_agree(self):
        timestamp = _now()
        request = _request(
            {"x-aq-signature": _sign(BODY, timestamp), "x-aq-timestamp": timestamp}
        )
        webhook = _webhook(
            node_inputs=[{"signing_secret": SECRET}, {"signing_secret": SECRET}]
        )

        await AllQuietWebhooksManager.verify_signature(webhook, request)


class TestFutureSkew:
    async def test_rejects_a_timestamp_from_the_future(self):
        # A clock-skewed or forged forward timestamp is as much a replay risk
        # as a stale one; the window is symmetric.
        ahead = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        )
        request = _request(
            {"x-aq-signature": _sign(BODY, ahead), "x-aq-timestamp": ahead}
        )

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)
        assert "window" in exc.value.detail

    async def test_allows_small_clock_skew(self):
        near = (datetime.now(timezone.utc) + timedelta(seconds=30)).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        )
        request = _request(
            {"x-aq-signature": _sign(BODY, near), "x-aq-timestamp": near}
        )

        await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)


class TestSignatureHeaderRobustness:
    async def test_a_non_ascii_signature_is_rejected_not_a_500(self):
        # hmac.compare_digest raises TypeError on a str with non-ASCII, which
        # would escape the handler as a 500 rather than a 403.
        timestamp = _now()
        request = _request(
            {"x-aq-signature": "ünicode-signature", "x-aq-timestamp": timestamp}
        )

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)
        assert exc.value.status_code == 403

    async def test_an_unreadable_stored_secret_fails_closed(self):
        # "Configured but unreadable" must not be mistaken for "unsigned",
        # which would accept every delivery.
        timestamp = _now()
        request = _request(
            {"x-aq-signature": _sign(BODY, timestamp), "x-aq-timestamp": timestamp}
        )
        webhook = _webhook(node_inputs=[{"signing_secret": {"not": "a secret"}}])

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(webhook, request)
        assert exc.value.status_code == 403
        assert "could not be read" in exc.value.detail
