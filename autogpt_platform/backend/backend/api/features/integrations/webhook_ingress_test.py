"""
Tests for the webhook ingress endpoint's signature verification path.

The endpoint is intentionally unauthenticated and delegates per-provider
signature checks to each manager's `verify_signature`. This file pins:

* Providers that DO sign (GitHub, Telegram, Exa, Airtable) reject
  missing/invalid sigs and accept valid ones, including correct use of the
  platform-issued signing secret (Exa's `config["exa_secret"]`, Airtable's
  base64-decoded `config["mac_secret"]`).
* Generic webhook honors an optional `secret_token` on the triggered block:
  passes through when unset, enforces when set.
* Providers without a signing scheme (Compass, Slant3D) pass through.
* `verify_signature` runs before `validate_payload` (call ordering).
"""

import base64
import hashlib
import hmac
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import fastapi.testclient

from backend.api.features.integrations.router import router
from backend.data.integrations import WebhookWithRelations
from backend.integrations.providers import ProviderName

app = fastapi.FastAPI()
app.include_router(router)
# Surface downstream provider errors (e.g. ValueError from a payload-schema
# check) as 500 responses rather than propagating, so we can assert on
# response codes here. The contract under test is signature verification.
client = fastapi.testclient.TestClient(app, raise_server_exceptions=False)


WEBHOOK_ID = "wh-12345"
WEBHOOK_SECRET = "s" * 64
USER_ID = "user-1"


def _make_webhook(
    provider: ProviderName,
    *,
    secret: str = WEBHOOK_SECRET,
    config: dict | None = None,
    triggered_nodes=None,
    triggered_presets=None,
) -> WebhookWithRelations:
    # `model_construct` skips field validation so we can pass duck-typed
    # stubs for `triggered_nodes`/`triggered_presets` instead of full
    # NodeModel / LibraryAgentPreset instances. The generic webhook manager
    # only reads `.input_default` / `.inputs` from those, so simple
    # MagicMocks suffice.
    return WebhookWithRelations.model_construct(
        id=WEBHOOK_ID,
        user_id=USER_ID,
        provider=provider,
        credentials_id="",
        webhook_type="test",
        resource="",
        events=[],
        config=config or {},
        secret=secret,
        provider_webhook_id="",
        triggered_nodes=triggered_nodes or [],
        triggered_presets=triggered_presets or [],
    )


def _patch_ingress(webhook):
    # Important: every downstream DB-touching call on the success path must be
    # mocked. Real Prisma calls inside FastAPI TestClient's blocking portal
    # would open connections on the portal's event loop, corrupting Prisma's
    # global client state for any session-scoped async fixture that runs
    # later in the suite (e.g. `oauth_test.py::test_user`).
    return [
        patch(
            "backend.api.features.integrations.router.get_webhook",
            AsyncMock(return_value=webhook),
        ),
        patch("backend.api.features.integrations.router.creds_manager"),
        patch(
            "backend.api.features.integrations.router.publish_webhook_event",
            AsyncMock(),
        ),
        patch(
            "backend.api.features.integrations.router.complete_onboarding_step",
            AsyncMock(),
        ),
        patch(
            "backend.api.features.integrations.router._execute_webhook_node_trigger",
            AsyncMock(),
        ),
        patch(
            "backend.api.features.integrations.router._execute_webhook_preset_trigger",
            AsyncMock(),
        ),
    ]


def _post(provider: str, headers: dict | None = None, body: bytes = b"{}"):
    return client.post(
        f"/{provider}/webhooks/{WEBHOOK_ID}/ingress",
        content=body,
        headers={"Content-Type": "application/json", **(headers or {})},
    )


def _manager(provider: ProviderName):
    if provider == ProviderName.COMPASS:
        from backend.integrations.webhooks.compass import CompassWebhookManager

        return CompassWebhookManager()
    if provider == ProviderName.SLANT3D:
        from backend.integrations.webhooks.slant3d import Slant3DWebhooksManager

        return Slant3DWebhooksManager()
    if provider == ProviderName.GITHUB:
        from backend.integrations.webhooks.github import GithubWebhooksManager

        return GithubWebhooksManager()
    if provider == ProviderName.TELEGRAM:
        from backend.integrations.webhooks.telegram import TelegramWebhooksManager

        return TelegramWebhooksManager()
    if provider == ProviderName("exa"):
        from backend.blocks.exa._webhook import ExaWebhookManager

        return ExaWebhookManager()
    if provider == ProviderName("airtable"):
        from backend.blocks.airtable._webhook import AirtableWebhookManager

        return AirtableWebhookManager()
    if provider == ProviderName("generic_webhook"):
        from backend.blocks.generic_webhook._webhook import GenericWebhooksManager

        return GenericWebhooksManager()
    raise ValueError(f"No manager mapping for {provider}")


def _run(webhook, provider_path: str, **kwargs):
    """Boilerplate: patch the router's deps, set the right manager, fire request."""
    patches = _patch_ingress(webhook) + [
        patch(
            "backend.api.features.integrations.router.get_webhook_manager",
            return_value=_manager(webhook.provider),
        )
    ]
    # ExitStack guarantees previously started patchers are stopped even if
    # a later one's __enter__ raises.
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return _post(provider_path, **kwargs)


# ---------------------------------------------------------------------------
# Unsigned providers: default no-op verify, ingress passes through.
# ---------------------------------------------------------------------------


class TestUnsignedProvidersPassThrough:
    """Compass and Slant3D have no upstream signing scheme. The router must
    not 403 their deliveries — the default no-op `verify_signature` covers
    them, and 403'ing here would break every legitimate delivery from those
    devices/services."""

    def test_compass_accepts_unsigned_request(self):
        webhook = _make_webhook(ProviderName.COMPASS)
        resp = _run(webhook, "compass")
        assert resp.status_code != 403, resp.text

    def test_slant3d_accepts_unsigned_request(self):
        webhook = _make_webhook(ProviderName.SLANT3D)
        # Use a Slant3D-shaped payload so the provider's own payload check
        # doesn't 500; the assertion is about signature, not schema.
        body = (
            b'{"orderId":"o","status":"shipped","trackingNumber":"t","carrierCode":"c"}'
        )
        resp = _run(webhook, "slant3d", body=body)
        assert resp.status_code != 403, resp.text


# ---------------------------------------------------------------------------
# Always-signed providers (GitHub, Telegram): existing contract is intact.
# ---------------------------------------------------------------------------


class TestAlwaysSignedProviders:
    def test_github_missing_signature_403(self):
        webhook = _make_webhook(ProviderName.GITHUB)
        resp = _run(webhook, "github", headers={"X-GitHub-Event": "push"})
        assert resp.status_code == 403, resp.text

    def test_github_wrong_signature_403(self):
        webhook = _make_webhook(ProviderName.GITHUB)
        body = b'{"x":1}'
        # HMAC over the same body but using a different secret — well-formed
        # `sha256=...` header, wrong digest.
        wrong_sig = (
            "sha256=" + hmac.new(b"other-secret", body, hashlib.sha256).hexdigest()
        )
        resp = _run(
            webhook,
            "github",
            body=body,
            headers={"X-Hub-Signature-256": wrong_sig, "X-GitHub-Event": "push"},
        )
        assert resp.status_code == 403, resp.text

    def test_github_correct_signature_accepted(self):
        webhook = _make_webhook(ProviderName.GITHUB)
        body = b'{"x":1}'
        sig = (
            "sha256="
            + hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
        )
        resp = _run(
            webhook,
            "github",
            body=body,
            headers={"X-Hub-Signature-256": sig, "X-GitHub-Event": "push"},
        )
        assert resp.status_code != 403, resp.text

    def test_telegram_missing_signature_403(self):
        webhook = _make_webhook(ProviderName.TELEGRAM)
        resp = _run(webhook, "telegram")
        assert resp.status_code == 403, resp.text

    def test_telegram_wrong_token_403(self):
        webhook = _make_webhook(ProviderName.TELEGRAM)
        resp = _run(
            webhook,
            "telegram",
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-token"},
        )
        assert resp.status_code == 403, resp.text

    def test_telegram_correct_signature_accepted(self):
        webhook = _make_webhook(ProviderName.TELEGRAM)
        resp = _run(
            webhook,
            "telegram",
            headers={"X-Telegram-Bot-Api-Secret-Token": WEBHOOK_SECRET},
        )
        assert resp.status_code != 403, resp.text


# ---------------------------------------------------------------------------
# Exa signature verification.
# ---------------------------------------------------------------------------


class TestExaSignature:
    """Exa enforces a correct `Exa-Signature` header of the form
    `t=<ts>,v1=<hex>`, where the signed payload is `<ts>.<raw body>` and the
    HMAC key is `config["exa_secret"]`.
    See https://docs.exa.ai/websets/api/webhooks/verifying-signatures"""

    EXA_SECRET = "exa-signing-secret-deadbeef"
    TS = "1234567890"

    def _webhook(self):
        return _make_webhook(
            ProviderName("exa"), config={"exa_secret": self.EXA_SECRET}
        )

    def _header(self, body: bytes, ts: str | None = None) -> str:
        ts = ts or self.TS
        sig = hmac.new(
            self.EXA_SECRET.encode(), f"{ts}.".encode() + body, hashlib.sha256
        ).hexdigest()
        return f"t={ts},v1={sig}"

    def test_missing_signature_403(self):
        resp = _run(self._webhook(), "exa")
        assert resp.status_code == 403, resp.text

    def test_malformed_signature_403(self):
        resp = _run(self._webhook(), "exa", headers={"Exa-Signature": "nope"})
        assert resp.status_code == 403, resp.text

    def test_wrong_signature_403(self):
        resp = _run(
            self._webhook(),
            "exa",
            headers={"Exa-Signature": f"t={self.TS},v1=deadbeef"},
        )
        assert resp.status_code == 403, resp.text

    def test_signature_over_body_only_rejected(self):
        """A signature computed over the raw body WITHOUT the `<ts>.` prefix
        (the previous broken implementation) must NOT verify."""
        body = b'{"eventType":"webset.created"}'
        body_only_sig = hmac.new(
            self.EXA_SECRET.encode(), body, hashlib.sha256
        ).hexdigest()
        resp = _run(
            self._webhook(),
            "exa",
            body=body,
            headers={"Exa-Signature": f"t={self.TS},v1={body_only_sig}"},
        )
        assert resp.status_code == 403, resp.text

    def test_correct_signature_accepted(self):
        body = b'{"eventType":"webset.created"}'
        resp = _run(
            self._webhook(),
            "exa",
            body=body,
            headers={"Exa-Signature": self._header(body)},
        )
        assert resp.status_code != 403, resp.text

    def test_missing_config_secret_403(self):
        """If a stored Exa webhook somehow lacks `config["exa_secret"]`
        (e.g. legacy DB row), we must fail-closed rather than silently
        accept."""
        wh = _make_webhook(ProviderName("exa"), config={})
        resp = _run(wh, "exa", headers={"Exa-Signature": f"t={self.TS},v1=anything"})
        assert resp.status_code == 403, resp.text


# ---------------------------------------------------------------------------
# Airtable signature verification.
# ---------------------------------------------------------------------------


class TestAirtableSignature:
    """Airtable's signing key is stored base64-encoded as Airtable returns it
    — must be decoded before use as the HMAC key. Header format is
    `hmac-sha256=<hex>`."""

    AIRTABLE_RAW_KEY = b"airtable-binary-signing-key-32-byte-x"
    AIRTABLE_B64_KEY = base64.b64encode(AIRTABLE_RAW_KEY).decode()

    def _webhook(self, config_override: dict | None = None):
        cfg = (
            {"mac_secret": self.AIRTABLE_B64_KEY}
            if config_override is None
            else config_override
        )
        return _make_webhook(ProviderName("airtable"), config=cfg)

    def _expected_header(self, body: bytes) -> str:
        return (
            "hmac-sha256="
            + hmac.new(self.AIRTABLE_RAW_KEY, body, hashlib.sha256).hexdigest()
        )

    def test_missing_signature_403(self):
        resp = _run(self._webhook(), "airtable")
        assert resp.status_code == 403, resp.text

    def test_correct_signature_accepted(self):
        body = b'{"base":{"id":"app"},"webhook":{"id":"w"},"timestamp":"t"}'
        resp = _run(
            self._webhook(),
            "airtable",
            body=body,
            headers={"X-Airtable-Content-MAC": self._expected_header(body)},
        )
        assert resp.status_code != 403, resp.text

    def test_missing_config_mac_secret_403(self):
        """Symmetric with the Exa case: a stored Airtable webhook lacking
        `config["mac_secret"]` (legacy/corrupted row) must fail-closed, not
        silently accept the request."""
        wh = self._webhook(config_override={})
        resp = _run(
            wh,
            "airtable",
            headers={"X-Airtable-Content-MAC": "hmac-sha256=anything"},
        )
        assert resp.status_code == 403, resp.text

    def test_base64_string_used_as_key_would_fail(self):
        """Regression: the previous implementation used the base64 STRING as
        the HMAC key. A signature built that way must NOT verify under the
        new (correct) implementation that base64-decodes first."""
        body = b'{"base":{"id":"app"},"webhook":{"id":"w"},"timestamp":"t"}'
        wrong_sig = (
            "hmac-sha256="
            + hmac.new(self.AIRTABLE_B64_KEY.encode(), body, hashlib.sha256).hexdigest()
        )
        resp = _run(
            self._webhook(),
            "airtable",
            body=body,
            headers={"X-Airtable-Content-MAC": wrong_sig},
        )
        assert resp.status_code == 403, resp.text


# ---------------------------------------------------------------------------
# Generic webhook: optional secret_token on the triggered block.
# ---------------------------------------------------------------------------


class TestGenericWebhookOptionalToken:
    """Generic webhook is unauthenticated by default. When the user sets a
    `secret_token` on the trigger block, the manager requires inbound
    requests to echo it in `X-Webhook-Secret`."""

    def _webhook(self, secret_token: str | None = None):
        nodes = []
        if secret_token is not None:
            n = MagicMock()
            n.input_default = {"secret_token": secret_token}
            nodes.append(n)
        return _make_webhook(ProviderName("generic_webhook"), triggered_nodes=nodes)

    def test_no_token_configured_accepts_unsigned(self):
        resp = _run(self._webhook(secret_token=None), "generic_webhook")
        assert resp.status_code != 403, resp.text

    def test_empty_token_configured_accepts_unsigned(self):
        """An empty string for `secret_token` is treated as 'not set' — back
        compat for users who left the field blank."""
        resp = _run(self._webhook(secret_token=""), "generic_webhook")
        assert resp.status_code != 403, resp.text

    def test_token_configured_missing_header_403(self):
        resp = _run(self._webhook(secret_token="t0ken"), "generic_webhook")
        assert resp.status_code == 403, resp.text

    def test_token_configured_wrong_header_403(self):
        resp = _run(
            self._webhook(secret_token="t0ken"),
            "generic_webhook",
            headers={"X-Webhook-Secret": "nope"},
        )
        assert resp.status_code == 403, resp.text

    def test_token_configured_correct_header_accepted(self):
        resp = _run(
            self._webhook(secret_token="t0ken"),
            "generic_webhook",
            headers={"X-Webhook-Secret": "t0ken"},
        )
        assert resp.status_code != 403, resp.text


# ---------------------------------------------------------------------------
# Ordering: verify_signature runs BEFORE validate_payload.
# ---------------------------------------------------------------------------


class TestVerificationOrder:
    """A future refactor must not invert these two calls — otherwise a
    payload parser (e.g. Airtable's `list_webhook_payloads` API call) could
    fire before the request is authenticated."""

    def test_validate_payload_not_called_on_bad_signature(self):
        webhook = _make_webhook(ProviderName.GITHUB)

        mock_manager = MagicMock()
        mock_manager.verify_signature = AsyncMock(
            side_effect=fastapi.HTTPException(status_code=403, detail="nope")
        )
        mock_manager.validate_payload = AsyncMock(return_value=({}, "evt"))

        patches = _patch_ingress(webhook) + [
            patch(
                "backend.api.features.integrations.router.get_webhook_manager",
                return_value=mock_manager,
            )
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            resp = _post("github")

        assert resp.status_code == 403
        mock_manager.verify_signature.assert_awaited_once()
        mock_manager.validate_payload.assert_not_called()
