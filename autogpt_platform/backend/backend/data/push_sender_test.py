"""Tests for fire-and-forget Web Push delivery."""

import base64
import hashlib
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
import pywebpush
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from backend.api.model import NotificationPayload
from backend.data import push_sender
from backend.data.push_subscription import PushSubscriptionDTO
from backend.util import secrets_guard


@pytest.fixture(autouse=True)
def clear_debounce():
    """Reset the per-user debounce state between tests."""
    push_sender._user_last_push.clear()
    push_sender._warn_published_vapid_key_once.cache_clear()
    yield
    push_sender._user_last_push.clear()
    push_sender._warn_published_vapid_key_once.cache_clear()


@pytest.fixture
def mock_db_client(mocker):
    """Provides a mocked DatabaseManagerAsyncClient with stub async methods."""
    client = MagicMock()
    client.get_user_push_subscriptions = AsyncMock(return_value=[])
    client.delete_push_subscription = AsyncMock()
    client.increment_push_fail_count = AsyncMock()
    mocker.patch(
        "backend.data.push_sender.get_database_manager_async_client",
        return_value=client,
    )
    return client


def _make_settings(
    private: str = "vapid-private",
    public: str = "vapid-public",
    email: str = "mailto:push@agpt.co",
) -> MagicMock:
    settings = MagicMock()
    settings.secrets.vapid_private_key = private
    settings.secrets.vapid_public_key = public
    settings.secrets.vapid_claim_email = email
    return settings


def _make_subscription(
    user_id: str = "user-1",
    endpoint: str = "https://fcm.googleapis.com/fcm/send/sub/1",
    p256dh: str = "test-p256dh",
    auth: str = "test-auth",
) -> PushSubscriptionDTO:
    return PushSubscriptionDTO(
        user_id=user_id, endpoint=endpoint, p256dh=p256dh, auth=auth
    )


def _make_payload(**kwargs) -> NotificationPayload:
    defaults = {"type": "agent_run", "event": "completed"}
    defaults.update(kwargs)
    return NotificationPayload(**defaults)


class TestBuildPushPayload:
    def test_includes_type_and_event(self):
        payload = _make_payload(type="agent_run", event="completed")

        result = push_sender._build_push_payload(payload)

        import json

        parsed = json.loads(result)
        assert parsed["type"] == "agent_run"
        assert parsed["event"] == "completed"

    def test_forwards_known_fields(self):
        payload = _make_payload(
            execution_id="exec-1",
            graph_id="graph-1",
            status="completed",
        )

        result = push_sender._build_push_payload(payload)

        import json

        parsed = json.loads(result)
        assert parsed["execution_id"] == "exec-1"
        assert parsed["graph_id"] == "graph-1"
        assert parsed["status"] == "completed"

    def test_excludes_unknown_fields(self):
        payload = _make_payload(
            custom_field="should-not-appear",
        )

        result = push_sender._build_push_payload(payload)

        import json

        parsed = json.loads(result)
        assert "custom_field" not in parsed

    def test_uses_model_dump_json_mode(self):
        """Ensure model_dump(mode='json') serializes enums to strings."""
        payload = _make_payload(type="agent_run", event="completed")

        result = push_sender._build_push_payload(payload)

        import json

        parsed = json.loads(result)
        assert isinstance(parsed["type"], str)
        assert isinstance(parsed["event"], str)

    def test_includes_unique_id_per_call(self):
        """Each push gets a fresh UUID so repeats don't collapse under the same SW tag."""
        import json

        payload = _make_payload(type="agent_run", event="completed")

        first = json.loads(push_sender._build_push_payload(payload))
        second = json.loads(push_sender._build_push_payload(payload))

        assert "id" in first and "id" in second
        assert first["id"] != second["id"]


class TestSendPushForUser:
    @pytest.mark.asyncio
    async def test_skips_when_vapid_private_key_missing(self, mocker, mock_db_client):
        mocker.patch.object(push_sender, "_settings", _make_settings(private=""))

        await push_sender.send_push_for_user("user-1", _make_payload())

        mock_db_client.get_user_push_subscriptions.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_vapid_public_key_missing(self, mocker, mock_db_client):
        mocker.patch.object(push_sender, "_settings", _make_settings(public=""))

        await push_sender.send_push_for_user("user-1", _make_payload())

        mock_db_client.get_user_push_subscriptions.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_vapid_email_missing(self, mocker, mock_db_client):
        mocker.patch.object(push_sender, "_settings", _make_settings(email=""))

        await push_sender.send_push_for_user("user-1", _make_payload())

        mock_db_client.get_user_push_subscriptions.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refuses_to_send_with_the_published_vapid_key(
        self, mocker, mock_db_client, caplog
    ):
        published = "pretend-this-was-shipped-in-env-default"
        mocker.patch.object(
            secrets_guard,
            "_RETIRED_VAPID_PRIVATE_KEY_DIGESTS",
            {
                hashlib.sha256(
                    secrets_guard._canonical_vapid_key(published).encode()
                ).hexdigest()
            },
        )
        mocker.patch.object(push_sender, "_settings", _make_settings(private=published))
        mock_db_client.get_user_push_subscriptions.return_value = [_make_subscription()]
        mock_webpush = mocker.patch("backend.data.push_sender.webpush")

        await push_sender.send_push_for_user("user-1", _make_payload())
        await push_sender.send_push_for_user("user-2", _make_payload())

        mock_db_client.get_user_push_subscriptions.assert_not_awaited()
        mock_webpush.assert_not_called()
        # Every notification is refused, but the process says why only once:
        # startup already warned, and this runs before the per-user debounce.
        warnings = [r for r in caplog.records if "make init-env" in r.getMessage()]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_debounces_rapid_calls(self, mocker, mock_db_client):
        mocker.patch.object(push_sender, "_settings", _make_settings())

        await push_sender.send_push_for_user("user-1", _make_payload())
        assert mock_db_client.get_user_push_subscriptions.await_count == 1

        await push_sender.send_push_for_user("user-1", _make_payload())
        assert mock_db_client.get_user_push_subscriptions.await_count == 1

    @pytest.mark.asyncio
    async def test_different_users_not_debounced(self, mocker, mock_db_client):
        mocker.patch.object(push_sender, "_settings", _make_settings())

        await push_sender.send_push_for_user("user-1", _make_payload())
        await push_sender.send_push_for_user("user-2", _make_payload())

        assert mock_db_client.get_user_push_subscriptions.await_count == 2

    @pytest.mark.asyncio
    async def test_returns_early_when_no_subscriptions(self, mocker, mock_db_client):
        mocker.patch.object(push_sender, "_settings", _make_settings())
        mock_webpush = mocker.patch("backend.data.push_sender.webpush")

        await push_sender.send_push_for_user("user-1", _make_payload())

        mock_webpush.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_webpush_for_each_subscription(self, mocker, mock_db_client):
        mocker.patch.object(push_sender, "_settings", _make_settings())
        sub1 = _make_subscription(endpoint="https://fcm.googleapis.com/fcm/send/sub/1")
        sub2 = _make_subscription(endpoint="https://fcm.googleapis.com/fcm/send/sub/2")
        mock_db_client.get_user_push_subscriptions.return_value = [sub1, sub2]
        mock_webpush = mocker.patch("backend.data.push_sender.webpush")

        await push_sender.send_push_for_user("user-1", _make_payload())

        assert mock_webpush.call_count == 2

        calls = mock_webpush.call_args_list
        endpoints_called = [c.kwargs["subscription_info"]["endpoint"] for c in calls]
        assert "https://fcm.googleapis.com/fcm/send/sub/1" in endpoints_called
        assert "https://fcm.googleapis.com/fcm/send/sub/2" in endpoints_called

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://fcm.googleapis.com/fcm/send/sub/1",
            "https://jmt17.google.com/fcm/send/sub/1",
            "https://wns2-par02p.notify.windows.com/w/?token=sub1",
        ],
    )
    async def test_webpush_called_with_correct_args(
        self, mocker, mock_db_client, endpoint
    ):
        mocker.patch.object(push_sender, "_settings", _make_settings())
        sub = _make_subscription(
            user_id="user-1",
            endpoint=endpoint,
            p256dh="key-p256dh",
            auth="key-auth",
        )
        mock_db_client.get_user_push_subscriptions.return_value = [sub]
        mock_webpush = mocker.patch("backend.data.push_sender.webpush")

        await push_sender.send_push_for_user("user-1", _make_payload())

        mock_webpush.assert_called_once()
        call_kwargs = mock_webpush.call_args.kwargs
        assert call_kwargs["subscription_info"] == {
            "endpoint": endpoint,
            "keys": {"p256dh": "key-p256dh", "auth": "key-auth"},
        }
        assert call_kwargs["vapid_private_key"] == "vapid-private"
        assert call_kwargs["vapid_claims"] == {"sub": "mailto:push@agpt.co"}
        assert isinstance(call_kwargs["data"], str)
        mock_db_client.delete_push_subscription.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_signs_each_request_for_its_own_push_service(
        self, mocker, mock_db_client
    ):
        # Real pywebpush with only its HTTP transport faked: pywebpush writes `aud`
        # into the claims dict it is given, which a mocked `webpush` never does.
        mocker.patch.object(
            push_sender, "_settings", _make_settings(private=_make_vapid_private_key())
        )
        mocker.patch.object(push_sender, "validate_push_endpoint", AsyncMock())
        chrome = "https://fcm.googleapis.com/fcm/send/made-up-chrome"
        safari = "https://web.push.apple.com/made-up-safari"
        mock_db_client.get_user_push_subscriptions.return_value = [
            _make_browser_subscription(chrome),
            _make_browser_subscription(safari),
        ]
        transport = MagicMock()
        transport.post.return_value = MagicMock(status_code=201)
        mocker.patch.object(pywebpush, "requests", transport)

        await push_sender.send_push_for_user("user-1", _make_payload())

        audiences = {
            call.args[0]: _jwt_claims(call.kwargs["headers"]["Authorization"])["aud"]
            for call in transport.post.call_args_list
        }
        assert audiences == {
            chrome: "https://fcm.googleapis.com",
            safari: "https://web.push.apple.com",
        }

    @pytest.mark.asyncio
    async def test_removes_subscription_on_410_gone(self, mocker, mock_db_client):
        from pywebpush import WebPushException

        mocker.patch.object(push_sender, "_settings", _make_settings())
        sub = _make_subscription()
        mock_db_client.get_user_push_subscriptions.return_value = [sub]

        mock_response = MagicMock()
        mock_response.status_code = 410
        exc = WebPushException("Gone", response=mock_response)
        mocker.patch("backend.data.push_sender.webpush", side_effect=exc)

        await push_sender.send_push_for_user("user-1", _make_payload())

        mock_db_client.delete_push_subscription.assert_awaited_once_with(
            sub.user_id, sub.endpoint
        )

    @pytest.mark.asyncio
    async def test_removes_subscription_on_404(self, mocker, mock_db_client):
        from pywebpush import WebPushException

        mocker.patch.object(push_sender, "_settings", _make_settings())
        sub = _make_subscription()
        mock_db_client.get_user_push_subscriptions.return_value = [sub]

        mock_response = MagicMock()
        mock_response.status_code = 404
        exc = WebPushException("Not Found", response=mock_response)
        mocker.patch("backend.data.push_sender.webpush", side_effect=exc)

        await push_sender.send_push_for_user("user-1", _make_payload())

        mock_db_client.delete_push_subscription.assert_awaited_once_with(
            sub.user_id, sub.endpoint
        )

    @pytest.mark.asyncio
    async def test_removes_subscription_when_status_only_in_message(
        self, mocker, mock_db_client
    ):
        """Some pywebpush versions don't expose .response.status_code; the
        sender must still detect 410 from the exception message and clean up."""
        from pywebpush import WebPushException

        mocker.patch.object(push_sender, "_settings", _make_settings())
        sub = _make_subscription()
        mock_db_client.get_user_push_subscriptions.return_value = [sub]

        # No usable response object — only the message carries the status.
        exc = WebPushException("Push failed: 410 Gone\nResponse body:gone")
        exc.response = None  # type: ignore[assignment]
        mocker.patch("backend.data.push_sender.webpush", side_effect=exc)

        await push_sender.send_push_for_user("user-1", _make_payload())

        mock_db_client.delete_push_subscription.assert_awaited_once_with(
            sub.user_id, sub.endpoint
        )

    @pytest.mark.asyncio
    async def test_increments_fail_count_on_other_webpush_error(
        self, mocker, mock_db_client
    ):
        from pywebpush import WebPushException

        mocker.patch.object(push_sender, "_settings", _make_settings())
        sub = _make_subscription()
        mock_db_client.get_user_push_subscriptions.return_value = [sub]

        mock_response = MagicMock()
        mock_response.status_code = 429
        exc = WebPushException("Too Many Requests", response=mock_response)
        mocker.patch("backend.data.push_sender.webpush", side_effect=exc)

        await push_sender.send_push_for_user("user-1", _make_payload())

        mock_db_client.increment_push_fail_count.assert_awaited_once_with(
            sub.user_id, sub.endpoint
        )

    @pytest.mark.asyncio
    async def test_increments_fail_count_when_no_response_object(
        self, mocker, mock_db_client
    ):
        from pywebpush import WebPushException

        mocker.patch.object(push_sender, "_settings", _make_settings())
        sub = _make_subscription()
        mock_db_client.get_user_push_subscriptions.return_value = [sub]

        exc = WebPushException("Connection error")
        mocker.patch("backend.data.push_sender.webpush", side_effect=exc)

        await push_sender.send_push_for_user("user-1", _make_payload())

        mock_db_client.increment_push_fail_count.assert_awaited_once_with(
            sub.user_id, sub.endpoint
        )

    @pytest.mark.asyncio
    async def test_handles_unexpected_exception_gracefully(
        self, mocker, mock_db_client
    ):
        mocker.patch.object(push_sender, "_settings", _make_settings())
        sub = _make_subscription()
        mock_db_client.get_user_push_subscriptions.return_value = [sub]
        mocker.patch(
            "backend.data.push_sender.webpush",
            side_effect=RuntimeError("network down"),
        )

        await push_sender.send_push_for_user("user-1", _make_payload())

    @pytest.mark.asyncio
    async def test_debounce_expires_after_threshold(self, mocker, mock_db_client):
        mocker.patch.object(push_sender, "_settings", _make_settings())

        await push_sender.send_push_for_user("user-1", _make_payload())
        assert mock_db_client.get_user_push_subscriptions.await_count == 1

        # Simulate TTL expiry (cachetools evicts on access after TTL elapses).
        push_sender._user_last_push.pop("user-1", None)

        await push_sender.send_push_for_user("user-1", _make_payload())
        assert mock_db_client.get_user_push_subscriptions.await_count == 2


def _make_vapid_private_key() -> str:
    """A raw P-256 scalar in base64url, the form `make init-env` writes."""
    key = ec.generate_private_key(ec.SECP256R1())
    return _b64url(key.private_numbers().private_value.to_bytes(32, "big"))


def _make_browser_subscription(endpoint: str) -> PushSubscriptionDTO:
    """A subscription with real browser keys, so pywebpush can encrypt to it."""
    public_point = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
        )
    )
    return _make_subscription(
        endpoint=endpoint, p256dh=_b64url(public_point), auth=_b64url(os.urandom(16))
    )


def _jwt_claims(authorization: str) -> dict:
    token = authorization.removeprefix("vapid t=").split(",")[0]
    claims = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(claims + "=" * (-len(claims) % 4)))


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
