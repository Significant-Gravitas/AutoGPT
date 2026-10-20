"""Tests for Web Push subscription CRUD operations."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from prisma.errors import RecordNotFoundError, UniqueViolationError

from backend.data import push_subscription


@pytest.fixture
def mock_prisma(mocker):
    """Mock PushSubscription.prisma() and return the mock client."""
    mock_client = MagicMock()
    mock_client.upsert = AsyncMock()
    mock_client.find_many = AsyncMock(return_value=[])
    mock_client.find_unique = AsyncMock(return_value=None)
    mock_client.delete_many = AsyncMock()
    mock_client.update_many = AsyncMock()
    mock_client.tx = MagicMock(name="tx")

    @asynccontextmanager
    async def fake_transaction():
        yield mock_client.tx

    mocker.patch("backend.data.push_subscription.transaction", fake_transaction)
    mock_client.model_prisma = mocker.patch(
        "backend.data.push_subscription.PushSubscription.prisma",
        return_value=mock_client,
    )
    return mock_client


class TestUpsertPushSubscription:
    @pytest.mark.asyncio
    async def test_calls_prisma_upsert_with_correct_params(self, mock_prisma):
        mock_prisma.upsert.return_value = MagicMock()

        await push_subscription.upsert_push_subscription(
            user_id="user-1",
            endpoint="https://fcm.googleapis.com/fcm/send/sub/1",
            p256dh="test-p256dh",
            auth="test-auth",
            user_agent="Mozilla/5.0",
        )

        mock_prisma.upsert.assert_awaited_once()
        call_kwargs = mock_prisma.upsert.call_args.kwargs
        assert call_kwargs["where"] == {
            "endpoint": "https://fcm.googleapis.com/fcm/send/sub/1"
        }
        assert call_kwargs["data"]["create"] == {
            "userId": "user-1",
            "endpoint": "https://fcm.googleapis.com/fcm/send/sub/1",
            "p256dh": "test-p256dh",
            "auth": "test-auth",
            "userAgent": "Mozilla/5.0",
        }
        assert call_kwargs["data"]["update"] == {
            "User": {"connect": {"id": "user-1"}},
            "p256dh": "test-p256dh",
            "auth": "test-auth",
            "userAgent": "Mozilla/5.0",
            "failCount": 0,
            "lastFailedAt": None,
        }

    @pytest.mark.asyncio
    async def test_upsert_without_user_agent(self, mock_prisma):
        mock_prisma.upsert.return_value = MagicMock()

        await push_subscription.upsert_push_subscription(
            user_id="user-1",
            endpoint="https://fcm.googleapis.com/fcm/send/sub/1",
            p256dh="test-p256dh",
            auth="test-auth",
        )

        call_kwargs = mock_prisma.upsert.call_args.kwargs
        assert call_kwargs["data"]["create"]["userAgent"] is None
        assert call_kwargs["data"]["update"]["userAgent"] is None

    @pytest.mark.asyncio
    async def test_upsert_returns_prisma_result(self, mock_prisma):
        expected = MagicMock()
        mock_prisma.upsert.return_value = expected

        result = await push_subscription.upsert_push_subscription(
            user_id="user-1",
            endpoint="https://fcm.googleapis.com/fcm/send/sub/1",
            p256dh="test-p256dh",
            auth="test-auth",
        )

        assert result is expected

    @pytest.mark.asyncio
    async def test_upsert_resets_fail_count_on_update(self, mock_prisma):
        mock_prisma.upsert.return_value = MagicMock()

        await push_subscription.upsert_push_subscription(
            user_id="user-1",
            endpoint="https://fcm.googleapis.com/fcm/send/sub/1",
            p256dh="test-p256dh",
            auth="test-auth",
        )

        call_kwargs = mock_prisma.upsert.call_args.kwargs
        assert call_kwargs["data"]["update"]["failCount"] == 0
        assert call_kwargs["data"]["update"]["lastFailedAt"] is None

    @pytest.mark.asyncio
    async def test_rejects_new_endpoint_past_cap(self, mock_prisma):
        existing = [
            MagicMock(endpoint=f"https://fcm.googleapis.com/fcm/send/sub/{i}")
            for i in range(push_subscription.MAX_SUBSCRIPTIONS_PER_USER)
        ]
        mock_prisma.find_many.return_value = existing

        with pytest.raises(ValueError, match="Subscription limit"):
            await push_subscription.upsert_push_subscription(
                user_id="user-1",
                endpoint="https://fcm.googleapis.com/fcm/send/sub/NEW",
                p256dh="test-p256dh",
                auth="test-auth",
            )

        mock_prisma.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allows_update_of_existing_endpoint_at_cap(self, mock_prisma):
        existing = [
            MagicMock(endpoint=f"https://fcm.googleapis.com/fcm/send/sub/{i}")
            for i in range(push_subscription.MAX_SUBSCRIPTIONS_PER_USER)
        ]
        mock_prisma.find_many.return_value = existing
        mock_prisma.upsert.return_value = MagicMock()

        await push_subscription.upsert_push_subscription(
            user_id="user-1",
            endpoint="https://fcm.googleapis.com/fcm/send/sub/0",
            p256dh="rotated-p256dh",
            auth="rotated-auth",
        )

        mock_prisma.upsert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_takes_over_another_users_endpoint_at_the_cap(self, mock_prisma):
        # Refusing the takeover would keep the previous account's
        # notifications arriving on this browser, so the user's least
        # recently refreshed subscription makes room and they stay at the cap.
        existing = [
            MagicMock(
                id=f"row-{i}",
                endpoint=f"https://fcm.googleapis.com/fcm/send/sub/{i}",
                updatedAt=datetime(2026, 10, 1 + i, tzinfo=timezone.utc),
                failCount=0,
            )
            for i in range(push_subscription.MAX_SUBSCRIPTIONS_PER_USER)
        ]
        existing[0], existing[5] = existing[5], existing[0]
        mock_prisma.find_many.return_value = existing
        mock_prisma.find_unique.return_value = MagicMock(userId="user-1")
        writes: list[str] = []
        mock_prisma.delete_many.side_effect = lambda **_: writes.append("delete")
        mock_prisma.upsert.side_effect = lambda **_: writes.append("upsert")

        await push_subscription.upsert_push_subscription(
            user_id="user-2",
            endpoint="https://fcm.googleapis.com/fcm/send/sub/TAKEN",
            p256dh="test-p256dh",
            auth="test-auth",
        )

        mock_prisma.find_unique.assert_awaited_once_with(
            where={"endpoint": "https://fcm.googleapis.com/fcm/send/sub/TAKEN"}
        )
        # Exactly one of the user's rows, the stalest, goes before the one
        # taken over is added, so they end at the cap rather than one past it.
        mock_prisma.delete_many.assert_awaited_once_with(where={"id": "row-0"})
        assert writes == ["delete", "upsert"]

    @pytest.mark.asyncio
    async def test_takeover_at_the_cap_drops_a_failing_device_first(self, mock_prisma):
        # A failing push bumps updatedAt, so a dying device looks freshly
        # refreshed; it must still go before a healthy idle one.
        existing = [
            MagicMock(
                id=f"row-{i}",
                endpoint=f"https://fcm.googleapis.com/fcm/send/sub/{i}",
                updatedAt=datetime(2026, 10, 1 + i, tzinfo=timezone.utc),
                failCount=0,
            )
            for i in range(push_subscription.MAX_SUBSCRIPTIONS_PER_USER)
        ]
        existing[-1].failCount = 3
        mock_prisma.find_many.return_value = existing
        mock_prisma.find_unique.return_value = MagicMock(userId="user-1")
        mock_prisma.upsert.return_value = MagicMock()

        await push_subscription.upsert_push_subscription(
            user_id="user-2",
            endpoint="https://fcm.googleapis.com/fcm/send/sub/TAKEN",
            p256dh="test-p256dh",
            auth="test-auth",
        )

        mock_prisma.delete_many.assert_awaited_once_with(
            where={"id": f"row-{push_subscription.MAX_SUBSCRIPTIONS_PER_USER - 1}"}
        )

    @pytest.mark.asyncio
    async def test_cap_check_eviction_and_claim_share_one_transaction(
        self, mock_prisma
    ):
        # A claim that fails after the eviction must roll the eviction back,
        # or the account loses a registration for nothing.
        mock_prisma.find_many.return_value = [
            MagicMock(
                id=f"row-{i}",
                endpoint=f"https://fcm.googleapis.com/fcm/send/sub/{i}",
                updatedAt=datetime(2026, 10, 1 + i, tzinfo=timezone.utc),
                failCount=0,
            )
            for i in range(push_subscription.MAX_SUBSCRIPTIONS_PER_USER)
        ]
        mock_prisma.find_unique.return_value = MagicMock(userId="user-1")
        mock_prisma.upsert.return_value = MagicMock()

        await push_subscription.upsert_push_subscription(
            user_id="user-2",
            endpoint="https://fcm.googleapis.com/fcm/send/sub/TAKEN",
            p256dh="test-p256dh",
            auth="test-auth",
        )

        assert mock_prisma.model_prisma.call_args_list == [call(mock_prisma.tx)]
        mock_prisma.delete_many.assert_awaited_once()
        mock_prisma.upsert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_once_when_the_row_vanished_mid_upsert(self, mock_prisma):
        # The failed-push cleanup can delete the row between Prisma's read
        # and write; the second run creates it.
        expected = MagicMock()
        mock_prisma.upsert.side_effect = [
            RecordNotFoundError(MagicMock()),
            expected,
        ]

        result = await push_subscription.upsert_push_subscription(
            user_id="user-2",
            endpoint="https://fcm.googleapis.com/fcm/send/sub/1",
            p256dh="test-p256dh",
            auth="test-auth",
        )

        assert result is expected
        assert mock_prisma.upsert.await_count == 2

    @pytest.mark.asyncio
    async def test_refreshing_own_endpoint_at_the_cap_drops_nothing(self, mock_prisma):
        mock_prisma.find_many.return_value = [
            MagicMock(endpoint=f"https://fcm.googleapis.com/fcm/send/sub/{i}")
            for i in range(push_subscription.MAX_SUBSCRIPTIONS_PER_USER)
        ]
        mock_prisma.upsert.return_value = MagicMock()

        await push_subscription.upsert_push_subscription(
            user_id="user-1",
            endpoint="https://fcm.googleapis.com/fcm/send/sub/3",
            p256dh="test-p256dh",
            auth="test-auth",
        )

        mock_prisma.find_unique.assert_not_awaited()
        mock_prisma.delete_many.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_moves_the_endpoint_to_the_subscribing_user(self, mock_prisma):
        # A browser has one push endpoint: after switching accounts on it, the
        # previous account must stop receiving pushes there. The row is keyed
        # by endpoint alone, so the upsert hands it to the new user.
        mock_prisma.upsert.return_value = MagicMock()

        await push_subscription.upsert_push_subscription(
            user_id="user-2",
            endpoint="https://fcm.googleapis.com/fcm/send/sub/1",
            p256dh="test-p256dh",
            auth="test-auth",
        )

        call_kwargs = mock_prisma.upsert.call_args.kwargs
        assert call_kwargs["where"] == {
            "endpoint": "https://fcm.googleapis.com/fcm/send/sub/1"
        }
        assert call_kwargs["data"]["update"]["User"] == {"connect": {"id": "user-2"}}
        mock_prisma.delete_many.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retries_once_when_a_concurrent_subscribe_created_the_row(
        self, mock_prisma
    ):
        expected = MagicMock()
        mock_prisma.upsert.side_effect = [
            UniqueViolationError(MagicMock()),
            expected,
        ]

        result = await push_subscription.upsert_push_subscription(
            user_id="user-2",
            endpoint="https://fcm.googleapis.com/fcm/send/sub/1",
            p256dh="test-p256dh",
            auth="test-auth",
        )

        assert result is expected
        assert mock_prisma.upsert.await_count == 2


class TestGetUserPushSubscriptions:
    @pytest.mark.asyncio
    async def test_returns_list_of_subscription_dtos(self, mock_prisma):
        sub1 = MagicMock(
            userId="user-1",
            endpoint="https://fcm.googleapis.com/fcm/send/sub/1",
            p256dh="key1",
            auth="auth1",
        )
        sub2 = MagicMock(
            userId="user-1",
            endpoint="https://fcm.googleapis.com/fcm/send/sub/2",
            p256dh="key2",
            auth="auth2",
        )
        mock_prisma.find_many.return_value = [sub1, sub2]

        result = await push_subscription.get_user_push_subscriptions("user-1")

        assert [r.endpoint for r in result] == [
            "https://fcm.googleapis.com/fcm/send/sub/1",
            "https://fcm.googleapis.com/fcm/send/sub/2",
        ]
        assert all(r.user_id == "user-1" for r in result)
        mock_prisma.find_many.assert_awaited_once_with(where={"userId": "user-1"})

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_subscriptions(self, mock_prisma):
        mock_prisma.find_many.return_value = []

        result = await push_subscription.get_user_push_subscriptions("user-1")

        assert result == []


class TestDeletePushSubscription:
    @pytest.mark.asyncio
    async def test_deletes_by_user_id_and_endpoint(self, mock_prisma):
        await push_subscription.delete_push_subscription(
            "user-1",
            "https://fcm.googleapis.com/fcm/send/sub/1",
        )

        mock_prisma.delete_many.assert_awaited_once_with(
            where={
                "userId": "user-1",
                "endpoint": "https://fcm.googleapis.com/fcm/send/sub/1",
            }
        )


class TestIncrementFailCount:
    @pytest.mark.asyncio
    async def test_includes_user_id_in_where(self, mock_prisma):
        await push_subscription.increment_fail_count(
            "user-1",
            "https://fcm.googleapis.com/fcm/send/sub/1",
        )

        mock_prisma.update_many.assert_awaited_once()
        call_kwargs = mock_prisma.update_many.call_args.kwargs
        assert call_kwargs["where"] == {
            "userId": "user-1",
            "endpoint": "https://fcm.googleapis.com/fcm/send/sub/1",
        }

    @pytest.mark.asyncio
    async def test_increments_fail_count_by_one(self, mock_prisma):
        await push_subscription.increment_fail_count(
            "user-1",
            "https://fcm.googleapis.com/fcm/send/sub/1",
        )

        call_kwargs = mock_prisma.update_many.call_args.kwargs
        assert call_kwargs["data"]["failCount"] == {"increment": 1}

    @pytest.mark.asyncio
    async def test_sets_last_failed_at_to_utc_now(self, mock_prisma):
        await push_subscription.increment_fail_count(
            "user-1",
            "https://fcm.googleapis.com/fcm/send/sub/1",
        )

        call_kwargs = mock_prisma.update_many.call_args.kwargs
        last_failed = call_kwargs["data"]["lastFailedAt"]
        assert isinstance(last_failed, datetime)
        assert last_failed.tzinfo is not None


class TestCleanupFailedSubscriptions:
    @pytest.mark.asyncio
    async def test_deletes_subscriptions_exceeding_threshold(self, mock_prisma):
        mock_prisma.delete_many.return_value = 3

        result = await push_subscription.cleanup_failed_subscriptions(
            max_failures=5,
        )

        assert result == 3
        mock_prisma.delete_many.assert_awaited_once_with(
            where={"failCount": {"gte": 5}}
        )

    @pytest.mark.asyncio
    async def test_uses_default_max_failures(self, mock_prisma):
        mock_prisma.delete_many.return_value = 0

        await push_subscription.cleanup_failed_subscriptions()

        call_kwargs = mock_prisma.delete_many.call_args.kwargs
        assert call_kwargs["where"]["failCount"]["gte"] == 5

    @pytest.mark.asyncio
    async def test_returns_zero_when_none_deleted(self, mock_prisma):
        mock_prisma.delete_many.return_value = 0

        result = await push_subscription.cleanup_failed_subscriptions()

        assert result == 0

    @pytest.mark.asyncio
    async def test_returns_zero_when_result_is_none(self, mock_prisma):
        mock_prisma.delete_many.return_value = None

        result = await push_subscription.cleanup_failed_subscriptions()

        assert result == 0


class TestValidatePushEndpoint:
    """Endpoints from clients must land on a known Web Push service — otherwise
    the backend can be coerced into POSTing to internal hosts (SSRF)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://fcm.googleapis.com/fcm/send/abc",
            "https://jmt17.google.com/fcm/send/abc",
            "https://updates.push.services.mozilla.com/wpush/v2/xyz",
            "https://web.push.apple.com/some-token",
            "https://wns2-par02p.notify.windows.com/w/?token=abc",
        ],
    )
    async def test_allows_known_push_services(self, endpoint):
        await push_subscription.validate_push_endpoint(endpoint)

    @pytest.mark.asyncio
    async def test_rejects_http_scheme(self):
        with pytest.raises(ValueError):
            await push_subscription.validate_push_endpoint(
                "http://fcm.googleapis.com/fcm/send/abc"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://localhost/evil",
            "https://127.0.0.1/evil",
            "https://169.254.169.254/latest/meta-data/",
            "https://internal-service.local/api",
            "https://attacker.example.com/push",
        ],
    )
    async def test_rejects_untrusted_hosts(self, endpoint):
        with pytest.raises(ValueError):
            await push_subscription.validate_push_endpoint(endpoint)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://jmt17.google.com.evil.example/fcm/send/abc",
            "https://evil-jmt17.google.com/fcm/send/abc",
            "https://www.google.com/fcm/send/abc",
            "https://notify.windows.com.evil.example/w/?token=abc",
            "https://evilnotify.windows.com/w/?token=abc",
            "https://windows.com/w/?token=abc",
            "https://push.apple.com.evil.example/some-token",
            "https://evilpush.apple.com/some-token",
            "https://apple.com/some-token",
        ],
    )
    async def test_rejects_lookalike_hosts(self, mocker, endpoint):
        # Resolve publicly so the allowlist, not a DNS failure, does the refusing.
        mocker.patch(
            "backend.util.request._resolve_host",
            new_callable=AsyncMock,
            return_value=["8.8.8.8"],
        )
        with pytest.raises(ValueError, match="not a recognised Web Push service"):
            await push_subscription.validate_push_endpoint(endpoint)

    @pytest.mark.asyncio
    async def test_rejects_non_http_scheme(self):
        with pytest.raises(ValueError):
            await push_subscription.validate_push_endpoint("file:///etc/passwd")

    @pytest.mark.asyncio
    async def test_custom_max_failures_threshold(self, mock_prisma):
        mock_prisma.delete_many.return_value = 1

        result = await push_subscription.cleanup_failed_subscriptions(
            max_failures=10,
        )

        assert result == 1
        call_kwargs = mock_prisma.delete_many.call_args.kwargs
        assert call_kwargs["where"]["failCount"]["gte"] == 10
