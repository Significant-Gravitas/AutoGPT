"""Tests for Web Push subscription CRUD operations."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data import push_subscription


@pytest.fixture
def mock_prisma(mocker):
    """Mock PushSubscription.prisma() and return the mock client."""
    mock_client = MagicMock()
    mock_client.upsert = AsyncMock()
    mock_client.find_many = AsyncMock(return_value=[])
    mock_client.delete_many = AsyncMock()
    mock_client.update_many = AsyncMock()
    mocker.patch(
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
            "userId_endpoint": {
                "userId": "user-1",
                "endpoint": "https://fcm.googleapis.com/fcm/send/sub/1",
            }
        }
        assert call_kwargs["data"]["create"] == {
            "userId": "user-1",
            "endpoint": "https://fcm.googleapis.com/fcm/send/sub/1",
            "p256dh": "test-p256dh",
            "auth": "test-auth",
            "userAgent": "Mozilla/5.0",
        }
        assert call_kwargs["data"]["update"] == {
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
            "https://updates.push.services.mozilla.com/wpush/v2/xyz",
            "https://web.push.apple.com/some-token",
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
