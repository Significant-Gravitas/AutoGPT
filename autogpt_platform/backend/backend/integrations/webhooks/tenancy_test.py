import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

import pytest
from pydantic import SecretStr

from backend.data import integrations
from backend.data.model import APIKeyCredentials
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks import _base as webhooks_base
from backend.integrations.webhooks import telegram
from backend.integrations.webhooks.github import GithubWebhooksManager
from backend.integrations.webhooks.telegram import (
    TelegramWebhooksManager,
    TelegramWebhookType,
)
from backend.util.exceptions import WebhookRegistrationError


class _ImmediateMutex:
    def __init__(self, _redis: object) -> None:
        pass

    async def acquire(self, _key: object) -> None:
        pass

    async def release(self, _key: object) -> None:
        pass


@pytest.fixture(autouse=True)
def _mock_telegram_setup_mutex(monkeypatch) -> None:
    async def get_redis() -> object:
        return object()

    monkeypatch.setattr(telegram, "get_redis_async", get_redis)
    monkeypatch.setattr(telegram, "AsyncRedisKeyedMutex", _ImmediateMutex)


def _credentials(
    provider: ProviderName, credentials_id: str = "cred-1"
) -> APIKeyCredentials:
    return APIKeyCredentials(
        id=credentials_id,
        provider=provider.value,
        api_key=SecretStr("secret"),
    )


def _webhook(
    *,
    provider: ProviderName,
    webhook_type: str,
    resource: str,
    organization_id: str | None,
    team_id: str | None,
    events: list[str] | None = None,
    webhook_id: str = "webhook-1",
) -> integrations.Webhook:
    return integrations.Webhook(
        id=webhook_id,
        user_id="user-1",
        provider=provider,
        credentials_id="cred-1",
        webhook_type=webhook_type,
        resource=resource,
        events=events or ["push"],
        config={},
        secret="secret",
        provider_webhook_id="provider-webhook-1",
        organization_id=organization_id,
        team_id=team_id,
    )


@pytest.mark.asyncio
async def test_auto_webhook_does_not_reuse_mismatched_tenant(monkeypatch) -> None:
    monkeypatch.setattr(
        webhooks_base.app_config, "platform_base_url", "https://example.com"
    )
    manager = GithubWebhooksManager()
    mismatched = _webhook(
        provider=ProviderName.GITHUB,
        webhook_type=manager.WebhookType.REPO,
        resource="owner/repo",
        organization_id="other-org",
        team_id="other-team",
    )
    replacement = mismatched.model_copy(
        update={
            "id": "webhook-2",
            "organization_id": "org-1",
            "team_id": "team-1",
        }
    )
    find_webhook = AsyncMock(return_value=mismatched)
    create_webhook = AsyncMock(return_value=replacement)

    with (
        patch.object(
            integrations,
            "find_webhook_by_credentials_and_props",
            find_webhook,
        ),
        patch.object(manager, "_create_webhook", create_webhook),
    ):
        result = await manager.get_suitable_auto_webhook(
            user_id="user-1",
            credentials=_credentials(ProviderName.GITHUB),
            webhook_type=manager.WebhookType.REPO,
            resource="owner/repo",
            events=["push"],
            organization_id="org-1",
            team_id="team-1",
        )

    assert result == replacement
    find_webhook.assert_awaited_once_with(
        user_id="user-1",
        credentials_id="cred-1",
        webhook_type=manager.WebhookType.REPO,
        resource="owner/repo",
        organization_id="org-1",
        team_id="team-1",
        events=["push"],
    )
    create_webhook.assert_awaited_once()


@pytest.mark.asyncio
async def test_manual_webhook_does_not_update_mismatched_tenant(monkeypatch) -> None:
    monkeypatch.setattr(
        webhooks_base.app_config, "platform_base_url", "https://example.com"
    )
    manager = GithubWebhooksManager()
    mismatched = _webhook(
        provider=ProviderName.GITHUB,
        webhook_type=manager.WebhookType.REPO,
        resource="",
        events=["issues"],
        organization_id="other-org",
        team_id="other-team",
    )
    replacement = mismatched.model_copy(
        update={
            "id": "webhook-2",
            "events": ["push"],
            "organization_id": "org-1",
            "team_id": "team-1",
        }
    )
    find_webhook = AsyncMock(return_value=mismatched)
    update_webhook = AsyncMock()
    create_webhook = AsyncMock(return_value=replacement)

    with (
        patch.object(
            integrations,
            "find_webhook_by_graph_and_props",
            find_webhook,
        ),
        patch.object(integrations, "update_webhook", update_webhook),
        patch.object(manager, "_create_webhook", create_webhook),
    ):
        result = await manager.get_manual_webhook(
            user_id="user-1",
            webhook_type=manager.WebhookType.REPO,
            events=["push"],
            preset_id="preset-1",
            organization_id="org-1",
            team_id="team-1",
        )

    assert result == replacement
    find_webhook.assert_awaited_once_with(
        user_id="user-1",
        provider=ProviderName.GITHUB.value,
        webhook_type=manager.WebhookType.REPO,
        organization_id="org-1",
        team_id="team-1",
        graph_id=None,
        preset_id="preset-1",
    )
    update_webhook.assert_not_awaited()
    create_webhook.assert_awaited_once()


@pytest.mark.asyncio
async def test_telegram_rejects_cross_tenant_bot_before_provider_mutation() -> None:
    manager = TelegramWebhooksManager()
    mismatched = _webhook(
        provider=ProviderName.TELEGRAM,
        webhook_type=TelegramWebhookType.BOT,
        resource="",
        events=["message.text"],
        organization_id="other-org",
        team_id="other-team",
    )
    find_same_tenant = AsyncMock(return_value=None)
    find_any_tenant = AsyncMock(return_value=mismatched)
    register_webhook = AsyncMock()
    update_webhook = AsyncMock()
    create_webhook = AsyncMock()

    with (
        patch.object(
            telegram,
            "Config",
            return_value=SimpleNamespace(platform_base_url="https://example.com"),
        ),
        patch.object(
            integrations,
            "find_webhook_by_credentials_and_props",
            find_same_tenant,
        ),
        patch.object(
            integrations,
            "find_webhook_by_credentials_and_props_any_tenant",
            find_any_tenant,
        ),
        patch.object(manager, "_register_webhook", register_webhook),
        patch.object(integrations, "update_webhook", update_webhook),
        patch.object(manager, "_create_webhook", create_webhook),
    ):
        with pytest.raises(WebhookRegistrationError, match="another tenancy"):
            await manager.get_suitable_auto_webhook(
                user_id="user-1",
                credentials=_credentials(ProviderName.TELEGRAM),
                webhook_type=TelegramWebhookType.BOT,
                resource="",
                events=["message.text"],
                organization_id="org-1",
                team_id="team-1",
            )

    assert find_same_tenant.await_args_list == [
        call(
            user_id="user-1",
            credentials_id="cred-1",
            webhook_type=TelegramWebhookType.BOT,
            resource="",
            organization_id="org-1",
            team_id="team-1",
            events=["message.text"],
        ),
        call(
            user_id="user-1",
            credentials_id="cred-1",
            webhook_type=TelegramWebhookType.BOT,
            resource="",
            organization_id="org-1",
            team_id="team-1",
        ),
    ]
    find_any_tenant.assert_awaited_once_with(
        user_id="user-1",
        credentials_id="cred-1",
        webhook_type=TelegramWebhookType.BOT,
        resource="",
    )
    register_webhook.assert_not_awaited()
    update_webhook.assert_not_awaited()
    create_webhook.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_updates_same_tenant_before_cross_tenant_lookup() -> None:
    manager = TelegramWebhooksManager()
    same_tenant = _webhook(
        provider=ProviderName.TELEGRAM,
        webhook_type=TelegramWebhookType.BOT,
        resource="",
        events=["message.photo"],
        organization_id="org-1",
        team_id="team-1",
    )
    cross_tenant = same_tenant.model_copy(
        update={
            "id": "webhook-cross-tenant",
            "organization_id": "other-org",
            "team_id": "other-team",
        }
    )
    updated = same_tenant.model_copy(update={"events": ["message.text"]})
    find_same_tenant = AsyncMock(side_effect=[None, same_tenant])
    find_any_tenant = AsyncMock(return_value=cross_tenant)
    register_webhook = AsyncMock(return_value=("", {"allowed_updates": ["message"]}))
    update_webhook = AsyncMock(return_value=updated)

    with (
        patch.object(
            telegram,
            "Config",
            return_value=SimpleNamespace(platform_base_url="https://example.com"),
        ),
        patch.object(
            integrations,
            "find_webhook_by_credentials_and_props",
            find_same_tenant,
        ),
        patch.object(
            integrations,
            "find_webhook_by_credentials_and_props_any_tenant",
            find_any_tenant,
        ),
        patch.object(manager, "_register_webhook", register_webhook),
        patch.object(integrations, "update_webhook", update_webhook),
    ):
        result = await manager.get_suitable_auto_webhook(
            user_id="user-1",
            credentials=_credentials(ProviderName.TELEGRAM),
            webhook_type=TelegramWebhookType.BOT,
            resource="",
            events=["message.text"],
            organization_id="org-1",
            team_id="team-1",
        )

    assert result == updated
    find_any_tenant.assert_not_awaited()
    register_webhook.assert_awaited_once()
    update_webhook.assert_awaited_once_with(
        same_tenant.id,
        events=["message.text"],
        config={"allowed_updates": ["message"]},
    )


@pytest.mark.asyncio
async def test_concurrent_telegram_setup_allows_only_one_tenancy(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        webhooks_base.app_config, "platform_base_url", "https://example.com"
    )
    winner_manager = TelegramWebhooksManager()
    loser_manager = TelegramWebhooksManager()
    credentials = _credentials(ProviderName.TELEGRAM)
    created_webhooks: list[integrations.Webhook] = []
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()
    distributed_lock = asyncio.Lock()
    acquired_keys: list[object] = []
    released_keys: list[object] = []
    mutex_instances: list[object] = []

    class SerializedMutex:
        def __init__(self, _redis: object) -> None:
            mutex_instances.append(self)

        async def acquire(self, key: object) -> None:
            acquired_keys.append(key)
            await distributed_lock.acquire()

        async def release(self, key: object) -> None:
            released_keys.append(key)
            distributed_lock.release()

    redis_mock = AsyncMock(return_value=object())
    monkeypatch.setattr(telegram, "get_redis_async", redis_mock)
    monkeypatch.setattr(telegram, "AsyncRedisKeyedMutex", SerializedMutex)

    async def find_same_tenant(
        *,
        user_id: str,
        credentials_id: str,
        webhook_type: str,
        resource: str,
        organization_id: str | None,
        team_id: str | None,
        events: list[str] | None = None,
    ) -> integrations.Webhook | None:
        for webhook in created_webhooks:
            if (
                webhook.user_id == user_id
                and webhook.credentials_id == credentials_id
                and webhook.webhook_type == webhook_type
                and webhook.resource == resource
                and webhook.organization_id == organization_id
                and webhook.team_id == team_id
                and (events is None or set(events).issubset(webhook.events))
            ):
                return webhook
        return None

    async def find_any_tenant(
        *,
        user_id: str,
        credentials_id: str,
        webhook_type: str,
        resource: str,
    ) -> integrations.Webhook | None:
        return next(
            (
                webhook
                for webhook in created_webhooks
                if webhook.user_id == user_id
                and webhook.credentials_id == credentials_id
                and webhook.webhook_type == webhook_type
                and webhook.resource == resource
            ),
            None,
        )

    async def register_webhook(
        _credentials: APIKeyCredentials,
        _webhook_type: TelegramWebhookType,
        _resource: str,
        _events: list[str],
        _ingress_url: str,
        _secret: str,
    ) -> tuple[str, dict]:
        provider_started.set()
        await release_provider.wait()
        return "provider-webhook-1", {}

    async def create_webhook(webhook: integrations.Webhook) -> integrations.Webhook:
        created_webhooks.append(webhook)
        return webhook

    register_mock = AsyncMock(side_effect=register_webhook)
    create_mock = AsyncMock(side_effect=create_webhook)
    update_mock = AsyncMock()

    with (
        patch.object(
            telegram,
            "Config",
            return_value=SimpleNamespace(platform_base_url="https://example.com"),
        ),
        patch.object(
            integrations,
            "find_webhook_by_credentials_and_props",
            side_effect=find_same_tenant,
        ),
        patch.object(
            integrations,
            "find_webhook_by_credentials_and_props_any_tenant",
            side_effect=find_any_tenant,
        ),
        patch.object(winner_manager, "_register_webhook", register_mock),
        patch.object(loser_manager, "_register_webhook", register_mock),
        patch.object(integrations, "create_webhook", create_mock),
        patch.object(integrations, "update_webhook", update_mock),
    ):
        winner = asyncio.create_task(
            winner_manager.get_suitable_auto_webhook(
                user_id="user-1",
                credentials=credentials,
                webhook_type=TelegramWebhookType.BOT,
                resource="",
                events=["message.text"],
                organization_id="org-1",
                team_id="team-1",
            )
        )
        await asyncio.wait_for(provider_started.wait(), timeout=1)
        loser = asyncio.create_task(
            loser_manager.get_suitable_auto_webhook(
                user_id="user-1",
                credentials=credentials,
                webhook_type=TelegramWebhookType.BOT,
                resource="",
                events=["message.text"],
                organization_id="org-2",
                team_id="team-2",
            )
        )
        await asyncio.sleep(0)
        release_provider.set()
        results = await asyncio.wait_for(
            asyncio.gather(winner, loser, return_exceptions=True), timeout=1
        )

    assert isinstance(results[0], integrations.Webhook)
    assert isinstance(results[1], WebhookRegistrationError)
    assert "another tenancy" in str(results[1])
    assert len(created_webhooks) == 1
    assert created_webhooks[0].organization_id == "org-1"
    assert created_webhooks[0].team_id == "team-1"
    register_mock.assert_awaited_once()
    create_mock.assert_awaited_once()
    update_mock.assert_not_awaited()
    assert redis_mock.await_count == 2
    assert len(mutex_instances) == 2
    expected_key = (
        "webhook-setup",
        ProviderName.TELEGRAM.value,
        "user-1",
        "cred-1",
        "",
    )
    assert acquired_keys == [expected_key, expected_key]
    assert released_keys == [expected_key, expected_key]


@pytest.mark.asyncio
async def test_telegram_fails_closed_when_redis_is_unavailable(monkeypatch) -> None:
    manager = TelegramWebhooksManager()
    locked_setup = AsyncMock()
    monkeypatch.setattr(
        telegram,
        "Config",
        lambda: SimpleNamespace(platform_base_url="https://example.com"),
    )
    monkeypatch.setattr(
        telegram,
        "get_redis_async",
        AsyncMock(side_effect=RuntimeError("redis unavailable")),
    )

    with patch.object(manager, "_get_suitable_auto_webhook_locked", locked_setup):
        with pytest.raises(WebhookRegistrationError, match="safely lock"):
            await manager.get_suitable_auto_webhook(
                user_id="user-1",
                credentials=_credentials(ProviderName.TELEGRAM),
                webhook_type=TelegramWebhookType.BOT,
                resource="",
                events=["message.text"],
            )

    locked_setup.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_fails_closed_when_lock_acquisition_fails(
    monkeypatch,
) -> None:
    manager = TelegramWebhooksManager()
    mutex = SimpleNamespace(
        acquire=AsyncMock(side_effect=RuntimeError("lock unavailable")),
        release=AsyncMock(),
    )
    locked_setup = AsyncMock()
    monkeypatch.setattr(
        telegram,
        "Config",
        lambda: SimpleNamespace(platform_base_url="https://example.com"),
    )
    monkeypatch.setattr(telegram, "AsyncRedisKeyedMutex", lambda _redis: mutex)

    with patch.object(manager, "_get_suitable_auto_webhook_locked", locked_setup):
        with pytest.raises(WebhookRegistrationError, match="safely lock"):
            await manager.get_suitable_auto_webhook(
                user_id="user-1",
                credentials=_credentials(ProviderName.TELEGRAM),
                webhook_type=TelegramWebhookType.BOT,
                resource="",
                events=["message.text"],
            )

    mutex.release.assert_not_awaited()
    locked_setup.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_release_failure_does_not_report_setup_failure(
    monkeypatch,
) -> None:
    manager = TelegramWebhooksManager()
    webhook = _webhook(
        provider=ProviderName.TELEGRAM,
        webhook_type=TelegramWebhookType.BOT,
        resource="",
        organization_id=None,
        team_id=None,
    )
    mutex = SimpleNamespace(
        acquire=AsyncMock(),
        release=AsyncMock(side_effect=RuntimeError("redis disconnected")),
    )
    monkeypatch.setattr(
        telegram,
        "Config",
        lambda: SimpleNamespace(platform_base_url="https://example.com"),
    )
    monkeypatch.setattr(telegram, "AsyncRedisKeyedMutex", lambda _redis: mutex)

    with patch.object(
        manager,
        "_get_suitable_auto_webhook_locked",
        AsyncMock(return_value=webhook),
    ):
        result = await manager.get_suitable_auto_webhook(
            user_id="user-1",
            credentials=_credentials(ProviderName.TELEGRAM),
            webhook_type=TelegramWebhookType.BOT,
            resource="",
            events=["message.text"],
        )

    assert result == webhook
    mutex.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_telegram_creates_mutex_per_event_loop(monkeypatch) -> None:
    mutex_loop_ids: list[int] = []

    class RecordingMutex:
        def __init__(self, _redis: object) -> None:
            mutex_loop_ids.append(id(asyncio.get_running_loop()))

        async def acquire(self, _key: object) -> None:
            pass

        async def release(self, _key: object) -> None:
            pass

    async def locked_setup(*_args, **_kwargs) -> integrations.Webhook:
        return _webhook(
            provider=ProviderName.TELEGRAM,
            webhook_type=TelegramWebhookType.BOT,
            resource="",
            organization_id=None,
            team_id=None,
        )

    async def invoke_setup() -> integrations.Webhook:
        return await TelegramWebhooksManager().get_suitable_auto_webhook(
            user_id="user-1",
            credentials=_credentials(ProviderName.TELEGRAM),
            webhook_type=TelegramWebhookType.BOT,
            resource="",
            events=["message.text"],
        )

    monkeypatch.setattr(
        telegram,
        "Config",
        lambda: SimpleNamespace(platform_base_url="https://example.com"),
    )
    monkeypatch.setattr(telegram, "AsyncRedisKeyedMutex", RecordingMutex)

    with patch.object(
        TelegramWebhooksManager,
        "_get_suitable_auto_webhook_locked",
        side_effect=locked_setup,
    ):
        await invoke_setup()
        await asyncio.to_thread(lambda: asyncio.run(invoke_setup()))

    assert len(mutex_loop_ids) == 2
    assert len(set(mutex_loop_ids)) == 2


@pytest.mark.asyncio
async def test_auto_webhook_lookup_filters_exact_tenant() -> None:
    client = SimpleNamespace(find_first=AsyncMock(return_value=None))

    with patch.object(integrations.IntegrationWebhook, "prisma", return_value=client):
        result = await integrations.find_webhook_by_credentials_and_props(
            user_id="user-1",
            credentials_id="cred-1",
            webhook_type="repo",
            resource="owner/repo",
            organization_id="org-1",
            team_id="team-1",
            events=["push"],
        )

    assert result is None
    client.find_first.assert_awaited_once_with(
        where={
            "userId": "user-1",
            "credentialsId": "cred-1",
            "webhookType": "repo",
            "resource": "owner/repo",
            "organizationId": "org-1",
            "teamId": "team-1",
            "events": {"has_every": ["push"]},
        }
    )


@pytest.mark.asyncio
async def test_cross_tenant_conflict_lookup_does_not_filter_tenant() -> None:
    client = SimpleNamespace(find_first=AsyncMock(return_value=None))

    with patch.object(integrations.IntegrationWebhook, "prisma", return_value=client):
        result = await integrations.find_webhook_by_credentials_and_props_any_tenant(
            user_id="user-1",
            credentials_id="cred-1",
            webhook_type="bot",
            resource="",
        )

    assert result is None
    client.find_first.assert_awaited_once_with(
        where={
            "userId": "user-1",
            "credentialsId": "cred-1",
            "webhookType": "bot",
            "resource": "",
        }
    )


@pytest.mark.asyncio
async def test_manual_webhook_lookup_filters_exact_tenant() -> None:
    client = SimpleNamespace(find_first=AsyncMock(return_value=None))

    with patch.object(integrations.IntegrationWebhook, "prisma", return_value=client):
        result = await integrations.find_webhook_by_graph_and_props(
            user_id="user-1",
            provider=ProviderName.GITHUB.value,
            webhook_type="repo",
            organization_id="org-1",
            team_id="team-1",
            preset_id="preset-1",
        )

    assert result is None
    client.find_first.assert_awaited_once_with(
        where={
            "userId": "user-1",
            "provider": ProviderName.GITHUB.value,
            "webhookType": "repo",
            "organizationId": "org-1",
            "teamId": "team-1",
            "AgentPresets": {"some": {"id": "preset-1"}},
        }
    )
