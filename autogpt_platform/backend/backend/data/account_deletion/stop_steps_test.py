"""The steps that stop the account (sign-in, schedules, runs, billing,
integrations) and remove its memory: dry runs only read, applies reuse the
platform's own functions, and an account with nothing left is a no-op."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.account_deletion import activity, integrations, memory, subscriptions
from backend.data.account_deletion.models import StepStatus, Subject

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
SUBJECT = Subject(user_id=USER_ID, has_user_row=True, has_auth_identity=True)


def _table(**actions) -> MagicMock:
    table = MagicMock()
    for name, value in actions.items():
        setattr(table.prisma.return_value, name, AsyncMock(return_value=value))
    return table


@pytest.mark.asyncio
async def test_sign_in_dry_run_counts_sessions_only(monkeypatch):
    sessions = _table(count=2, delete_many=2)
    identity = _table(update=None)
    monkeypatch.setattr(activity, "AuthSession", sessions)
    monkeypatch.setattr(activity, "AuthUser", identity)

    report = await activity.disable_sign_in(SUBJECT, dry_run=True)

    assert report.counts == {"sessions": 2}
    sessions.prisma.return_value.delete_many.assert_not_awaited()
    identity.prisma.return_value.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_sign_in_is_disabled_and_marketing_refused(monkeypatch):
    sessions = _table(delete_many=2)
    identity = _table(update=None)
    user = _table(update_many=1)
    monkeypatch.setattr(activity, "AuthSession", sessions)
    monkeypatch.setattr(activity, "AuthUser", identity)
    monkeypatch.setattr(activity, "User", user)

    report = await activity.disable_sign_in(SUBJECT, dry_run=False)

    ban = identity.prisma.return_value.update.await_args.kwargs["data"]
    assert ban["banned"] is True
    opt_out = user.prisma.return_value.update_many.await_args.kwargs
    assert opt_out["where"] == {"id": USER_ID, "marketingOptOutAt": None}
    assert opt_out["data"]["marketingOptOutSource"] == activity.OPT_OUT_SOURCE
    assert report.counts == {"sessions": 2}


@pytest.mark.asyncio
async def test_without_an_identity_nothing_is_banned(monkeypatch):
    identity = _table(update=None)
    monkeypatch.setattr(activity, "AuthSession", _table(delete_many=0))
    monkeypatch.setattr(activity, "AuthUser", identity)
    monkeypatch.setattr(activity, "User", _table(update_many=0))

    await activity.disable_sign_in(Subject(user_id=USER_ID), dry_run=False)

    identity.prisma.return_value.update.assert_not_awaited()


@pytest.fixture
def scheduler(monkeypatch):
    client = MagicMock()
    client.get_execution_schedules = AsyncMock(
        return_value=[SimpleNamespace(id="s1"), SimpleNamespace(id="s2")]
    )
    for name in (
        "delete_schedule",
        "delete_community_rebuild_schedule",
        "delete_nightly_batch_schedule",
        "remove_morning_briefing_schedule",
    ):
        setattr(client, name, AsyncMock())
    monkeypatch.setattr(activity, "get_scheduler_client", lambda: client)
    return client


@pytest.mark.asyncio
async def test_a_schedules_dry_run_deletes_nothing(scheduler):
    report = await activity.stop_schedules(SUBJECT, dry_run=True)

    assert report.counts == {"schedules": 2}
    scheduler.delete_schedule.assert_not_awaited()
    scheduler.delete_nightly_batch_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_every_schedule_and_system_cron_is_removed(scheduler):
    await activity.stop_schedules(SUBJECT, dry_run=False)

    assert [c.args[0] for c in scheduler.delete_schedule.await_args_list] == [
        "s1",
        "s2",
    ]
    assert scheduler.get_execution_schedules.await_args.kwargs["include_paused"]
    scheduler.delete_community_rebuild_schedule.assert_awaited_once_with(USER_ID)
    scheduler.delete_nightly_batch_schedule.assert_awaited_once_with(USER_ID)
    scheduler.remove_morning_briefing_schedule.assert_awaited_once_with(USER_ID)


@pytest.mark.asyncio
async def test_running_executions_are_stopped(monkeypatch):
    runs = [SimpleNamespace(id="e1"), SimpleNamespace(id="e2")]
    monkeypatch.setattr(
        activity.execution_db, "get_graph_executions", AsyncMock(return_value=runs)
    )
    stop = AsyncMock()
    monkeypatch.setattr(activity, "stop_graph_execution", stop)

    dry = await activity.stop_executions(SUBJECT, dry_run=True)
    stop.assert_not_awaited()
    await activity.stop_executions(SUBJECT, dry_run=False)

    assert dry.counts == {"running": 2}
    assert {c.kwargs["graph_exec_id"] for c in stop.await_args_list} == {"e1", "e2"}


@pytest.fixture
def stripe_configured(monkeypatch):
    monkeypatch.setattr(
        subscriptions,
        "Settings",
        lambda: SimpleNamespace(secrets=SimpleNamespace(stripe_api_key="sk_test")),
    )


@pytest.mark.asyncio
async def test_subscriptions_of_every_customer_are_cancelled_now(
    monkeypatch, stripe_configured
):
    cancel = AsyncMock(return_value=1)
    monkeypatch.setattr(subscriptions, "_cancel_customer_subscriptions", cancel)
    subject = SUBJECT.model_copy(update={"stripe_customer_ids": ["cus_a", "cus_b"]})

    report = await subscriptions.cancel_subscriptions(subject, dry_run=False)

    assert [c.args[0] for c in cancel.await_args_list] == ["cus_a", "cus_b"]
    for call in cancel.await_args_list:
        assert call.kwargs.get("at_period_end", False) is False
        assert call.kwargs["cancellation_comment"] == "autogpt:account_deleted"
    assert report.counts == {"subscriptions": 2}


@pytest.mark.asyncio
async def test_a_subscriptions_dry_run_only_counts(monkeypatch, stripe_configured):
    cancel = AsyncMock()
    monkeypatch.setattr(subscriptions, "_cancel_customer_subscriptions", cancel)
    monkeypatch.setattr(subscriptions, "_count_billable", AsyncMock(return_value=1))
    subject = SUBJECT.model_copy(update={"stripe_customer_ids": ["cus_a"]})

    report = await subscriptions.cancel_subscriptions(subject, dry_run=True)

    cancel.assert_not_awaited()
    assert report.counts == {"subscriptions": 1}


@pytest.mark.asyncio
async def test_no_stripe_customer_is_a_no_op(monkeypatch):
    cancel = AsyncMock()
    monkeypatch.setattr(subscriptions, "_cancel_customer_subscriptions", cancel)

    report = await subscriptions.cancel_subscriptions(SUBJECT, dry_run=False)

    cancel.assert_not_awaited()
    assert report.counts == {"subscriptions": 0}


@pytest.mark.asyncio
async def test_webhooks_are_deregistered_with_their_own_credentials(monkeypatch):
    mine = SimpleNamespace(id="cred-1", is_managed=False)
    managed = SimpleNamespace(id="cred-2", is_managed=True)
    system = SimpleNamespace(id="system-openai", is_managed=False)
    store = MagicMock()
    store.get_all_creds = AsyncMock(return_value=[mine, managed, system])
    monkeypatch.setattr(integrations, "IntegrationCredentialsStore", lambda: store)
    monkeypatch.setattr(
        integrations,
        "IntegrationWebhook",
        _table(find_many=[SimpleNamespace(credentialsId="cred-1")]),
    )
    remove = AsyncMock()
    cleanup = AsyncMock()
    monkeypatch.setattr(integrations, "remove_all_webhooks_for_credentials", remove)
    monkeypatch.setattr(integrations, "cleanup_user_managed_credentials", cleanup)

    dry = await integrations.release_integrations(SUBJECT, dry_run=True)
    remove.assert_not_awaited()
    cleanup.assert_not_awaited()
    await integrations.release_integrations(SUBJECT, dry_run=False)

    assert dry.counts == {"webhooks": 1, "managed_credentials": 1}
    remove.assert_awaited_once_with(USER_ID, mine, force=True)
    cleanup.assert_awaited_once_with(USER_ID, store)


class FakeFalkor:
    def __init__(self, graphs: list[str]):
        self.graphs = graphs
        self.deleted: list[str] = []
        self.closed = False

    async def list_graphs(self) -> list[str]:
        return list(self.graphs)

    def select_graph(self, name: str):
        graph = MagicMock()

        async def delete():
            self.deleted.append(name)

        graph.delete = delete
        return graph

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_only_the_accounts_memory_graphs_are_deleted(monkeypatch):
    subject = SUBJECT.model_copy(update={"expert_ids": ["expert-1"]})
    expert_graph = memory.derive_memory_group_id(USER_ID, "expert-1")
    falkor = FakeFalkor([f"user_{USER_ID}", expert_graph, "user_someone-else"])
    monkeypatch.setattr(memory, "FalkorDB", lambda **kwargs: falkor)
    monkeypatch.setattr(memory, "evict_client", AsyncMock())

    dry = await memory.delete_memory(subject, dry_run=True)
    assert falkor.deleted == []
    report = await memory.delete_memory(subject, dry_run=False)

    assert dry.counts == {"memory_graphs": 2}
    assert sorted(falkor.deleted) == sorted([f"user_{USER_ID}", expert_graph])
    assert falkor.closed
    assert report.status == StepStatus.DONE
