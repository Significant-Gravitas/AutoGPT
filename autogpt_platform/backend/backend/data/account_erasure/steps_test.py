"""The steps that stop the account's work again, cancel its sole orgs'
plans and remove its memory: dry runs only read, applies reuse the
platform's own functions, and an account with nothing left is a no-op."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import AccountDeletionSource

from backend.data.account_erasure import memory, subscriptions, work
from backend.data.account_erasure.models import StepStatus, Subject

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
SUBJECT = Subject(
    user_id=USER_ID,
    has_user_row=True,
    has_auth_identity=True,
    request_source=AccountDeletionSource.SETTINGS,
)


@pytest.fixture
def work_calls(monkeypatch):
    scheduler = MagicMock()
    scheduler.get_execution_schedules = AsyncMock(
        return_value=[SimpleNamespace(id="s1"), SimpleNamespace(id="s2")]
    )
    monkeypatch.setattr(work, "get_scheduler_client", lambda: scheduler)
    runs = AsyncMock(return_value=[SimpleNamespace(id="e1")])
    monkeypatch.setattr(work.execution_db, "get_graph_executions", runs)
    stop = AsyncMock()
    monkeypatch.setattr(work, "stop_user_work", stop)
    return SimpleNamespace(scheduler=scheduler, runs=runs, stop=stop)


@pytest.mark.asyncio
async def test_a_work_dry_run_only_counts(work_calls):
    report = await work.stop_work(SUBJECT, dry_run=True)

    assert report.counts == {"schedules": 2, "unfinished_runs": 1}
    assert work_calls.scheduler.get_execution_schedules.await_args.kwargs[
        "include_paused"
    ]
    work_calls.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_apply_reruns_the_close_steps_work_stop(work_calls):
    """`stop_user_work` (SECRT-2877) does the stopping: every schedule and
    system cron, the triggers, chat links, AutoPilot turns and runs."""
    report = await work.stop_work(SUBJECT, dry_run=False)

    work_calls.stop.assert_awaited_once_with(USER_ID)
    assert report.status == StepStatus.DONE


@pytest.fixture
def stripe_configured(monkeypatch):
    monkeypatch.setattr(
        subscriptions,
        "Settings",
        lambda: SimpleNamespace(secrets=SimpleNamespace(stripe_api_key="sk_test")),
    )


@pytest.mark.asyncio
async def test_org_subscriptions_are_cancelled_now_like_the_accounts_own(
    monkeypatch, stripe_configured
):
    expire = AsyncMock()
    cancel = AsyncMock(return_value=1)
    monkeypatch.setattr(subscriptions, "expire_all_open_checkouts", expire)
    monkeypatch.setattr(subscriptions, "_cancel_customer_subscriptions", cancel)
    subject = SUBJECT.model_copy(update={"org_stripe_customer_ids": ["cus_a", "cus_b"]})

    report = await subscriptions.cancel_org_subscriptions(subject, dry_run=False)

    assert [c.args[0] for c in expire.await_args_list] == ["cus_a", "cus_b"]
    assert [c.args[0] for c in cancel.await_args_list] == ["cus_a", "cus_b"]
    for call in cancel.await_args_list:
        # Passed explicitly, so a change to the helper's default can't turn
        # this into a cancellation at period end.
        assert call.kwargs["at_period_end"] is False
        assert call.kwargs["cancellation_comment"] == "autogpt:account_deleted"
        assert call.kwargs["statuses"] == subscriptions._ACCOUNT_DELETION_STATUSES
    assert report.counts == {"subscriptions": 2}


@pytest.mark.asyncio
async def test_an_org_subscriptions_dry_run_only_counts(monkeypatch, stripe_configured):
    cancel = AsyncMock()
    monkeypatch.setattr(subscriptions, "_cancel_customer_subscriptions", cancel)
    monkeypatch.setattr(subscriptions, "_count_cancellable", AsyncMock(return_value=1))
    subject = SUBJECT.model_copy(update={"org_stripe_customer_ids": ["cus_a"]})

    report = await subscriptions.cancel_org_subscriptions(subject, dry_run=True)

    cancel.assert_not_awaited()
    assert report.counts == {"subscriptions": 1}


@pytest.mark.asyncio
async def test_no_org_customer_is_a_no_op(monkeypatch):
    cancel = AsyncMock()
    monkeypatch.setattr(subscriptions, "_cancel_customer_subscriptions", cancel)

    report = await subscriptions.cancel_org_subscriptions(SUBJECT, dry_run=False)

    cancel.assert_not_awaited()
    assert report.counts == {"subscriptions": 0}


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


@pytest.fixture
def falkordb_at(monkeypatch):
    def configure(host: str):
        monkeypatch.setattr(
            memory,
            "graphiti_config",
            SimpleNamespace(
                falkordb_host=host, falkordb_port=6379, falkordb_password=""
            ),
        )

    return configure


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["", "  "])
async def test_memory_is_skipped_on_a_stack_without_falkordb(
    monkeypatch, falkordb_at, host
):
    falkordb_at(host)
    connect = MagicMock()
    monkeypatch.setattr(memory, "FalkorDB", connect)

    report = await memory.delete_memory(SUBJECT, dry_run=False)

    assert report.status == StepStatus.SKIPPED
    assert "GRAPHITI_FALKORDB_HOST" in report.note
    connect.assert_not_called()


@pytest.mark.asyncio
async def test_an_unreachable_falkordb_fails_the_step(monkeypatch, falkordb_at):
    """Configured but down is not "not configured": the step raises, so the
    routine reports it failed and stops before the account's rows go."""
    falkordb_at("falkordb.internal")
    falkor = FakeFalkor([])
    falkor.list_graphs = AsyncMock(side_effect=ConnectionError("refused"))
    monkeypatch.setattr(memory, "FalkorDB", lambda **kwargs: falkor)

    with pytest.raises(ConnectionError):
        await memory.delete_memory(SUBJECT, dry_run=False)

    assert falkor.closed
