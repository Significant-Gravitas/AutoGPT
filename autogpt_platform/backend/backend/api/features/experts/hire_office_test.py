"""DB-level tests for the hire-office flow.

Real-postgres tests via the session ``server`` fixture, mirroring
``experts_db_test.py``: seed templates + an OfficeTemplate row, then drive
:mod:`hire_office` directly. The rollback test is the point — every write
of an office hire lands in ONE transaction, so a failure on the Nth expert
must leave no Expert copies and no DelegatedTasks behind.
"""

import uuid
from unittest.mock import AsyncMock, patch

import prisma.enums
import prisma.models
import pytest

from backend.api.features.experts import hire_office
from backend.data.user import get_or_create_user
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer

# DB tests share the session loop so the Prisma pool stays bound to one loop
# (mirrors experts_db_test.py / task_review_test.py).
pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _create_seed_user():
    suffix = uuid.uuid4().hex[:8]
    return await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"office-seed-{suffix}@example.com",
            "name": "Office Owner",
        }
    )


async def _seed_template(role: str) -> prisma.models.Expert:
    return await prisma.models.Expert.prisma().create(
        data={
            "name": f"Office {role} {uuid.uuid4().hex[:8]}",
            "role": role,
            "identity": f"You are a {role} expert.",
            "isTemplate": True,
        }
    )


def _entry(template_id: str, *, cron: str | None = None) -> dict:
    return {
        "template_id": template_id,
        "schedule_cron": cron,
        "intro_task_title": f"Intro for {template_id[:8]}",
        "intro_task_spec": "Introduce yourself and propose a first task.",
    }


@pytest.fixture
def kickoff():
    """The intro-task kickoff reaches the copilot queue, which these DB-level
    tests do not run — stub it and assert on the call instead. The
    expert-task-management flag is pinned on so the kickoff gate opens."""
    with (
        patch.object(
            hire_office,
            "start_task_in_new_session",
            AsyncMock(return_value="session-1"),
        ) as mock,
        patch.object(hire_office, "is_feature_enabled", AsyncMock(return_value=True)),
    ):
        yield mock


async def _seed_office(entries: list[dict]) -> prisma.models.OfficeTemplate:
    return await prisma.models.OfficeTemplate.prisma().create(
        data={
            "name": f"Office pack {uuid.uuid4().hex[:8]}",
            "description": "Test pack",
            "config": SafeJson({"experts": entries}),
        }
    )


async def test_hire_office_hires_experts_and_opens_intro_tasks(
    server: SpinTestServer, kickoff
):
    user = await _create_seed_user()
    templates = [await _seed_template("Marketing"), await _seed_template("Sales")]
    office = await _seed_office([_entry(t.id) for t in templates])

    result = await hire_office.hire_office(user.id, office.id)

    assert result.office_template_id == office.id
    assert result.office_name == office.name
    assert len(result.hired) == 2
    for hired, template in zip(result.hired, templates):
        assert hired.expert.source_template_id == template.id
        assert hired.schedule_created is False

        task = await prisma.models.DelegatedTask.prisma().find_unique(
            where={"id": hired.intro_task_id}
        )
        assert task is not None
        assert task.userId == user.id
        assert task.ownerId == hired.expert.id
        assert task.status == prisma.enums.DelegatedTaskStatus.QUEUED
        assert task.createdByType == prisma.enums.TaskCreatedByType.HIRE
        assert task.createdById == user.id
        assert task.rootTaskId == task.id
        assert task.originSessionId is None
        assert task.title == hired.intro_task_title


async def test_hire_office_starts_every_intro_task(server: SpinTestServer, kickoff):
    """The whole point of the intro task: the expert starts on it. Without a
    kickoff it would sit QUEUED with nothing driving it."""
    user = await _create_seed_user()
    templates = [await _seed_template("Marketing"), await _seed_template("Sales")]
    office = await _seed_office([_entry(t.id) for t in templates])

    result = await hire_office.hire_office(user.id, office.id)

    started = {
        (call.kwargs["task_id"], call.kwargs["expert_id"], call.kwargs["title"])
        for call in kickoff.call_args_list
    }
    assert started == {
        (hired.intro_task_id, hired.expert.id, hired.intro_task_title)
        for hired in result.hired
    }
    assert all(call.args == (user.id,) for call in kickoff.call_args_list)


async def test_hire_office_task_management_off_skips_the_kickoff(
    server: SpinTestServer,
):
    """Without expert-task-management the hire still lands every expert and
    intro-task row, but no worker session is opened — the tasks stay QUEUED
    behind the flag instead of running invisibly."""
    user = await _create_seed_user()
    template = await _seed_template("Ops")
    office = await _seed_office([_entry(template.id)])

    with (
        patch.object(
            hire_office, "start_task_in_new_session", AsyncMock()
        ) as kickoff_mock,
        patch.object(hire_office, "is_feature_enabled", AsyncMock(return_value=False)),
    ):
        result = await hire_office.hire_office(user.id, office.id)

    assert len(result.hired) == 1
    kickoff_mock.assert_not_awaited()
    task = await prisma.models.DelegatedTask.prisma().find_unique(
        where={"id": result.hired[0].intro_task_id}
    )
    assert task is not None
    assert task.status == prisma.enums.DelegatedTaskStatus.QUEUED


async def test_hire_office_survives_a_failed_kickoff(server: SpinTestServer, kickoff):
    """A kickoff that cannot reach the queue must not fail the hire — the
    team is already committed, and the overseer retries the stalled task."""
    user = await _create_seed_user()
    template = await _seed_template("Ops")
    office = await _seed_office([_entry(template.id)])
    kickoff.side_effect = RuntimeError("queue is down")

    result = await hire_office.hire_office(user.id, office.id)

    assert len(result.hired) == 1
    task = await prisma.models.DelegatedTask.prisma().find_unique(
        where={"id": result.hired[0].intro_task_id}
    )
    assert task is not None
    assert task.status == prisma.enums.DelegatedTaskStatus.QUEUED


async def test_hire_office_rolls_back_all_writes_on_failure(
    server: SpinTestServer,
):
    user = await _create_seed_user()
    templates = [await _seed_template("Marketing"), await _seed_template("Sales")]
    office = await _seed_office([_entry(t.id) for t in templates])

    real_create_intro_task = hire_office._create_intro_task
    calls = 0

    async def fail_on_second(tx, user_id, expert_id, entry):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second intro task exploded")
        return await real_create_intro_task(tx, user_id, expert_id, entry)

    with patch.object(hire_office, "_create_intro_task", new=fail_on_second):
        with pytest.raises(RuntimeError, match="second intro task exploded"):
            await hire_office.hire_office(user.id, office.id)

    assert calls == 2
    assert (
        await prisma.models.Expert.prisma().count(where={"ownerUserId": user.id}) == 0
    )
    assert (
        await prisma.models.DelegatedTask.prisma().count(where={"userId": user.id}) == 0
    )


async def test_hire_office_unknown_office_raises(server: SpinTestServer):
    user = await _create_seed_user()
    with pytest.raises(hire_office.OfficeTemplateNotFoundError):
        await hire_office.hire_office(user.id, str(uuid.uuid4()))


async def test_list_office_templates_joins_expert_rows(server: SpinTestServer):
    template = await _seed_template("Ops")
    office = await _seed_office([_entry(template.id, cron="0 9 * * 1")])

    summaries = await hire_office.list_office_templates()
    summary = next(s for s in summaries if s.id == office.id)

    assert summary.name == office.name
    assert summary.description == "Test pack"
    assert len(summary.experts) == 1
    line = summary.experts[0]
    assert line.template_id == template.id
    assert line.name == template.name
    assert line.role == "Ops"
    assert line.schedule_cron == "0 9 * * 1"
    assert line.intro_task_title == f"Intro for {template.id[:8]}"
