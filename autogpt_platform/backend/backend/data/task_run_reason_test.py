"""How a run's end is read, without the database: the status and reason for
each ending, the worker's last words as the reason, and which statuses count
as another try."""

from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
from prisma.enums import TaskStatus
from pytest_mock import MockerFixture

from backend.data import task_run as run_db
from backend.data.task_run import NO_RESULT_REASON, _settled_status, reply_reason


def _row(
    result: str | None = None,
    artifacts: list[prisma.models.TaskArtifact] | None = None,
    evidence: list[prisma.models.TaskEvidence] | None = None,
) -> prisma.models.Task:
    return prisma.models.Task.model_construct(
        result=result, Subtasks=[], Artifacts=artifacts or [], Evidence=evidence or []
    )


def _file(is_missing: bool = False) -> prisma.models.TaskArtifact:
    return prisma.models.TaskArtifact.model_construct(isMissing=is_missing)


_BLOCKED = (
    "Blocked: Shopify isn't connected, so I couldn't pull Powai's Nov 2 to 8 "
    "orders and no sheet was made. Nothing in email or Drive shows Powai uses "
    "Shopify. Connect Shopify and I'll build the sheet."
)


@pytest.mark.parametrize(
    "summary",
    [
        _BLOCKED,
        "**Blocked** on the Shopify connection.",
        "Could not find any Powai orders for last week.",
        "I couldn't reach the Shopify store.",
        "Unable to export: Shopify isn't connected.",
        "I was unable to open the shared sheet.",
    ],
)
def test_a_result_that_only_says_it_was_blocked_fails(summary):
    status, reason = _settled_status(_row(summary), "completed")

    assert status == TaskStatus.FAILED
    assert reason and reason.split()[0].strip("*") == summary.split()[0].strip("*")


def test_the_blocked_summary_becomes_a_short_reason():
    _, reason = _settled_status(_row(_BLOCKED), "completed")

    assert reason == (
        "Blocked: Shopify isn't connected, so I couldn't pull Powai's Nov 2 to 8 "
        "orders and no sheet was made. Nothing in email or Drive shows Powai "
        "uses Shopify."
    )


@pytest.mark.parametrize(
    "row",
    [
        # A real result that mentions one thing it could not do.
        _row("Saved the Powai orders sheet; I couldn't get delivery times for 3."),
        _row("Sheet of 42 Powai orders saved. Unable to match 2 refunds."),
        # Opens like a blocker, but the work left a file or proof.
        _row("Could not reach Shopify, so I used the Swiggy export.", [_file()]),
        _row(
            "Blocked on one creator; emailed the other two.",
            evidence=[prisma.models.TaskEvidence.model_construct()],
        ),
    ],
)
def test_a_real_result_still_settles_done(row):
    assert _settled_status(row, "completed") == (TaskStatus.DONE, None)


def test_a_removed_file_is_not_work_done():
    row = _row("Could not finish the sheet.", [_file(is_missing=True)])

    assert _settled_status(row, "completed")[0] == TaskStatus.FAILED


def test_a_clean_end_without_a_result_fails_with_the_workers_words():
    reply = (
        "I couldn't pull the Powai orders: Shopify isn't connected for this "
        "account. Connect it and try again. Meanwhile I drafted nothing."
    )

    status, reason = _settled_status(_row(), "completed", reply)

    assert status == TaskStatus.FAILED
    assert reason == (
        "I couldn't pull the Powai orders: Shopify isn't connected for this "
        "account. Connect it and try again."
    )


@pytest.mark.parametrize("reply", [None, "", "   ", "**  **"])
def test_no_last_words_keeps_the_generic_reason(reply):
    assert _settled_status(_row(), "completed", reply) == (
        TaskStatus.FAILED,
        NO_RESULT_REASON,
    )


@pytest.mark.parametrize(
    "ending,reason",
    [
        ("failed", "The work stopped before it finished."),
        ("timed_out", "The work ran out of time."),
    ],
)
def test_errors_and_timeouts_keep_their_wording(ending, reason):
    assert _settled_status(_row(), ending, "Something I said") == (
        TaskStatus.FAILED,
        reason,
    )


def test_a_result_still_wins_over_the_last_words():
    assert _settled_status(_row("Saved the sheet."), "completed", "Bye") == (
        TaskStatus.DONE,
        None,
    )


def test_the_reason_fits_the_activity_line_and_drops_secrets():
    reply = "Tried the key sk-abcdefghijklmnop " + "and then waited " * 40

    reason = reply_reason(reply)

    assert reason is not None
    assert "sk-abcdefghijklmnop" not in reason
    assert len(f"Failed: {reason}") <= 300
    assert reason.endswith("…")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,retry,attempt",
    [
        (TaskStatus.FAILED, True, 3),
        (TaskStatus.CANCELLED, True, 3),
        (TaskStatus.DONE, False, None),
        (TaskStatus.TODO, False, None),
    ],
)
async def test_failed_and_cancelled_tasks_count_another_attempt(
    mocker: MockerFixture, status: TaskStatus, retry: bool, attempt: int | None
):
    before = MagicMock(status=status, attempt=2, title="Plan", missed_start=False)
    mocker.patch(f"{run_db.__name__}.get_task", AsyncMock(return_value=before))
    tasks = MagicMock()
    tasks.update_many = AsyncMock(return_value=0)
    mocker.patch.object(prisma.models.Task, "prisma", return_value=tasks)

    with pytest.raises(run_db.ConflictError):
        await run_db.claim_task_run("task-1", "user-1")

    data = tasks.update_many.await_args.kwargs["data"]
    assert data.get("attempt") == attempt
    assert (status in run_db.RETRY_STATUSES) is retry
