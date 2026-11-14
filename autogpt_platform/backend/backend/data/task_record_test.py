"""Integration tests for the task record, against the test database: who did
what, what a task asked, what it made, the proof, and how subtasks roll up."""

import asyncio
import logging
from uuid import uuid4

import pytest
from prisma.enums import (
    TaskActivityKind,
    TaskArtifactKind,
    TaskAskKind,
    TaskAskStatus,
    TaskCreator,
    TaskEvidenceKind,
    TaskStatus,
)
from prisma.models import ChatSession, Expert, Task, User

from backend.data import task as task_db
from backend.data import task_record as record_db
from backend.data import task_result as result_db
from backend.data.task import MAX_SUBTASKS, TaskPatch
from backend.data.task_record import (
    ArtifactInput,
    AskInput,
    EvidenceInput,
    ToolCallRecord,
    redact,
)
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)


async def _user() -> str:
    user_id = f"task-record-{uuid4()}"
    await User.prisma().create(
        data={"id": user_id, "email": f"{user_id}@example.com", "name": "Record Test"}
    )
    return user_id


async def _chat(user_id: str) -> str:
    return (await ChatSession.prisma().create(data={"userId": user_id})).id


async def _cleanup(*user_ids: str) -> None:
    for user_id in user_ids:
        try:
            await Task.prisma().delete_many(where={"userId": user_id})
            await ChatSession.prisma().delete_many(where={"userId": user_id})
            await Expert.prisma().delete_many(where={"ownerUserId": user_id})
            await User.prisma().delete_many(where={"id": user_id})
        except Exception as exc:
            logger.warning("cleanup for %s failed: %s", user_id, exc)


async def _working_task(
    user_id: str,
    title: str = "Plan the launch",
    parent_task_id: str | None = None,
    done_when: str | None = None,
) -> tuple[task_db.Task, str]:
    """A task in progress and the chat working on it."""
    chat_id = await _chat(user_id)
    task = await task_db.create_task(
        user_id=user_id,
        title=title,
        created_by=TaskCreator.OTTO,
        session_id=chat_id,
        goal="Launch on Nov 20",
        done_when=done_when,
        parent_task_id=parent_task_id,
    )
    task = await task_db.update_task(
        task.id,
        user_id,
        TaskPatch(status=TaskStatus.IN_PROGRESS, work_session_id=chat_id),
    )
    return task, chat_id


def _call(tool: str, call_id: str = "call-1", **extra) -> ToolCallRecord:
    label = extra.pop("label", f"Used {tool}")
    return ToolCallRecord(
        tool=tool,
        tool_call_id=call_id,
        label=label,
        label_many=extra.pop("label_many", label + " {count} times"),
        **extra,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_tool_calls_become_activity_and_repeats_fold_into_one_line(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        task, chat_id = await _working_task(user_id)
        search = _call("web_search", label="Searched the web")
        for call_id in ("a", "b", "c"):
            search.tool_call_id = call_id
            assert await record_db.record_tool_call(chat_id, user_id, search)
        await record_db.record_tool_call(
            chat_id, user_id, _call("bash_exec", label="Ran a command")
        )

        record = await record_db.get_task_record(task.id, user_id)
        assert record is not None
        assert [line.text for line in record.activity] == [
            "Ran a command",
            "Searched the web 3 times",
        ]
        # Nothing said by the worker yet: the last thing it did stands in.
        assert record.task.status_text == "Ran a command"
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_the_workers_own_words_win_over_the_tool_fallback(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        task, chat_id = await _working_task(user_id)
        updated = await record_db.set_progress(
            chat_id, user_id, "Comparing three CRMs", "Write the summary"
        )
        assert updated is not None
        await record_db.record_tool_call(chat_id, user_id, _call("web_search"))

        record = await record_db.get_task_record(task.id, user_id)
        assert record is not None
        assert record.task.status_text == "Comparing three CRMs"
        assert record.task.next_step == "Write the summary"
        assert "Comparing three CRMs" in record.so_far
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_chat_working_on_no_task_records_nothing(server: SpinTestServer):
    user_id = await _user()
    try:
        chat_id = await _chat(user_id)
        assert not await record_db.record_tool_call(
            chat_id, user_id, _call("web_search")
        )
        assert await record_db.set_progress(chat_id, user_id, "x") is None
        assert (
            await record_db.add_evidence(chat_id, user_id, EvidenceInput(claim="x"))
            is None
        )
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_written_file_is_one_artifact_and_reads_as_removed_once_deleted(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        task, chat_id = await _working_task(user_id)
        wrote = _call(
            "write_workspace_file",
            label="Wrote plan.md",
            artifact=ArtifactInput(title="plan.md", workspace_file_id="file-1"),
        )
        await record_db.record_tool_call(chat_id, user_id, wrote)
        # Written again: still one artifact, pointing at the newer file.
        wrote.artifact = ArtifactInput(title="plan.md", workspace_file_id="file-2")
        await record_db.record_tool_call(chat_id, user_id, wrote)

        record = await record_db.get_task_record(task.id, user_id)
        assert record is not None
        assert [(a.title, a.workspace_file_id) for a in record.artifacts] == [
            ("plan.md", "file-2")
        ]
        assert record.task.artifact_count == 1

        await record_db.record_tool_call(
            chat_id, user_id, _call("delete_workspace_file", deleted_file_id="file-2")
        )
        record = await record_db.get_task_record(task.id, user_id)
        assert record is not None and record.artifacts[0].is_missing
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_evidence_is_verified_only_when_the_task_really_did_it(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        task, chat_id = await _working_task(user_id, done_when="Email sent")
        # A call that changed something outside is recorded as proof by itself.
        await record_db.record_tool_call(
            chat_id,
            user_id,
            _call("mcp:gmail/send_email", "call-9", evidence_claim="Used send email"),
        )
        await record_db.record_tool_call(
            chat_id,
            user_id,
            _call(
                "write_workspace_file",
                artifact=ArtifactInput(title="report.md", workspace_file_id="f"),
            ),
        )
        by_tool = await record_db.add_evidence(
            chat_id,
            user_id,
            EvidenceInput(claim="Sent the invoice", tool="mcp:gmail/send_email"),
        )
        by_file = await record_db.add_evidence(
            chat_id, user_id, EvidenceInput(claim="Report written", file="REPORT.md")
        )
        invented = await record_db.add_evidence(
            chat_id, user_id, EvidenceInput(claim="Paid the bill", tool="stripe_pay")
        )
        no_file = await record_db.add_evidence(
            chat_id, user_id, EvidenceInput(claim="Deck made", file="deck.pptx")
        )
        link = await record_db.add_evidence(
            chat_id, user_id, EvidenceInput(claim="Page is live", url="https://x.test")
        )
        # A worker names its source loosely: a prefixed tool, two tools at
        # once, a sandbox path. Each still finds what the task really did.
        prefixed = await record_db.add_evidence(
            chat_id,
            user_id,
            EvidenceInput(claim="Invoice sent", tool="mcp__copilot__send_email"),
        )
        several = await record_db.add_evidence(
            chat_id,
            user_id,
            EvidenceInput(claim="Sent", tool="write_file + send_email"),
        )
        by_path = await record_db.add_evidence(
            chat_id,
            user_id,
            EvidenceInput(claim="Report saved", file="/home/user/report.md"),
        )
        assert prefixed and prefixed.verified
        assert several and several.verified
        assert by_path and by_path.verified and by_path.artifact_id

        assert by_tool and by_tool.verified and by_tool.tool_call_id == "call-9"
        assert by_file and by_file.verified and by_file.artifact_id
        assert invented and not invented.verified
        assert no_file and not no_file.verified
        assert link and not link.verified and link.kind == TaskEvidenceKind.LINK

        record = await record_db.get_task_record(task.id, user_id)
        assert record is not None
        automatic = [e for e in record.evidence if e.claim == "Used send email"]
        assert automatic and automatic[0].verified
    finally:
        await _cleanup(user_id)


@pytest.mark.parametrize(
    "quote, kept, removed",
    [
        (
            "Sent with key sk-abcdefghijklmnop ok",
            "Sent with key",
            "sk-abcdefghijklmnop",
        ),
        ("token eyJhbGciOiJIUzI1NiJ9abc here", "token", "eyJhbGciOiJIUzI1NiJ9abc"),
        ("Order 1234 confirmed for Acme", "Order 1234 confirmed for Acme", None),
    ],
)
def test_a_quote_keeps_its_words_and_loses_anything_shaped_like_a_secret(
    quote, kept, removed
):
    excerpt = redact(quote)
    assert excerpt is not None and kept in excerpt
    if removed:
        assert removed not in excerpt and "[removed]" in excerpt


def test_a_long_quote_is_cut_short():
    excerpt = redact("word " * 400)
    assert excerpt is not None and len(excerpt) <= 280


@pytest.mark.asyncio(loop_scope="session")
async def test_done_against_a_done_when_with_no_proof_reads_as_unverified(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        task, chat_id = await _working_task(user_id, done_when="Invoice sent")
        await record_db.settle_work_tasks(chat_id, user_id, TaskStatus.DONE, "Sent it")
        record = await record_db.get_task_record(task.id, user_id)
        assert record is not None and record.is_unverified
        assert record.task.result == "Sent it"

        plain, plain_chat = await _working_task(user_id, title="No bar set")
        await record_db.settle_work_tasks(plain_chat, user_id, TaskStatus.DONE, "ok")
        record = await record_db.get_task_record(plain.id, user_id)
        assert record is not None and not record.is_unverified
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_question_waits_on_the_task_and_the_answer_is_kept(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        task, chat_id = await _working_task(user_id)
        asked = await record_db.open_asks(
            chat_id,
            user_id,
            [
                AskInput(
                    question="Which list gets the email?",
                    kind=TaskAskKind.PICK,
                    options=["Customers", "Everyone"],
                    default_guess="Customers",
                ),
                AskInput(question="Who signs it?"),
            ],
        )
        assert asked == 2
        waiting = await task_db.get_task(task.id, user_id)
        assert waiting is not None and waiting.open_ask_count == 2

        answered = await record_db.answer_session_asks(
            chat_id, user_id, "Customers only, signed by Sam"
        )
        assert answered == 2
        record = await record_db.get_task_record(task.id, user_id)
        assert record is not None
        assert {ask.status for ask in record.asks} == {TaskAskStatus.ANSWERED}
        assert all(ask.answer == "Customers only, signed by Sam" for ask in record.asks)
        assert record.task.open_ask_count == 0
        assert record.so_far[0] == "You answered: Customers only, signed by Sam"
        # A second reply finds nothing left to answer.
        assert await record_db.answer_session_asks(chat_id, user_id, "again") == 0
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_settling_or_cancelling_withdraws_the_questions_still_open(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        task, chat_id = await _working_task(user_id)
        await record_db.open_asks(chat_id, user_id, [AskInput(question="Which one?")])
        await task_db.update_task(
            task.id, user_id, TaskPatch(status=TaskStatus.CANCELLED)
        )
        record = await record_db.get_task_record(task.id, user_id)
        assert record is not None
        assert [ask.status for ask in record.asks] == [TaskAskStatus.WITHDRAWN]
        assert record.task.open_ask_count == 0
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_finished_worker_leaves_the_result_and_a_failed_one_the_reason(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        done, done_chat = await _working_task(user_id, title="Done one")
        failed, failed_chat = await _working_task(user_id, title="Failed one")
        waiting, waiting_chat = await _working_task(user_id, title="Waiting one")
        await task_db.update_task(
            waiting.id, user_id, TaskPatch(status=TaskStatus.NEEDS_INPUT)
        )

        assert (
            await record_db.settle_work_tasks(
                done_chat, user_id, TaskStatus.DONE, "The plan is ready."
            )
            == 1
        )
        await record_db.settle_work_tasks(
            failed_chat, user_id, TaskStatus.FAILED, None, "The work ran out of time."
        )
        # A worker that ended by asking leaves its task waiting.
        assert (
            await record_db.settle_work_tasks(waiting_chat, user_id, TaskStatus.DONE)
            == 0
        )
        # Settling twice moves nothing twice.
        assert (
            await record_db.settle_work_tasks(done_chat, user_id, TaskStatus.DONE) == 0
        )

        done_now = await record_db.get_task_record(done.id, user_id)
        failed_now = await record_db.get_task_record(failed.id, user_id)
        waiting_now = await task_db.get_task(waiting.id, user_id)
        assert done_now and done_now.task.status == TaskStatus.DONE
        assert done_now.task.result == "The plan is ready."
        assert done_now.activity[0].kind == TaskActivityKind.DONE
        assert failed_now and failed_now.task.status == TaskStatus.FAILED
        assert failed_now.task.failure_reason == "The work ran out of time."
        assert failed_now.so_far[0] == "Failed: The work ran out of time."
        assert waiting_now and waiting_now.status == TaskStatus.NEEDS_INPUT
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_failed_task_counts_its_retries_and_a_finished_one_does_not(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        task, chat_id = await _working_task(user_id)
        await record_db.settle_work_tasks(
            chat_id, user_id, TaskStatus.FAILED, None, "The work stopped."
        )
        retried = await task_db.update_task(
            task.id, user_id, TaskPatch(status=TaskStatus.IN_PROGRESS, retry=True)
        )
        assert retried.attempt == 2
        assert retried.failure_reason is None and retried.completed_at is None

        await record_db.settle_work_tasks(chat_id, user_id, TaskStatus.DONE, "ok")
        reopened = await task_db.update_task(
            task.id, user_id, TaskPatch(status=TaskStatus.IN_PROGRESS)
        )
        assert reopened.attempt == 2 and reopened.result == "ok"
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_subtasks_are_one_level_deep_and_capped(server: SpinTestServer):
    user_id = await _user()
    try:
        parent, _ = await _working_task(user_id)
        children = [
            (await _working_task(user_id, f"Part {n}", parent.id))[0]
            for n in range(MAX_SUBTASKS)
        ]
        with pytest.raises(ValueError, match="at most"):
            await _working_task(user_id, "One too many", parent.id)
        with pytest.raises(ValueError, match="cannot have subtasks"):
            await _working_task(user_id, "Grandchild", children[0].id)

        # The board lists top tasks; a subtask shows under its parent.
        top = await task_db.list_tasks(user_id, include_subtasks=False)
        assert [task.id for task in top] == [parent.id]
        under = await task_db.list_tasks(user_id, parent_task_id=parent.id)
        assert {task.id for task in under} == {child.id for child in children}
        refreshed = await task_db.get_task(parent.id, user_id)
        assert refreshed is not None and refreshed.subtask_count == MAX_SUBTASKS
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_parent_shows_its_subtasks_work_and_waits_for_them(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        parent, parent_chat = await _working_task(user_id)
        emails, emails_chat = await _working_task(user_id, "Draft emails", parent.id)
        copy, copy_chat = await _working_task(user_id, "Landing copy", parent.id)

        await record_db.record_tool_call(emails_chat, user_id, _call("web_search"))
        await record_db.open_asks(
            emails_chat, user_id, [AskInput(question="Which list?")]
        )
        await record_db.open_asks(
            copy_chat, user_id, [AskInput(question="Which tone?")]
        )
        record = await record_db.get_task_record(parent.id, user_id)
        assert record is not None
        # Two subtasks ask at once: the parent shows both.
        assert record.task.open_ask_count == 2
        assert {ask.task_id for ask in record.asks} == {emails.id, copy.id}
        assert any(line.task_id == emails.id for line in record.activity)

        # The parent's worker ends while a subtask is open: the parent stays.
        await record_db.settle_work_tasks(emails_chat, user_id, TaskStatus.DONE, "ok")
        await task_db.update_task(emails.id, user_id, TaskPatch(status=TaskStatus.DONE))
        assert (
            await record_db.settle_work_tasks(parent_chat, user_id, TaskStatus.DONE)
            == 0
        )
        # A failed subtask does not hold the parent back once it is settled.
        await task_db.update_task(copy.id, user_id, TaskPatch(status=TaskStatus.FAILED))
        assert (
            await record_db.settle_work_tasks(
                parent_chat, user_id, TaskStatus.DONE, "Done without the copy."
            )
            == 1
        )
        finished = await task_db.get_task(parent.id, user_id)
        assert finished is not None and finished.status == TaskStatus.DONE
        assert (finished.subtask_count, finished.subtasks_done) == (2, 1)
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_cancelling_a_parent_cancels_its_open_subtasks_only(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        parent, _ = await _working_task(user_id)
        running, running_chat = await _working_task(user_id, "Running", parent.id)
        finished, _ = await _working_task(user_id, "Finished", parent.id)
        await task_db.update_task(
            finished.id, user_id, TaskPatch(status=TaskStatus.DONE)
        )
        await record_db.open_asks(running_chat, user_id, [AskInput(question="Which?")])

        await task_db.update_task(
            parent.id, user_id, TaskPatch(status=TaskStatus.CANCELLED)
        )

        record = await record_db.get_task_record(parent.id, user_id)
        assert record is not None
        by_id = {task.id: task.status for task in record.subtasks}
        assert by_id[running.id] == TaskStatus.CANCELLED
        assert by_id[finished.id] == TaskStatus.DONE
        assert [ask.status for ask in record.asks] == [TaskAskStatus.WITHDRAWN]
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_asking_for_the_same_work_twice_finds_the_first_task(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        chat_id = await _chat(user_id)
        first = await task_db.create_task(
            user_id=user_id,
            title="2-day team offsite in Goa",
            created_by=TaskCreator.OTTO,
            session_id=chat_id,
        )
        found = await record_db.find_open_task_by_title(
            user_id, chat_id, "  2-Day team offsite   in goa "
        )
        assert found is not None and found.id == first.id
        assert (
            await record_db.find_open_task_by_title(user_id, chat_id, "Best CRM")
            is None
        )
        # Once it is settled, the same title starts a fresh task.
        await task_db.update_task(first.id, user_id, TaskPatch(status=TaskStatus.DONE))
        assert (
            await record_db.find_open_task_by_title(
                user_id, chat_id, "2-day team offsite in Goa"
            )
            is None
        )
        # Another chat asking for the same title is not a duplicate.
        other_chat = await _chat(user_id)
        again = await task_db.create_task(
            user_id=user_id,
            title="2-day team offsite in Goa",
            created_by=TaskCreator.OTTO,
            session_id=chat_id,
        )
        assert (
            await record_db.find_open_task_by_title(
                user_id, other_chat, "2-day team offsite in Goa"
            )
            is None
        )
        assert again.id != first.id
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_another_users_task_record_reads_as_missing(server: SpinTestServer):
    owner, stranger = await _user(), await _user()
    try:
        task, chat_id = await _working_task(owner)
        await record_db.record_tool_call(chat_id, owner, _call("web_search"))
        assert await record_db.get_task_record(task.id, stranger) is None
        # A stranger cannot write into it either, even naming the chat.
        assert not await record_db.record_tool_call(chat_id, stranger, _call("x"))
        assert await record_db.answer_session_asks(chat_id, stranger, "hi") == 0
        record = await record_db.get_task_record(task.id, owner)
        assert record is not None and len(record.activity) == 1
    finally:
        await _cleanup(owner, stranger)


@pytest.mark.asyncio(loop_scope="session")
async def test_calls_between_two_progress_lines_show_once_each_however_interleaved(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        task, chat_id = await _working_task(user_id)
        fetch = _call(
            "web_fetch", label="Read a web page", label_many="Read {count} web pages"
        )
        bash = _call(
            "bash_exec", label="Ran a command", label_many="Ran {count} commands"
        )
        await record_db.set_progress(chat_id, user_id, "Reading the pricing pages")
        for call in (fetch, bash, fetch, bash, fetch):
            await record_db.record_tool_call(chat_id, user_id, call)
        await record_db.set_progress(chat_id, user_id, "Writing the brief")
        # Said again: no second line.
        await record_db.set_progress(chat_id, user_id, "Writing the brief")
        await record_db.record_tool_call(chat_id, user_id, fetch)

        record = await record_db.get_task_record(task.id, user_id)
        assert record is not None
        assert [line.text for line in reversed(record.activity)] == [
            "Reading the pricing pages",
            "Read 3 web pages",
            "Ran 2 commands",
            "Writing the brief",
            "Read a web page",
        ]
        assert [line.is_milestone for line in reversed(record.activity)] == [
            True,
            False,
            False,
            True,
            False,
        ]
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_page_the_task_read_verifies_a_link_given_as_proof(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        _, chat_id = await _working_task(user_id)
        for url in ("https://cursor.com/pricing", "https://www.x.ai/api/"):
            await record_db.record_tool_call(
                chat_id, user_id, _call("web_fetch", label="Read a web page", url=url)
            )
        read = await record_db.add_evidence(
            chat_id,
            user_id,
            EvidenceInput(
                claim="Pro is $20 a month", url="http://www.cursor.com/pricing/"
            ),
        )
        folded = await record_db.add_evidence(
            chat_id,
            user_id,
            EvidenceInput(claim="API is priced per token", url="https://x.ai/api"),
        )
        unread = await record_db.add_evidence(
            chat_id,
            user_id,
            EvidenceInput(claim="Free tier", url="https://cursor.com/free"),
        )
        assert read and read.verified and read.kind == TaskEvidenceKind.LINK
        assert folded and folded.verified
        assert unread and not unread.verified
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_finishing_leaves_a_short_result_and_the_file_as_the_tasks_document(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        task, chat_id = await _working_task(user_id, done_when="Brief saved")
        await record_db.set_progress(chat_id, user_id, "Writing the brief")
        await record_db.record_tool_call(
            chat_id,
            user_id,
            _call(
                "write_workspace_file",
                artifact=ArtifactInput(
                    title="cursor-brief.md", workspace_file_id="f-1"
                ),
            ),
        )
        missing = await result_db.finish_task(chat_id, user_id, "Done.", "other.md")
        assert missing is not None and missing.missing_file
        assert (await task_db.get_task(task.id, user_id)).result is None

        finish = await result_db.finish_task(
            chat_id,
            user_id,
            "Brief ready: Cursor Pro is $20 a month.",
            "cursor-brief.md",
        )
        assert finish is not None and finish.document_title == "cursor-brief.md"
        # A chat the user started is not a background worker's.
        assert not finish.in_background
        assert not await result_db.is_worker_chat(chat_id, user_id)
        # The worker's last reply is long; the summary it left stands.
        await record_db.settle_work_tasks(
            chat_id, user_id, TaskStatus.DONE, "# Report\n\n" + "Lots of detail. " * 80
        )

        record = await record_db.get_task_record(task.id, user_id)
        assert record is not None
        assert record.task.result == "Brief ready: Cursor Pro is $20 a month."
        assert record.task.status_text is None
        assert [a.kind for a in record.artifacts] == [TaskArtifactKind.DOCUMENT]
        assert [(e.claim, e.verified) for e in record.evidence] == [
            ("Result saved as cursor-brief.md", True)
        ]
        assert not record.is_unverified

        outcome = await result_db.task_outcome(chat_id, user_id)
        assert outcome is not None
        assert (outcome.status, outcome.document_title, outcome.document_file_id) == (
            TaskStatus.DONE,
            "cursor-brief.md",
            "f-1",
        )
        assert outcome.result == "Brief ready: Cursor Pro is $20 a month."
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_worker_that_left_no_summary_gets_the_opening_of_its_reply(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        task, chat_id = await _working_task(user_id)
        reply = (
            "## Cursor brief\n\n**Pro is $20 a month.** " + "More detail here. " * 60
        )
        await record_db.settle_work_tasks(chat_id, user_id, TaskStatus.DONE, reply)
        result = (await task_db.get_task(task.id, user_id)).result
        assert result is not None and len(result) <= 400
        assert result.startswith("Cursor brief Pro is $20 a month.")
        assert "#" not in result and "*" not in result
        assert await result_db.finish_task(await _chat(user_id), user_id, "x") is None
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_calls_that_land_together_still_fold_into_one_line(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        task, chat_id = await _working_task(user_id)
        fetch = _call(
            "web_fetch", label="Read a web page", label_many="Read {count} web pages"
        )
        # One model step runs its tool calls at once.
        await asyncio.gather(
            *(record_db.record_tool_call(chat_id, user_id, fetch) for _ in range(4))
        )
        record = await record_db.get_task_record(task.id, user_id)
        assert record is not None
        assert [line.text for line in record.activity] == ["Read 4 web pages"]
    finally:
        await _cleanup(user_id)
