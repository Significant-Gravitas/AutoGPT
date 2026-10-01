"""Tests for the Capy client and the block paths the standard harness can't reach.

The harness runs one canned input per block with the client mocked out. These
cover parsing real (camelCase) payloads, the error envelope, the transcript
paging fallback, and the wait loop.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from backend.blocks.capy import _api
from backend.blocks.capy._api import CapyAPIError, CapyClient, _error
from backend.blocks.capy._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.capy._pull_requests import newest_pull_request_ref
from backend.blocks.capy._types import (
    Message,
    MessagePage,
    Project,
    ReviewRound,
    Thread,
)
from backend.blocks.capy.messages import CapyListThreadMessagesBlock
from backend.blocks.capy.threads import CapyListThreadsBlock
from backend.blocks.capy.usage import CapyGetUsageBlock
from backend.blocks.capy.wait import CapyWaitForThreadBlock

# Trimmed from a live GET /threads response.
LIVE_THREAD = {
    "id": "jam_01M0TQZWE84F58TN3793P1RMTV",
    "projectId": "be134130-4101-49dd-a5b3-efd3371b7229",
    "authorId": "user_3HT4A97W0EosvxZjfAToc3QNHlG",
    "previousAuthorId": None,
    "title": "Find PRs to close with reasons",
    "titleCustom": False,
    "status": "idle",
    "archived": False,
    "lastModelId": "supergrok/grok-4.5",
    "usage": {
        "llmCredits": 12.5,
        "imageCredits": 0,
        "vmCredits": 1,
        "totalCredits": 13.5,
    },
    "createdAt": "2026-08-24T20:39:32.781Z",
    "updatedAt": "2026-08-25T02:23:43.441Z",
    "lastActivityAt": "2026-08-25T02:23:43.441Z",
    "hasUnreads": False,
    "mentionCount": 0,
    "owed": False,
    "needsYou": True,
    "captain": None,
    "member": True,
}


def _response(status: int, body: Any) -> MagicMock:
    response = MagicMock()
    response.status = status
    response.ok = 200 <= status < 300
    response.json.return_value = body
    response.text.return_value = str(body)
    response.content = b"x" if body is not None else b""
    return response


class TestParsing:
    def test_thread_parses_camel_case_and_ignores_new_fields(self):
        thread = Thread.model_validate({**LIVE_THREAD, "brandNewField": 1})

        assert thread.project_id == LIVE_THREAD["projectId"]
        assert thread.needs_you is True
        assert thread.usage.total_credits == 13.5
        assert thread.last_model_id == "supergrok/grok-4.5"

    def test_review_round_parses_findings(self):
        review_round = ReviewRound.model_validate(
            {
                "requestId": "r1",
                "reviewId": "rev_1",
                "threadId": None,
                "repo": "acme/app",
                "prNumber": 7,
                "headSha": "a" * 40,
                "baseSha": "b" * 40,
                "status": "completed",
                "findings": [
                    {
                        "id": "f1",
                        "ruleId": None,
                        "kind": "issue",
                        "severity": "high",
                        "startLine": 3,
                        "line": 5,
                        "summary": "boom",
                    }
                ],
            }
        )

        assert review_round.pr_number == 7
        assert review_round.findings[0].start_line == 3

    def test_thread_and_project_link_into_the_capy_app(self):
        thread = Thread.model_validate(LIVE_THREAD)
        project = Project.model_validate(
            {"id": "be134130", "name": "AutoGPT", "code": "AGPT", "repos": []}
        )

        assert thread.url == f"https://capy.ai/thread/{LIVE_THREAD['id']}"
        assert project.environment_variables_url == (
            "https://capy.ai/settings/projects/be134130/environment-variables"
        )
        assert project.dev_environment_url.endswith("/be134130/dev-environment")

    def test_output_schema_uses_snake_case(self):
        schema = Thread.model_json_schema()
        assert "project_id" in schema["properties"]
        assert "projectId" not in schema["properties"]


class TestErrors:
    def test_tagged_error_names_the_cause(self):
        err = _error(_response(404, {"_tag": "capy/ThreadNotFound", "threadId": "x"}))

        assert err.tag == "capy/ThreadNotFound"
        assert err.status == 404
        assert "no thread with that ID" in str(err)

    def test_model_rejection_lists_candidates(self):
        err = _error(
            _response(
                400,
                {
                    "_tag": "ModelSelection.Rejected",
                    "message": "not connected",
                    "candidates": [{"entryId": "openai/gpt-6", "name": "GPT-6"}],
                },
            )
        )

        assert "not connected" in str(err)
        assert "openai/gpt-6" in str(err)

    @pytest.mark.parametrize(
        "candidates", [["openai/gpt-6", {"name": "Grok"}], "openai/gpt-6, Grok"]
    )
    def test_model_rejection_tolerates_plain_candidates(self, candidates):
        # A rejection must stay a tagged CapyAPIError whatever shape the
        # candidates take, or the balance fallback never sees the tag.
        err = _error(
            _response(
                400, {"_tag": "ModelSelection.Rejected", "candidates": candidates}
            )
        )

        assert err.tag == "ModelSelection.Rejected"
        assert "openai/gpt-6" in str(err) and "Grok" in str(err)

    def test_non_json_error_body(self):
        response = _response(502, None)
        response.json.side_effect = ValueError("not json")
        response.text.return_value = "Bad Gateway"

        err = _error(response)

        assert err.tag == ""
        assert "HTTP 502" in str(err) and "Bad Gateway" in str(err)


class TestClient:
    def test_retries_are_bounded_and_messages_are_sent_once(self):
        client = CapyClient(TEST_CREDENTIALS)

        # An outage must fail inside chat's five-minute block limit.
        assert client.requests.retry_max_attempts == 4
        assert client.requests.retry_max_wait <= 10
        assert client.requests_once.retry_max_attempts == 1

    async def test_send_message_is_never_retried(self):
        # Capy can't dedupe a message, so a retry after a gateway error could
        # hand the agent the same instruction twice.
        client = CapyClient(TEST_CREDENTIALS)
        client.requests = MagicMock()
        client.requests_once = MagicMock()
        client.requests_once.request = AsyncMock(
            return_value=_response(200, {"id": "01MSG", "deduped": False})
        )

        receipt = await client.send_message("t1", text="hi", delivery="queue")

        assert receipt.id == "01MSG"
        client.requests.request.assert_not_called()

    async def test_create_thread_sends_request_id_and_model(self):
        client = CapyClient(TEST_CREDENTIALS)
        client.requests = MagicMock()
        client.requests.request = AsyncMock(return_value=_response(200, LIVE_THREAD))

        await client.create_thread(
            project_id="p1",
            message="do it",
            model_id="openai/gpt-6",
            reasoning="high",
        )

        body = client.requests.request.call_args.kwargs["json"]
        assert body["projectId"] == "p1"
        assert body["requestId"]
        assert body["model"] == {"modelId": "openai/gpt-6", "reasoningMode": "high"}
        assert "machineSize" not in body
        assert "pullRequestAuthor" not in body

    async def test_create_thread_can_have_capy_open_the_pull_requests(self):
        client = CapyClient(TEST_CREDENTIALS)
        client.requests = MagicMock()
        client.requests.request = AsyncMock(
            return_value=_response(200, {**LIVE_THREAD, "pullRequestAuthor": "capy"})
        )

        thread = await client.create_thread(
            project_id="p1", message="do it", pull_request_author="capy"
        )

        body = client.requests.request.call_args.kwargs["json"]
        assert body["pullRequestAuthor"] == "capy"
        assert thread.pull_request_author == "capy"

    async def test_newest_messages_falls_back_to_forward_walk(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        client = CapyClient(TEST_CREDENTIALS)
        pages = [
            MessagePage(
                items=[Message(id=f"m{i}", source="assistant", text=str(i))],
                cursor=f"c{i}",
            )
            for i in range(3)
        ] + [MessagePage(items=[], cursor=None)]
        calls: list[dict] = []

        async def list_messages(thread_id, *, limit, after="", before=""):
            calls.append({"after": after, "before": before})
            if before:
                raise CapyAPIError(400, "capy/InvalidRequest", "bad cursor")
            return pages[len(calls) - 2]

        monkeypatch.setattr(client, "list_messages", list_messages)

        page = await client.newest_messages("t1", limit=2)

        assert [m.id for m in page.items] == ["m1", "m2"]
        assert page.cursor == "c2"

    async def test_newest_messages_reraises_other_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        client = CapyClient(TEST_CREDENTIALS)
        monkeypatch.setattr(
            client,
            "list_messages",
            AsyncMock(side_effect=CapyAPIError(401, "capy/Unauthorized", "nope")),
        )

        with pytest.raises(CapyAPIError):
            await client.newest_messages("t1", limit=5)

    async def test_newest_messages_fails_rather_than_return_an_older_page(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        client = CapyClient(TEST_CREDENTIALS)
        pages_read = 0

        async def list_messages(thread_id, *, limit, after="", before=""):
            nonlocal pages_read
            if before:
                raise CapyAPIError(400, "capy/InvalidRequest", "bad cursor")
            pages_read += 1
            return MessagePage(
                items=[Message(id=f"m{pages_read}", source="assistant", text="x")],
                cursor=f"c{pages_read}",
            )

        monkeypatch.setattr(client, "list_messages", list_messages)
        monkeypatch.setattr(_api, "_MAX_FORWARD_PAGES", 3)

        with pytest.raises(RuntimeError, match="newest messages"):
            await client.newest_messages("t1", limit=2)
        assert pages_read == 3


class TestPullRequestRefs:
    @pytest.mark.parametrize(
        "text,expected",
        [
            (
                "Closed https://github.com/acme/app/pull/7; opened PR #12.",
                ("", "12"),
            ),
            (
                "Opened https://github.com/acme/app/pull/12 (PR #12).",
                ("https://github.com/acme/app/pull/12", ""),
            ),
            (
                "PR #7 is replaced by https://github.com/acme/app/pull/12.",
                ("https://github.com/acme/app/pull/12", ""),
            ),
        ],
    )
    def test_the_last_reference_in_a_reply_wins(
        self, text: str, expected: tuple[str, str]
    ):
        message = Message(id="1", source="assistant", text=text)

        assert newest_pull_request_ref([message]) == expected


async def _run(block, **inputs) -> dict[str, Any]:
    collected: dict[str, Any] = {}
    async for name, value in block.run(
        block.input_schema(credentials=TEST_CREDENTIALS_INPUT, **inputs),
        credentials=TEST_CREDENTIALS,
    ):
        collected.setdefault(name, value)
    return collected


class TestWaitForThread:
    async def test_polls_until_idle(self, monkeypatch: pytest.MonkeyPatch):
        working = Thread.model_validate(
            {**LIVE_THREAD, "status": "working", "needsYou": False}
        )
        idle = Thread.model_validate(
            {**LIVE_THREAD, "status": "idle", "needsYou": False}
        )
        get_thread = AsyncMock(side_effect=[working, working, idle])
        monkeypatch.setattr(CapyClient, "get_thread", get_thread)
        monkeypatch.setattr(
            CapyClient,
            "newest_messages",
            AsyncMock(
                return_value=MessagePage(
                    items=[
                        Message(id="1", source="assistant", text="PR #12 is open"),
                        Message(id="2", source="tool", text="Push branch"),
                    ]
                )
            ),
        )
        monkeypatch.setattr(_api, "Requests", MagicMock())
        monkeypatch.setattr(
            CapyClient,
            "get_project",
            AsyncMock(side_effect=CapyAPIError(404, "capy/ProjectNotFound", "gone")),
        )
        sleep = AsyncMock()
        monkeypatch.setattr("backend.blocks.capy.wait.asyncio.sleep", sleep)

        out = await _run(CapyWaitForThreadBlock(), thread_id="t1", timeout_seconds=600)

        assert get_thread.await_count == 3
        assert out["finished"] is True
        assert out["status"] == "idle"
        assert out["last_reply"] == "PR #12 is open"

    async def test_keeps_waiting_while_the_brief_is_unanswered(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Capy reports a new thread idle for a moment before the agent picks
        # up the brief; that idle must not read as finished.
        idle = Thread.model_validate(
            {**LIVE_THREAD, "status": "idle", "needsYou": False}
        )
        working = Thread.model_validate(
            {**LIVE_THREAD, "status": "working", "needsYou": False}
        )
        get_thread = AsyncMock(side_effect=[idle, working, idle])
        brief = Message(id="1", source="user", text="Reply pong")
        newest = AsyncMock(
            side_effect=[
                MessagePage(items=[brief]),
                MessagePage(
                    items=[brief, Message(id="2", source="assistant", text="pong")]
                ),
            ]
        )
        monkeypatch.setattr(CapyClient, "get_thread", get_thread)
        monkeypatch.setattr(CapyClient, "newest_messages", newest)
        monkeypatch.setattr(_api, "Requests", MagicMock())
        monkeypatch.setattr("backend.blocks.capy.wait.asyncio.sleep", AsyncMock())

        out = await _run(CapyWaitForThreadBlock(), thread_id="t1", timeout_seconds=600)

        assert get_thread.await_count == 3
        assert out["finished"] is True
        assert out["last_reply"] == "pong"

    async def test_a_brand_new_thread_with_no_transcript_is_not_finished(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Seen live: two seconds after Create Thread, Capy reported the thread
        # idle while its transcript was still empty, and the wait returned
        # finished with no reply.
        idle = Thread.model_validate(
            {**LIVE_THREAD, "status": "idle", "needsYou": False}
        )
        brief = Message(id="1", source="user", text="Count the files")
        get_thread = AsyncMock(side_effect=[idle, idle, idle])
        newest = AsyncMock(
            side_effect=[
                MessagePage(items=[]),
                MessagePage(items=[brief]),
                MessagePage(
                    items=[brief, Message(id="2", source="assistant", text="12")]
                ),
            ]
        )
        monkeypatch.setattr(CapyClient, "get_thread", get_thread)
        monkeypatch.setattr(CapyClient, "newest_messages", newest)
        monkeypatch.setattr(_api, "Requests", MagicMock())
        monkeypatch.setattr("backend.blocks.capy.wait.asyncio.sleep", AsyncMock())

        out = await _run(CapyWaitForThreadBlock(), thread_id="t1", timeout_seconds=600)

        assert get_thread.await_count == 3
        assert out["finished"] is True
        assert out["last_reply"] == "12"

    async def test_stops_once_a_failed_thread_still_has_not_replied(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # A thread that failed before answering won't pick the brief up on
        # its own, so the second check ends the wait instead of the timeout.
        failed = Thread.model_validate(
            {**LIVE_THREAD, "status": "failed", "needsYou": False}
        )
        # Two reads only: a third means the wait kept polling.
        get_thread = AsyncMock(side_effect=[failed, failed])
        monkeypatch.setattr(CapyClient, "get_thread", get_thread)
        monkeypatch.setattr(
            CapyClient,
            "newest_messages",
            AsyncMock(
                return_value=MessagePage(
                    items=[Message(id="1", source="user", text="Fix the bug")]
                )
            ),
        )
        monkeypatch.setattr(_api, "Requests", MagicMock())
        monkeypatch.setattr("backend.blocks.capy.wait.asyncio.sleep", AsyncMock())

        out = await _run(CapyWaitForThreadBlock(), thread_id="t1", timeout_seconds=600)

        assert get_thread.await_count == 2
        assert out["finished"] is True
        assert out["status"] == "failed"

    async def test_keeps_waiting_when_a_failed_thread_takes_a_new_message(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # A failed thread sent a follow-up reads failed until the agent picks
        # it up, just as an idle one reads idle.
        failed, working, idle = (
            Thread.model_validate({**LIVE_THREAD, "status": s, "needsYou": False})
            for s in ("failed", "working", "idle")
        )
        get_thread = AsyncMock(side_effect=[failed, working, idle])
        retry = Message(id="1", source="user", text="Try again")
        monkeypatch.setattr(CapyClient, "get_thread", get_thread)
        monkeypatch.setattr(
            CapyClient,
            "newest_messages",
            AsyncMock(
                side_effect=[
                    MessagePage(items=[retry]),
                    MessagePage(
                        items=[retry, Message(id="2", source="assistant", text="Done")]
                    ),
                ]
            ),
        )
        monkeypatch.setattr(_api, "Requests", MagicMock())
        monkeypatch.setattr("backend.blocks.capy.wait.asyncio.sleep", AsyncMock())

        out = await _run(CapyWaitForThreadBlock(), thread_id="t1", timeout_seconds=600)

        assert get_thread.await_count == 3
        assert out["finished"] is True
        assert out["last_reply"] == "Done"

    async def test_a_follow_up_in_progress_keeps_the_threads_model(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # The newest reply is from the turn before the follow-up, so its model
        # says nothing about the one working now.
        working = Thread.model_validate(
            {
                **LIVE_THREAD,
                "status": "working",
                "needsYou": False,
                "lastModelId": "supergrok/grok-4.5",
            }
        )
        monkeypatch.setattr(CapyClient, "get_thread", AsyncMock(return_value=working))
        monkeypatch.setattr(
            CapyClient,
            "newest_messages",
            AsyncMock(
                return_value=MessagePage(
                    items=[
                        Message(
                            id="1",
                            source="assistant",
                            text="ok",
                            model="meta/muse-spark-1.3",
                        ),
                        Message(id="2", source="user", text="Redo it on Grok."),
                    ]
                )
            ),
        )
        monkeypatch.setattr(_api, "Requests", MagicMock())

        out = await _run(CapyWaitForThreadBlock(), thread_id="t1", timeout_seconds=0)

        assert out["finished"] is False
        assert out["model_id"] == "supergrok/grok-4.5"

    async def test_reports_the_model_that_wrote_the_reply(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Right after a model switch Capy's lastModelId can still name the old
        # model; the reply itself records the one that answered.
        idle = Thread.model_validate(
            {
                **LIVE_THREAD,
                "status": "idle",
                "needsYou": False,
                "lastModelId": "meta/muse-spark-1.3",
            }
        )
        monkeypatch.setattr(CapyClient, "get_thread", AsyncMock(return_value=idle))
        monkeypatch.setattr(
            CapyClient,
            "newest_messages",
            AsyncMock(
                return_value=MessagePage(
                    items=[
                        Message(id="1", source="user", text="Say ok again."),
                        Message(
                            id="2",
                            source="assistant",
                            text="ok",
                            model="supergrok/grok-4.5",
                        ),
                    ]
                )
            ),
        )
        monkeypatch.setattr(_api, "Requests", MagicMock())

        out = await _run(CapyWaitForThreadBlock(), thread_id="t1")

        assert out["model_id"] == "supergrok/grok-4.5"
        assert out["billed_via"] == "SuperGrok subscription"

    async def test_finds_the_pr_link_in_an_earlier_reply(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        idle = Thread.model_validate(
            {**LIVE_THREAD, "status": "idle", "needsYou": False}
        )
        monkeypatch.setattr(CapyClient, "get_thread", AsyncMock(return_value=idle))
        monkeypatch.setattr(
            CapyClient,
            "newest_messages",
            AsyncMock(
                return_value=MessagePage(
                    items=[
                        Message(
                            id="1",
                            source="assistant",
                            text="Opened https://github.com/acme/app/pull/7 (draft).",
                        ),
                        Message(
                            id="2",
                            source="user",
                            text="See https://github.com/acme/app/pull/99",
                        ),
                        Message(
                            id="3",
                            source="assistant",
                            text="Opened https://github.com/acme/app/pull/12 instead.",
                        ),
                        Message(id="4", source="assistant", text="CI is green."),
                    ]
                )
            ),
        )
        monkeypatch.setattr(_api, "Requests", MagicMock())

        out = await _run(CapyWaitForThreadBlock(), thread_id="t1")

        assert out["pull_request_url"] == "https://github.com/acme/app/pull/12"
        assert out["last_reply"] == "CI is green."

    @pytest.mark.parametrize(
        "repos,expected",
        [
            (
                [{"repoFullName": "significant-gravitas/autogpt"}],
                "https://github.com/significant-gravitas/autogpt/pull/14992",
            ),
            ([{"repoFullName": "a/one"}, {"repoFullName": "a/two"}], None),
        ],
    )
    async def test_resolves_a_bare_pr_number_against_the_project(
        self, monkeypatch: pytest.MonkeyPatch, repos: list, expected: str | None
    ):
        # Live Capy replies often say "PR #14992 is open" with no URL.
        idle = Thread.model_validate(
            {**LIVE_THREAD, "status": "idle", "needsYou": False}
        )
        monkeypatch.setattr(CapyClient, "get_thread", AsyncMock(return_value=idle))
        monkeypatch.setattr(
            CapyClient,
            "newest_messages",
            AsyncMock(
                return_value=MessagePage(
                    items=[
                        Message(
                            id="1",
                            source="assistant",
                            text="PR #14992 is open against `dev`.",
                        )
                    ]
                )
            ),
        )
        get_project = AsyncMock(
            return_value=Project.model_validate(
                {"id": "p", "name": "AutoGPT", "repos": repos}
            )
        )
        monkeypatch.setattr(CapyClient, "get_project", get_project)
        monkeypatch.setattr(_api, "Requests", MagicMock())

        out = await _run(CapyWaitForThreadBlock(), thread_id="t1")

        get_project.assert_awaited_once_with(LIVE_THREAD["projectId"])
        assert out.get("pull_request_url") == expected

    async def test_no_pr_link_emits_nothing(self, monkeypatch: pytest.MonkeyPatch):
        idle = Thread.model_validate(
            {**LIVE_THREAD, "status": "idle", "needsYou": False}
        )
        monkeypatch.setattr(CapyClient, "get_thread", AsyncMock(return_value=idle))
        monkeypatch.setattr(
            CapyClient,
            "newest_messages",
            AsyncMock(
                return_value=MessagePage(
                    items=[Message(id="1", source="assistant", text="pong")]
                )
            ),
        )
        monkeypatch.setattr(_api, "Requests", MagicMock())

        out = await _run(CapyWaitForThreadBlock(), thread_id="t1")

        assert "pull_request_url" not in out

    async def test_waits_for_the_reply_to_the_message_it_was_given(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # A queued follow-up isn't the last entry yet, and the agent's reply
        # to the previous turn still is, so only the message ID tells the
        # new reply apart from the old one.
        idle = Thread.model_validate(
            {**LIVE_THREAD, "status": "idle", "needsYou": False}
        )
        earlier = [
            Message(id="01A", source="user", text="Fix the bug"),
            Message(id="01B", source="assistant", text="Fixed", model="meta/m1"),
        ]
        answer = Message(id="01D", source="assistant", text="Tests added", model="x/m2")
        get_thread = AsyncMock(side_effect=[idle, idle])
        newest = AsyncMock(
            side_effect=[
                MessagePage(items=earlier),
                MessagePage(
                    items=[
                        *earlier,
                        Message(id="01C", source="user", text="Add tests"),
                        answer,
                    ]
                ),
            ]
        )
        monkeypatch.setattr(CapyClient, "get_thread", get_thread)
        monkeypatch.setattr(CapyClient, "newest_messages", newest)
        monkeypatch.setattr(_api, "Requests", MagicMock())
        monkeypatch.setattr("backend.blocks.capy.wait.asyncio.sleep", AsyncMock())

        out = await _run(
            CapyWaitForThreadBlock(),
            thread_id="t1",
            timeout_seconds=600,
            after_message_id="01C",
        )

        assert get_thread.await_count == 2
        assert out["finished"] is True
        assert out["last_reply"] == "Tests added"
        assert out["model_id"] == "x/m2"

    async def test_no_reply_yet_to_the_given_message_reads_as_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        working = Thread.model_validate(
            {**LIVE_THREAD, "status": "working", "needsYou": False}
        )
        monkeypatch.setattr(CapyClient, "get_thread", AsyncMock(return_value=working))
        monkeypatch.setattr(
            CapyClient,
            "newest_messages",
            AsyncMock(
                return_value=MessagePage(
                    items=[
                        Message(id="01B", source="assistant", text="Fixed"),
                        Message(id="01C", source="user", text="Add tests"),
                    ]
                )
            ),
        )
        monkeypatch.setattr(_api, "Requests", MagicMock())

        out = await _run(
            CapyWaitForThreadBlock(),
            thread_id="t1",
            timeout_seconds=0,
            after_message_id="01C",
        )

        assert out["finished"] is False
        assert out["last_reply"] == ""
        assert out["thread_url"] == f"https://capy.ai/thread/{LIVE_THREAD['id']}"

    async def test_stops_when_the_agent_needs_an_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        asking = Thread.model_validate(
            {**LIVE_THREAD, "status": "waiting", "needsYou": True}
        )
        monkeypatch.setattr(CapyClient, "get_thread", AsyncMock(return_value=asking))
        monkeypatch.setattr(
            CapyClient, "newest_messages", AsyncMock(return_value=MessagePage())
        )
        monkeypatch.setattr(_api, "Requests", MagicMock())

        out = await _run(CapyWaitForThreadBlock(), thread_id="t1")

        assert out["finished"] is True
        assert out["needs_you"] is True

    async def test_zero_timeout_returns_current_state(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        working = Thread.model_validate(
            {**LIVE_THREAD, "status": "working", "needsYou": False}
        )
        monkeypatch.setattr(CapyClient, "get_thread", AsyncMock(return_value=working))
        monkeypatch.setattr(
            CapyClient, "newest_messages", AsyncMock(return_value=MessagePage())
        )
        monkeypatch.setattr(_api, "Requests", MagicMock())

        out = await _run(CapyWaitForThreadBlock(), thread_id="t1", timeout_seconds=0)

        assert out["finished"] is False
        assert out["status"] == "working"


class TestListThreads:
    @staticmethod
    def _board() -> list[Thread]:
        return [
            Thread.model_validate({**LIVE_THREAD, "id": t, "status": s, "needsYou": n})
            for t, s, n in [
                ("busy", "working", False),
                ("asking", "waiting", True),
                ("done", "idle", False),
                ("broke", "failed", False),
            ]
        ]

    @pytest.mark.parametrize(
        "show, expected",
        [("active", ["busy", "asking"]), ("needs_you", ["asking"])],
    )
    async def test_a_filter_keeps_only_matching_threads_from_a_full_page(
        self, monkeypatch: pytest.MonkeyPatch, show: str, expected: list[str]
    ):
        block = CapyListThreadsBlock()
        list_threads = AsyncMock(return_value=(self._board(), "5:next"))
        monkeypatch.setattr(block, "list_threads", list_threads)

        out = await _run(block, project_id="p1", show=show)

        assert [t.id for t in out["threads"]] == expected
        assert out["next_cursor"] == "5:next"
        list_threads.assert_awaited_once_with(TEST_CREDENTIALS, "p1", 100, "")

    async def test_all_lists_the_requested_number(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = CapyListThreadsBlock()
        list_threads = AsyncMock(return_value=(self._board(), None))
        monkeypatch.setattr(block, "list_threads", list_threads)

        out = await _run(block, project_id="p1", limit=4)

        assert len(out["threads"]) == 4
        assert out["next_cursor"] == ""
        list_threads.assert_awaited_once_with(TEST_CREDENTIALS, "p1", 4, "")

    async def test_no_match_still_reports_an_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = CapyListThreadsBlock()
        monkeypatch.setattr(
            block, "list_threads", AsyncMock(return_value=(self._board()[2:], None))
        )

        out = await _run(block, project_id="p1", show="active")

        assert out["threads"] == []
        assert "thread" not in out


class TestListThreadMessages:
    async def test_cursor_falls_back_to_last_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = CapyListThreadMessagesBlock()
        monkeypatch.setattr(
            block,
            "list_messages",
            AsyncMock(
                return_value=MessagePage(
                    items=[Message(id="01ABC", source="assistant", text="hi")],
                    cursor=None,
                )
            ),
        )

        out = await _run(block, thread_id="t1")

        assert out["next_cursor"] == "01ABC"

    async def test_empty_page_keeps_callers_cursor(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = CapyListThreadMessagesBlock()
        monkeypatch.setattr(
            block, "list_messages", AsyncMock(return_value=MessagePage(cursor=None))
        )

        out = await _run(block, thread_id="t1", after_cursor="01OLD")

        assert out["next_cursor"] == "01OLD"
        assert out["messages"] == []

    async def test_pages_back_from_a_before_cursor(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        list_messages = AsyncMock(
            return_value=MessagePage(
                items=[Message(id="01B", source="assistant", text="earlier")],
                cursor="01B",
                before_cursor="01A",
            )
        )
        monkeypatch.setattr(CapyClient, "list_messages", list_messages)
        monkeypatch.setattr(_api, "Requests", MagicMock())

        out = await _run(
            CapyListThreadMessagesBlock(), thread_id="t1", before_cursor="01C"
        )

        list_messages.assert_awaited_once_with("t1", limit=20, after="", before="01C")
        assert out["last_reply"] == "earlier"
        assert out["older_cursor"] == "01A"


class TestGetUsage:
    async def test_reports_a_genuine_zero(self, monkeypatch: pytest.MonkeyPatch):
        block = CapyGetUsageBlock()
        monkeypatch.setattr(
            block,
            "get_usage",
            AsyncMock(return_value={"totals": {"totalDollars": 0}}),
        )

        out = await _run(block)

        assert out["total_dollars"] == 0.0

    @pytest.mark.parametrize(
        "report",
        [
            {},
            {"totals": {"totalDollars": None}},
            {"totals": {"totalDollars": "NaN"}},
        ],
    )
    async def test_refuses_a_missing_or_non_finite_total(
        self, monkeypatch: pytest.MonkeyPatch, report: dict
    ):
        # A budget check reading 0.0 here would treat unknown spend as none.
        block = CapyGetUsageBlock()
        monkeypatch.setattr(block, "get_usage", AsyncMock(return_value=report))

        with pytest.raises(ValidationError):
            await _run(block)
