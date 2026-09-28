"""Tests for the Capy client and the block paths the standard harness can't reach.

The harness runs one canned input per block with the client mocked out. These
cover parsing real (camelCase) payloads, the error envelope, the transcript
paging fallback, and the wait loop.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.blocks.capy import _api
from backend.blocks.capy._api import CapyAPIError, CapyClient, _error
from backend.blocks.capy._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.capy._types import (
    Message,
    MessagePage,
    Project,
    ReviewRound,
    Thread,
)
from backend.blocks.capy.messages import CapyListThreadMessagesBlock
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

    def test_non_json_error_body(self):
        response = _response(502, None)
        response.json.side_effect = ValueError("not json")
        response.text.return_value = "Bad Gateway"

        err = _error(response)

        assert err.tag == ""
        assert "HTTP 502" in str(err) and "Bad Gateway" in str(err)


class TestClient:
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
