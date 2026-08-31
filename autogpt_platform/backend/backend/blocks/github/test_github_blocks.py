import asyncio
import inspect
from unittest import mock

import pytest

from backend.blocks.github import notifications, pull_requests
from backend.blocks.github._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.github.commits import FileOperation, GithubMultiFileCommitBlock
from backend.blocks.github.notifications import (
    TEST_NOTIFICATION_ITEM,
    GithubListNotificationsBlock,
    GithubMarkNotificationsAsReadBlock,
    _notifications_url,
    _subject_html_url,
    _to_notification_item,
)
from backend.blocks.github.pull_requests import (
    GithubListPRReviewersBlock,
    GithubMergePullRequestBlock,
    prepare_pr_api_url,
)
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError

# ── prepare_pr_api_url tests ──


class TestPreparePrApiUrl:
    def test_https_scheme_preserved(self):
        result = prepare_pr_api_url("https://github.com/owner/repo/pull/42", "merge")
        assert result == "https://github.com/owner/repo/pulls/42/merge"

    def test_http_scheme_preserved(self):
        result = prepare_pr_api_url("http://github.com/owner/repo/pull/1", "files")
        assert result == "http://github.com/owner/repo/pulls/1/files"

    def test_no_scheme_defaults_to_https(self):
        result = prepare_pr_api_url("github.com/owner/repo/pull/5", "merge")
        assert result == "https://github.com/owner/repo/pulls/5/merge"

    def test_reviewers_path(self):
        result = prepare_pr_api_url(
            "https://github.com/owner/repo/pull/99", "requested_reviewers"
        )
        assert result == "https://github.com/owner/repo/pulls/99/requested_reviewers"

    def test_invalid_url_returned_as_is(self):
        url = "https://example.com/not-a-pr"
        assert prepare_pr_api_url(url, "merge") == url

    def test_empty_string(self):
        assert prepare_pr_api_url("", "merge") == ""


# ── Error-path block tests ──
# When a block's run() yields ("error", msg), _execute() converts it to a
# BlockExecutionError. We call block.execute() directly (not execute_block_test,
# which returns early on empty test_output).


def _mock_block(block, mocks: dict):
    """Apply mocks to a block's static methods, wrapping sync mocks as async."""
    for name, mock_fn in mocks.items():
        original = getattr(block, name)
        if inspect.iscoroutinefunction(original):

            async def async_mock(*args, _fn=mock_fn, **kwargs):
                return _fn(*args, **kwargs)

            setattr(block, name, async_mock)
        else:
            setattr(block, name, mock_fn)


def _raise(exc: Exception):
    """Helper that returns a callable which raises the given exception."""

    def _raiser(*args, **kwargs):
        raise exc

    return _raiser


@pytest.mark.asyncio
async def test_merge_pr_error_path():
    block = GithubMergePullRequestBlock()
    _mock_block(block, {"merge_pr": _raise(RuntimeError("PR not mergeable"))})
    input_data = {
        "pr_url": "https://github.com/owner/repo/pull/1",
        "merge_method": "squash",
        "commit_title": "",
        "commit_message": "",
        "credentials": TEST_CREDENTIALS_INPUT,
    }
    with pytest.raises(BlockExecutionError, match="PR not mergeable"):
        async for _ in block.execute(
            input_data,
            credentials=TEST_CREDENTIALS,
            execution_context=ExecutionContext(),
        ):
            pass


@pytest.mark.asyncio
async def test_multi_file_commit_error_path():
    block = GithubMultiFileCommitBlock()
    _mock_block(block, {"multi_file_commit": _raise(RuntimeError("ref update failed"))})
    input_data = {
        "repo_url": "https://github.com/owner/repo",
        "branch": "feature",
        "commit_message": "test",
        "files": [{"path": "a.py", "content": "x", "operation": "upsert"}],
        "credentials": TEST_CREDENTIALS_INPUT,
    }
    with pytest.raises(BlockExecutionError, match="ref update failed"):
        async for _ in block.execute(
            input_data,
            credentials=TEST_CREDENTIALS,
            execution_context=ExecutionContext(),
        ):
            pass


# ── FileOperation enum tests ──


class TestFileOperation:
    def test_upsert_value(self):
        assert FileOperation.UPSERT == "upsert"

    def test_delete_value(self):
        assert FileOperation.DELETE == "delete"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            FileOperation("create")

    def test_invalid_value_raises_typo(self):
        with pytest.raises(ValueError):
            FileOperation("upser")


# ── _subject_html_url tests ──
# These helpers are mocked out in every block test, so they are covered here
# directly against realistic GitHub API payloads.


class TestSubjectHtmlUrl:
    def test_issue_url(self):
        assert (
            _subject_html_url("https://api.github.com/repos/owner/repo/issues/42")
            == "https://github.com/owner/repo/issues/42"
        )

    def test_pull_request_url_is_singularised(self):
        assert (
            _subject_html_url("https://api.github.com/repos/owner/repo/pulls/42")
            == "https://github.com/owner/repo/pull/42"
        )

    def test_commit_url_is_singularised(self):
        # /commits/{sha} resolves to the ref's history page, not the commit
        assert (
            _subject_html_url("https://api.github.com/repos/owner/repo/commits/abc123")
            == "https://github.com/owner/repo/commit/abc123"
        )

    def test_release_id_falls_back_to_release_index(self):
        # Releases resolve by tag on the web; the numeric id 404s
        assert (
            _subject_html_url("https://api.github.com/repos/owner/repo/releases/9876")
            == "https://github.com/owner/repo/releases"
        )

    def test_repo_named_pulls_is_not_rewritten(self):
        assert (
            _subject_html_url("https://api.github.com/repos/owner/pulls/pulls/1")
            == "https://github.com/owner/pulls/pull/1"
        )

    def test_empty_url(self):
        assert _subject_html_url("") == ""

    def test_non_repo_url_passes_through(self):
        url = "https://api.github.com/user/12345"
        assert _subject_html_url(url) == url


# ── _to_notification_item tests ──

RAW_NOTIFICATION = {
    "id": "1337",
    "unread": True,
    "reason": "review_requested",
    "updated_at": "2026-07-21T12:00:00Z",
    "last_read_at": None,
    "subject": {
        "title": "Fix the flux capacitor",
        "url": "https://api.github.com/repos/owner/repo/pulls/1",
        "latest_comment_url": "https://api.github.com/repos/owner/repo/pulls/1",
        "type": "PullRequest",
    },
    "repository": {
        "id": 1296269,
        "name": "repo",
        "full_name": "owner/repo",
        "private": False,
    },
    "url": "https://api.github.com/notifications/threads/1337",
}


class TestToNotificationItem:
    def test_maps_a_realistic_payload(self):
        assert _to_notification_item(RAW_NOTIFICATION) == TEST_NOTIFICATION_ITEM

    def test_null_repository_degrades_to_empty_string(self):
        item = _to_notification_item({**RAW_NOTIFICATION, "repository": None})
        assert item["repository"] == ""
        assert item["thread_id"] == "1337"

    def test_missing_subject_degrades_to_empty_strings(self):
        item = _to_notification_item({**RAW_NOTIFICATION, "subject": None})
        assert item["title"] == ""
        assert item["subject_type"] == ""
        assert item["subject_url"] == ""


# ── _notifications_url tests ──


class TestNotificationsUrl:
    def test_global_scope(self):
        assert _notifications_url("") == "https://api.github.com/notifications"

    def test_repo_scope(self):
        assert (
            _notifications_url("owner/repo")
            == "https://api.github.com/repos/owner/repo/notifications"
        )


# ── Request-building tests ──
# The blocks' static methods are mocked wholesale in block tests, so the
# param/body construction inside them is exercised here instead.


def _list_notifications_call(**input_kwargs) -> dict:
    """Runs list_notifications with the network stubbed; returns the call args."""
    captured: dict = {}

    async def fake_get_paginated(api, url, **kwargs):
        captured.update(url=url, **kwargs)
        return []

    input_data = GithubListNotificationsBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **input_kwargs}
    )
    with (
        mock.patch.object(notifications, "get_api", lambda *a, **kw: object()),
        mock.patch.object(notifications, "get_paginated", fake_get_paginated),
    ):
        asyncio.run(
            GithubListNotificationsBlock.list_notifications(
                TEST_CREDENTIALS, input_data
            )
        )
    return captured


class TestListNotificationsRequest:
    def test_defaults_send_no_filters(self):
        call = _list_notifications_call()
        assert call["url"] == "https://api.github.com/notifications"
        assert call["params"] == {}

    def test_global_endpoint_caps_page_size_at_50(self):
        assert _list_notifications_call()["max_page_size"] == 50

    def test_repo_endpoint_allows_the_usual_100(self):
        call = _list_notifications_call(repo="owner/repo")
        assert call["url"] == "https://api.github.com/repos/owner/repo/notifications"
        assert call["max_page_size"] == 100

    def test_flags_and_timestamps_become_params(self):
        call = _list_notifications_call(
            include_read=True,
            participating_only=True,
            since="2026-01-01T00:00:00Z",
            before="2026-02-01T00:00:00Z",
        )
        assert call["params"] == {
            "all": "true",
            "participating": "true",
            "since": "2026-01-01T00:00:00Z",
            "before": "2026-02-01T00:00:00Z",
        }

    def test_limit_is_forwarded(self):
        assert _list_notifications_call(limit=7)["limit"] == 7


class _FakeApi:
    """Captures the request a block builds, and replays canned responses."""

    def __init__(self, responses: dict[str, object] | None = None):
        self.responses = responses or {}
        self.calls: list[tuple[str, str, dict]] = []

    async def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        return mock.Mock(json=lambda: self.responses.get(url, {}))

    async def put(self, url, **kwargs):
        self.calls.append(("put", url, kwargs))
        return mock.Mock(json=dict)


class TestMarkAllAsReadRequest:
    def _run(self, repo: str, last_read_at: str) -> tuple[str, str, dict]:
        api = _FakeApi()
        with mock.patch.object(notifications, "get_api", lambda *a, **kw: api):
            asyncio.run(
                GithubMarkNotificationsAsReadBlock.mark_all_as_read(
                    TEST_CREDENTIALS, repo, last_read_at
                )
            )
        return api.calls[0]

    def test_global_scope_with_empty_body(self):
        method, url, kwargs = self._run("", "")
        assert (method, url) == ("put", "https://api.github.com/notifications")
        assert kwargs["json"] == {}

    def test_repo_scope_with_last_read_at(self):
        method, url, kwargs = self._run("owner/repo", "2026-01-01T00:00:00Z")
        assert url == "https://api.github.com/repos/owner/repo/notifications"
        assert kwargs["json"] == {"last_read_at": "2026-01-01T00:00:00Z"}


# ── list_reviewers merge/dedup tests ──

PR_URL = "https://github.com/owner/repo/pull/1"
REQUESTED_REVIEWERS_URL = "https://github.com/owner/repo/pulls/1/requested_reviewers"


def _list_reviewers(
    reviews: list[dict], include_past: bool = True
) -> list[GithubListPRReviewersBlock.Output.ReviewerItem]:
    api = _FakeApi(
        {
            REQUESTED_REVIEWERS_URL: {
                "users": [
                    {
                        "login": "pending_alice",
                        "html_url": "https://github.com/pending_alice",
                    }
                ]
            }
        }
    )

    async def fake_get_paginated(_api, _url, **kwargs):
        return reviews

    with (
        mock.patch.object(pull_requests, "get_api", lambda *a, **kw: api),
        mock.patch.object(pull_requests, "get_paginated", fake_get_paginated),
    ):
        return asyncio.run(
            GithubListPRReviewersBlock.list_reviewers(
                TEST_CREDENTIALS, PR_URL, include_past
            )
        )


def _review(login: str | None, state: str) -> dict:
    user = (
        {"login": login, "html_url": f"https://github.com/{login}"} if login else None
    )
    return {"user": user, "state": state}


class TestListReviewers:
    def test_past_reviewers_are_skipped_when_not_requested(self):
        assert _list_reviewers([_review("bob", "APPROVED")], include_past=False) == [
            {
                "username": "pending_alice",
                "url": "https://github.com/pending_alice",
                "review_requested": True,
                "has_reviewed": False,
                "review_state": "",
            }
        ]

    def test_past_reviewer_is_appended(self):
        result = _list_reviewers([_review("bob", "APPROVED")])
        assert result[1] == {
            "username": "bob",
            "url": "https://github.com/bob",
            "review_requested": False,
            "has_reviewed": True,
            "review_state": "APPROVED",
        }

    def test_requested_reviewer_who_reviewed_is_updated_not_duplicated(self):
        result = _list_reviewers([_review("pending_alice", "CHANGES_REQUESTED")])
        assert len(result) == 1
        assert result[0]["review_requested"] is True
        assert result[0]["has_reviewed"] is True
        assert result[0]["review_state"] == "CHANGES_REQUESTED"

    def test_latest_review_state_wins(self):
        result = _list_reviewers(
            [_review("bob", "CHANGES_REQUESTED"), _review("bob", "APPROVED")]
        )
        assert len(result) == 2
        assert result[1]["review_state"] == "APPROVED"

    def test_pending_reviews_are_ignored(self):
        assert len(_list_reviewers([_review("bob", "PENDING")])) == 1

    def test_deleted_accounts_are_ignored(self):
        assert len(_list_reviewers([_review(None, "APPROVED")])) == 1
