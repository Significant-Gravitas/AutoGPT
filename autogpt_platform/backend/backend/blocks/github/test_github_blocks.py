import asyncio
import copy
import inspect
from datetime import datetime, timezone
from unittest import mock

import pytest

from backend.blocks.github import (
    commits,
    issues,
    notifications,
    pull_requests,
    repo,
    repo_branches,
    reviews,
    users,
)
from backend.blocks.github._api import (
    convert_comment_url_to_api_endpoint,
    get_paginated,
)
from backend.blocks.github._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.github._utils import normalize_repo_path
from backend.blocks.github.commits import (
    FileOperation,
    GithubListCommitsBlock,
    GithubMultiFileCommitBlock,
)
from backend.blocks.github.issues import GithubListCommentsBlock, GithubListIssuesBlock
from backend.blocks.github.notifications import (
    TEST_NOTIFICATION_ITEM,
    GithubGetNotificationThreadBlock,
    GithubListNotificationsBlock,
    GithubMarkNotificationsAsReadBlock,
    GithubMarkNotificationThreadAsDoneBlock,
    GithubMarkNotificationThreadAsReadBlock,
    GithubUnsubscribeNotificationThreadBlock,
    _notifications_url,
    _subject_html_url,
    _to_notification_item,
)
from backend.blocks.github.pull_requests import (
    TEST_PR_PAYLOAD,
    GithubListPRReviewersBlock,
    GithubListPullRequestsBlock,
    GithubMergePullRequestBlock,
    GithubReadPullRequestBlock,
    prepare_pr_api_url,
)
from backend.blocks.github.repo import (
    GithubListDiscussionsBlock,
    GithubListReleasesBlock,
    GithubListStargazersBlock,
    GithubListTagsBlock,
)
from backend.blocks.github.repo_branches import GithubListBranchesBlock
from backend.blocks.github.reviews import GithubListPRReviewsBlock
from backend.blocks.github.users import GithubGetUserInfoBlock
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

    def test_repo_url_is_accepted_too(self):
        assert (
            _notifications_url("https://github.com/owner/repo")
            == "https://api.github.com/repos/owner/repo/notifications"
        )

    def test_malformed_repo_is_rejected(self):
        with pytest.raises(ValueError):
            _notifications_url("owner/repo/../../secrets")


# ── normalize_repo_path tests ──
# The `repo` input on the notification blocks accepts both the bare
# {owner}/{repo} form and a full repo URL, so both are normalised here.


class TestNormalizeRepoPath:
    def test_bare_owner_repo(self):
        assert normalize_repo_path("owner/repo") == "owner/repo"

    def test_https_url(self):
        assert normalize_repo_path("https://github.com/owner/repo") == "owner/repo"

    def test_http_url(self):
        assert normalize_repo_path("http://github.com/owner/repo") == "owner/repo"

    def test_www_host(self):
        assert normalize_repo_path("https://www.github.com/owner/repo") == "owner/repo"

    def test_schemeless_url(self):
        assert normalize_repo_path("github.com/owner/repo") == "owner/repo"

    def test_trailing_slash(self):
        assert normalize_repo_path("https://github.com/owner/repo/") == "owner/repo"

    def test_git_suffix_is_dropped(self):
        assert normalize_repo_path("https://github.com/owner/repo.git") == "owner/repo"

    def test_surrounding_whitespace(self):
        assert normalize_repo_path("  owner/repo  ") == "owner/repo"

    def test_dots_and_dashes_in_repo_name(self):
        assert normalize_repo_path("my-org/my.repo_v2") == "my-org/my.repo_v2"

    def test_deep_link_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid repository"):
            normalize_repo_path("https://github.com/owner/repo/tree/main")

    def test_traversal_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid repository"):
            normalize_repo_path("owner/../../notifications")

    def test_extra_path_segment_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid repository"):
            normalize_repo_path("owner/repo/notifications")

    def test_bare_owner_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid repository"):
            normalize_repo_path("owner")

    def test_empty_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid repository"):
            normalize_repo_path("")

    def test_non_github_host_is_rejected(self):
        with pytest.raises(ValueError, match="Not a github.com repository URL"):
            normalize_repo_path("https://evil.example.com/owner/repo")

    def test_query_string_is_not_smuggled_into_the_path(self):
        assert (
            normalize_repo_path("https://github.com/owner/repo?tab=readme")
            == "owner/repo"
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

    async def patch(self, url, **kwargs):
        self.calls.append(("patch", url, kwargs))
        return mock.Mock(json=dict)

    async def delete(self, url, **kwargs):
        self.calls.append(("delete", url, kwargs))
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


# ── read_pr tests ──


PR_ENDPOINT_URL = "https://github.com/owner/repo/pulls/1"


class TestReadPr:
    def test_hits_the_pulls_endpoint_not_the_issues_endpoint(self):
        api = _FakeApi({PR_ENDPOINT_URL: {}})
        with mock.patch.object(pull_requests, "get_api", lambda *a, **kw: api):
            asyncio.run(GithubReadPullRequestBlock.read_pr(TEST_CREDENTIALS, PR_URL))
        assert api.calls[0][1] == PR_ENDPOINT_URL

    def test_returns_the_raw_response_body(self):
        api = _FakeApi({PR_ENDPOINT_URL: TEST_PR_PAYLOAD})
        with mock.patch.object(pull_requests, "get_api", lambda *a, **kw: api):
            result = asyncio.run(
                GithubReadPullRequestBlock.read_pr(TEST_CREDENTIALS, PR_URL)
            )
        assert result == TEST_PR_PAYLOAD


class TestPrSummary:
    def test_null_title_falls_back_instead_of_yielding_none(self):
        pr = {**TEST_PR_PAYLOAD, "title": None}
        assert GithubReadPullRequestBlock._pr_summary(pr)[0] == "No title found"

    def test_null_body_falls_back_instead_of_yielding_none(self):
        # GitHub returns an explicit "body": null for PRs with no description.
        pr = {**TEST_PR_PAYLOAD, "body": None}
        assert GithubReadPullRequestBlock._pr_summary(pr)[1] == (
            "No body content found"
        )

    def test_null_user_falls_back_instead_of_raising(self):
        # Reviews by deleted accounts come back with a null user.
        pr = {**TEST_PR_PAYLOAD, "user": None}
        assert GithubReadPullRequestBlock._pr_summary(pr)[2] == "Unknown author"


# ── get_paginated tests ──
# Every list block delegates its fetching here, and every block test mocks the
# list method wholesale, so the pagination logic is exercised directly.


class _PagedApi:
    """Serves canned pages and records the query params of each request."""

    def __init__(self, pages: list[list[dict]]):
        self.pages = pages
        self.requests: list[dict] = []

    async def get(self, url, **kwargs):
        params = kwargs.get("params") or {}
        self.requests.append({"url": url, **params})
        index = int(params["page"]) - 1
        page = self.pages[index] if index < len(self.pages) else []
        return mock.Mock(json=lambda: page)


def _paginate(pages: list[list[dict]], **kwargs) -> tuple[list[dict], _PagedApi]:
    api = _PagedApi(pages)
    items = asyncio.run(get_paginated(api, LIST_URL, **kwargs))  # type: ignore[arg-type]
    return items, api


LIST_URL = "https://api.github.com/repos/owner/repo/items"


class TestGetPaginated:
    def test_short_page_ends_the_fetch(self):
        items, api = _paginate([[{"n": 1}, {"n": 2}]], limit=10)
        assert items == [{"n": 1}, {"n": 2}]
        assert len(api.requests) == 1

    def test_empty_first_page_returns_nothing(self):
        items, api = _paginate([[]], limit=10)
        assert items == []
        assert len(api.requests) == 1

    def test_full_pages_are_followed_and_truncated_to_limit(self):
        pages = [[{"n": 1}, {"n": 2}, {"n": 3}], [{"n": 4}, {"n": 5}, {"n": 6}]]
        items, api = _paginate(pages, limit=4, max_page_size=3)
        assert [i["n"] for i in items] == [1, 2, 3, 4]
        assert [r["page"] for r in api.requests] == ["1", "2"]

    def test_page_size_is_capped_by_limit_when_unfiltered(self):
        _, api = _paginate([[]], limit=5)
        assert api.requests[0]["per_page"] == "5"

    def test_page_size_ignores_limit_when_filtering(self):
        # A keep filter can reject anything, so pages must be fetched in full
        _, api = _paginate([[]], limit=5, keep=lambda item: True)
        assert api.requests[0]["per_page"] == "100"

    def test_filtered_out_items_do_not_count_towards_the_limit(self):
        pages = [
            [{"n": 1, "ok": True}, {"n": 2, "ok": False}],
            [{"n": 3, "ok": True}, {"n": 4, "ok": True}],
        ]
        items, api = _paginate(
            pages, limit=2, max_page_size=2, keep=lambda item: item["ok"]
        )
        assert [i["n"] for i in items] == [1, 3]
        assert len(api.requests) == 2

    def test_start_page_is_honoured(self):
        _, api = _paginate([[], [], []], limit=10, start_page=3)
        assert api.requests[0]["page"] == "3"

    def test_extra_params_are_forwarded(self):
        _, api = _paginate([[]], limit=1, params={"state": "open"})
        assert api.requests[0]["state"] == "open"


# ── convert_comment_url_to_api_endpoint tests ──


class TestConvertCommentUrl:
    def test_api_url_passes_through(self):
        url = "https://api.github.com/repos/owner/repo/issues/comments/1"
        assert convert_comment_url_to_api_endpoint(url) == url

    def test_issue_comment_anchor(self):
        assert (
            convert_comment_url_to_api_endpoint(
                "https://github.com/owner/repo/issues/1#issuecomment-99"
            )
            == "https://api.github.com/repos/owner/repo/issues/comments/99"
        )

    def test_pr_review_comment_anchor(self):
        assert (
            convert_comment_url_to_api_endpoint(
                "https://github.com/owner/repo/pull/1#discussion_r77"
            )
            == "https://api.github.com/repos/owner/repo/pulls/comments/77"
        )

    def test_url_without_anchor_falls_back_to_plain_conversion(self):
        assert (
            convert_comment_url_to_api_endpoint(
                "https://github.com/owner/repo/issues/1"
            )
            == "https://api.github.com/repos/owner/repo/issues/1"
        )

    def test_repo_root_url(self):
        assert (
            convert_comment_url_to_api_endpoint("https://github.com/owner/repo")
            == "https://api.github.com/repos/owner/repo"
        )

    def test_url_without_a_repo_is_rejected(self):
        with pytest.raises(ValueError):
            convert_comment_url_to_api_endpoint("https://github.com/owner")


def test_subject_html_url_of_a_repo_root_is_not_rewritten():
    assert (
        _subject_html_url("https://api.github.com/repos/owner/repo")
        == "https://github.com/owner/repo"
    )


# ── Notification thread request tests ──

THREAD_URL = "https://api.github.com/notifications/threads/1337"


def _thread_call(method, *args):
    api = _FakeApi({THREAD_URL: RAW_NOTIFICATION})
    with mock.patch.object(notifications, "get_api", lambda *a, **kw: api):
        result = asyncio.run(method(TEST_CREDENTIALS, *args))
    return result, api.calls[0]


class TestNotificationThreadRequests:
    def test_get_thread_maps_the_payload(self):
        result, (method, url, _) = _thread_call(
            GithubGetNotificationThreadBlock.get_thread, "1337"
        )
        assert (method, url) == ("get", THREAD_URL)
        assert result == TEST_NOTIFICATION_ITEM

    def test_mark_thread_as_read_patches_the_thread(self):
        result, (method, url, _) = _thread_call(
            GithubMarkNotificationThreadAsReadBlock.mark_thread_as_read, "1337"
        )
        assert (result, method, url) == (True, "patch", THREAD_URL)

    def test_mark_thread_as_done_deletes_the_thread(self):
        result, (method, url, _) = _thread_call(
            GithubMarkNotificationThreadAsDoneBlock.mark_thread_as_done, "1337"
        )
        assert (result, method, url) == (True, "delete", THREAD_URL)

    def test_unsubscribe_deletes_the_subscription(self):
        result, (method, url, _) = _thread_call(
            GithubUnsubscribeNotificationThreadBlock.unsubscribe_thread, "1337"
        )
        assert (result, method, url) == (True, "delete", f"{THREAD_URL}/subscription")


# ── List-block request-building tests ──
# Same rationale as the notification request tests above: the params these
# blocks build never reach get_paginated in a block test, because the whole
# list method is mocked out.

REPO_URL = "https://github.com/owner/repo"

_LIST_METHOD = {
    GithubListCommitsBlock: "list_commits",
    GithubListBranchesBlock: "list_branches",
    GithubListIssuesBlock: "list_issues",
    GithubListPullRequestsBlock: "list_prs",
    GithubListReleasesBlock: "list_releases",
}


def _capture_paginated(module, block, input_kwargs, page: list[dict] | None = None):
    """Runs a block's list method with the network stubbed; returns (result, call)."""
    captured: dict = {}

    async def fake_get_paginated(api, url, **kwargs):
        captured.update(url=url, **kwargs)
        items = page or []
        keep = kwargs.get("keep")
        return [i for i in items if keep is None or keep(i)]

    method = getattr(block, _LIST_METHOD[block])
    input_data = block.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **input_kwargs}
    )
    with (
        mock.patch.object(module, "get_api", lambda *a, **kw: object()),
        mock.patch.object(module, "get_paginated", fake_get_paginated),
    ):
        result = asyncio.run(method(TEST_CREDENTIALS, input_data))
    return result, captured


class TestListCommitsRequest:
    def _call(self, **kwargs):
        return _capture_paginated(
            commits,
            GithubListCommitsBlock,
            {"repo_url": REPO_URL, **kwargs},
        )[1]

    def test_branch_becomes_the_sha_param(self):
        call = self._call(branch="dev")
        assert call["url"] == f"{REPO_URL}/commits"
        assert call["params"] == {"sha": "dev"}

    def test_optional_filters_become_params(self):
        call = self._call(
            path="src/main.py",
            author="alice",
            committer="bob",
            since="2026-01-01T00:00:00Z",
            until="2026-02-01T00:00:00Z",
        )
        assert call["params"] == {
            "sha": "main",
            "path": "src/main.py",
            "author": "alice",
            "committer": "bob",
            "since": "2026-01-01T00:00:00Z",
            "until": "2026-02-01T00:00:00Z",
        }

    def test_limit_is_used_by_default(self):
        call = self._call(limit=7)
        assert (call["limit"], call["start_page"]) == (7, 1)

    def test_legacy_paging_overrides_limit(self):
        call = self._call(limit=7, per_page=25, page=3)
        assert (call["limit"], call["start_page"]) == (25, 3)

    def test_legacy_page_alone_defaults_the_page_size(self):
        call = self._call(limit=7, page=2)
        assert (call["limit"], call["start_page"]) == (30, 2)

    def test_commit_items_are_mapped(self):
        raw = {
            "sha": "abc123",
            "commit": {
                "message": "Initial commit",
                "author": {"name": "octocat", "date": "2026-01-01T00:00:00Z"},
            },
        }
        result, _ = _capture_paginated(
            commits,
            GithubListCommitsBlock,
            {"repo_url": REPO_URL},
            page=[raw],
        )
        assert result[0]["url"] == f"{REPO_URL}/commit/abc123"
        assert result[0]["author"] == "octocat"

    def test_missing_commit_author_degrades(self):
        raw = {"sha": "abc123", "commit": {"message": "m", "author": None}}
        result, _ = _capture_paginated(
            commits,
            GithubListCommitsBlock,
            {"repo_url": REPO_URL},
            page=[raw],
        )
        assert (result[0]["author"], result[0]["date"]) == ("Unknown", "")


class TestListBranchesRequest:
    def _call(self, **kwargs):
        return _capture_paginated(
            repo_branches,
            GithubListBranchesBlock,
            {"repo_url": REPO_URL, **kwargs},
        )[1]

    def test_default_sends_no_protected_filter(self):
        call = self._call()
        assert call["url"] == f"{REPO_URL}/branches"
        assert call["params"] == {}

    def test_protected_filter(self):
        assert self._call(protected="protected")["params"] == {"protected": "true"}

    def test_unprotected_filter(self):
        assert self._call(protected="unprotected")["params"] == {"protected": "false"}

    def test_legacy_paging_overrides_limit(self):
        call = self._call(limit=7, per_page=10)
        assert (call["limit"], call["start_page"]) == (10, 1)

    def test_branch_items_are_mapped(self):
        result, _ = _capture_paginated(
            repo_branches,
            GithubListBranchesBlock,
            {"repo_url": REPO_URL},
            page=[{"name": "main"}],
        )
        assert result == [{"name": "main", "url": f"{REPO_URL}/tree/main"}]


class TestListIssuesRequest:
    def _call(self, **kwargs):
        return _capture_paginated(
            issues,
            GithubListIssuesBlock,
            {"repo_url": REPO_URL, **kwargs},
        )[1]

    def test_defaults(self):
        call = self._call()
        assert call["url"] == f"{REPO_URL}/issues"
        assert call["params"] == {
            "state": "open",
            "sort": "created",
            "direction": "desc",
        }

    def test_labels_are_joined(self):
        assert self._call(labels=["bug", "p0"])["params"]["labels"] == "bug,p0"

    def test_optional_filters_become_params(self):
        params = self._call(
            assignee="alice",
            creator="bob",
            mentioned="carol",
            milestone="4",
            since="2026-01-01T00:00:00Z",
        )["params"]
        assert params["assignee"] == "alice"
        assert params["creator"] == "bob"
        assert params["mentioned"] == "carol"
        assert params["milestone"] == "4"
        assert params["since"] == "2026-01-01T00:00:00Z"

    def test_pull_requests_are_filtered_out_by_default(self):
        page = [
            {"title": "issue", "html_url": "https://github.com/owner/repo/issues/1"},
            {
                "title": "pr",
                "html_url": "https://github.com/owner/repo/pull/2",
                "pull_request": {},
            },
        ]
        result, _ = _capture_paginated(
            issues,
            GithubListIssuesBlock,
            {"repo_url": REPO_URL},
            page=page,
        )
        assert [i["title"] for i in result] == ["issue"]

    def test_pull_requests_are_kept_when_requested(self):
        call = self._call(include_pull_requests=True)
        assert call["keep"] is None


class TestListCommentsRequest:
    def _call(self, issue_url: str, limit: int = 100, since: str = ""):
        captured: dict = {}

        async def fake_get_paginated(api, url, **kwargs):
            captured.update(url=url, **kwargs)
            return []

        with (
            mock.patch.object(issues, "get_api", lambda *a, **kw: object()),
            mock.patch.object(issues, "get_paginated", fake_get_paginated),
        ):
            asyncio.run(
                GithubListCommentsBlock.list_comments(
                    TEST_CREDENTIALS, issue_url, limit, since
                )
            )
        return captured

    def test_issue_url_becomes_the_comments_endpoint(self):
        call = self._call("https://github.com/owner/repo/issues/1")
        assert (
            call["url"] == "https://api.github.com/repos/owner/repo/issues/1/comments"
        )
        assert call["params"] is None

    def test_pr_url_uses_the_same_issues_endpoint(self):
        call = self._call("https://github.com/owner/repo/pull/42")
        assert (
            call["url"] == "https://api.github.com/repos/owner/repo/issues/42/comments"
        )

    def test_since_is_forwarded(self):
        call = self._call(
            "https://github.com/owner/repo/issues/1", since="2026-01-01T00:00:00Z"
        )
        assert call["params"] == {"since": "2026-01-01T00:00:00Z"}


class TestListPullRequestsRequest:
    def _call(self, **kwargs):
        return _capture_paginated(
            pull_requests,
            GithubListPullRequestsBlock,
            {"repo_url": REPO_URL, **kwargs},
        )[1]

    def test_defaults(self):
        call = self._call()
        assert call["url"] == f"{REPO_URL}/pulls"
        assert call["params"] == {
            "state": "open",
            "sort": "created",
            "direction": "desc",
        }

    def test_base_and_head_become_params(self):
        params = self._call(base="dev", head="octocat:feature")["params"]
        assert (params["base"], params["head"]) == ("dev", "octocat:feature")

    def test_limit_is_forwarded(self):
        assert self._call(limit=9)["limit"] == 9


# ── list_reviews filtering tests ──


def _list_reviews(reviews_page: list[dict], **kwargs):
    captured: dict = {}

    async def fake_get_paginated(api, url, **kw):
        captured.update(url=url, **kw)
        keep = kw.get("keep")
        return [r for r in reviews_page if keep is None or keep(r)]

    input_data = GithubListPRReviewsBlock.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "repo": "owner/repo",
            "pr_number": 1,
            **kwargs,
        }
    )
    with (
        mock.patch.object(reviews, "get_api", lambda *a, **kw: object()),
        mock.patch.object(reviews, "get_paginated", fake_get_paginated),
    ):
        result = asyncio.run(
            GithubListPRReviewsBlock.list_reviews(TEST_CREDENTIALS, input_data)
        )
    return result, captured


def _pr_review(login: str, state: str, id: int = 1) -> dict:
    return {
        "id": id,
        "user": {"login": login},
        "state": state,
        "body": "",
        "html_url": f"https://github.com/owner/repo/pull/1#pullrequestreview-{id}",
    }


class TestListReviews:
    def test_endpoint_is_built_from_repo_and_pr_number(self):
        _, call = _list_reviews([])
        assert call["url"] == "https://api.github.com/repos/owner/repo/pulls/1/reviews"

    def test_unfiltered_fetch_uses_the_limit(self):
        _, call = _list_reviews([], limit=12)
        assert call["limit"] == 12
        assert call["keep"] is None

    def test_post_fetch_filtering_raises_the_fetch_limit(self):
        # State/latest_only filtering happens after fetching, so a small fetch
        # limit would silently drop matches that sit further down the list.
        _, call = _list_reviews([], limit=12, latest_only=True)
        assert call["limit"] == 1000

    def test_reviewer_filter_is_applied_while_fetching(self):
        result, _ = _list_reviews(
            [_pr_review("alice", "APPROVED", 1), _pr_review("bob", "APPROVED", 2)],
            reviewer="alice",
        )
        assert [r["user"] for r in result] == ["alice"]

    def test_latest_only_keeps_the_last_review_per_user(self):
        result, _ = _list_reviews(
            [
                _pr_review("alice", "CHANGES_REQUESTED", 1),
                _pr_review("alice", "APPROVED", 2),
            ],
            latest_only=True,
        )
        assert [(r["id"], r["state"]) for r in result] == [(2, "APPROVED")]

    def test_latest_only_ignores_pending_drafts(self):
        result, _ = _list_reviews(
            [_pr_review("alice", "APPROVED", 1), _pr_review("alice", "PENDING", 2)],
            latest_only=True,
        )
        assert [r["id"] for r in result] == [1]

    def test_state_filter_is_case_insensitive_against_the_api(self):
        result, _ = _list_reviews(
            [
                _pr_review("alice", "APPROVED", 1),
                _pr_review("bob", "CHANGES_REQUESTED", 2),
            ],
            state="changes_requested",
        )
        assert [r["id"] for r in result] == [2]

    def test_limit_is_applied_after_filtering(self):
        result, _ = _list_reviews(
            [_pr_review("a", "APPROVED", 1), _pr_review("b", "APPROVED", 2)],
            state="approved",
            limit=1,
        )
        assert len(result) == 1


# ── list_releases filtering tests ──


def _release(
    tag: str,
    *,
    prerelease: bool = False,
    draft: bool = False,
    published_at: str | None = "2026-01-01T00:00:00Z",
    created_at: str = "2026-01-01T00:00:00Z",
) -> dict:
    return {
        "name": None,
        "tag_name": tag,
        "html_url": f"https://github.com/owner/repo/releases/tag/{tag}",
        "published_at": published_at,
        "created_at": created_at,
        "prerelease": prerelease,
        "draft": draft,
    }


def _list_releases(page: list[dict], **kwargs):
    return _capture_paginated(
        repo,
        GithubListReleasesBlock,
        {"repo_url": REPO_URL, **kwargs},
        page=page,
    )


class TestListReleases:
    def test_no_filter_means_no_keep_predicate(self):
        _, call = _list_releases(
            [],
        )
        assert call["url"] == f"{REPO_URL}/releases"
        assert call["keep"] is None

    def test_stable_excludes_prereleases_and_drafts(self):
        result, _ = _list_releases(
            [
                _release("v1"),
                _release("v2-rc", prerelease=True),
                _release("v3", draft=True),
            ],
            release_type="stable",
        )
        assert [r["tag_name"] for r in result] == ["v1"]

    def test_prerelease_filter(self):
        result, _ = _list_releases(
            [_release("v1"), _release("v2-rc", prerelease=True)],
            release_type="prerelease",
        )
        assert [r["tag_name"] for r in result] == ["v2-rc"]

    def test_draft_filter_uses_created_at_for_the_date(self):
        result, _ = _list_releases(
            [
                _release(
                    "v9",
                    draft=True,
                    published_at=None,
                    created_at="2026-06-01T00:00:00Z",
                ),
                _release("v1", published_at="2026-06-01T00:00:00Z"),
            ],
            release_type="draft",
            published_after="2026-05-01T00:00:00Z",
        )
        assert [r["tag_name"] for r in result] == ["v9"]

    def test_published_after_and_before_bracket_the_range(self):
        result, _ = _list_releases(
            [
                _release("old", published_at="2025-01-01T00:00:00Z"),
                _release("hit", published_at="2026-06-01T00:00:00Z"),
                _release("new", published_at="2027-01-01T00:00:00Z"),
            ],
            published_after="2026-01-01T00:00:00Z",
            published_before="2026-12-31T00:00:00Z",
        )
        assert [r["tag_name"] for r in result] == ["hit"]

    def test_undated_release_is_dropped_when_a_date_filter_is_set(self):
        result, _ = _list_releases(
            [_release("v1", published_at=None, created_at="")],
            published_after="2026-01-01T00:00:00Z",
        )
        assert result == []

    def test_name_falls_back_to_the_tag(self):
        result, _ = _list_releases([_release("v1")])
        assert result[0]["name"] == "v1"


class TestParseTimestamp:
    def test_empty_string_is_none(self):
        assert repo._parse_timestamp("") is None

    def test_zulu_suffix_is_understood(self):
        assert repo._parse_timestamp("2026-01-01T00:00:00Z") == datetime(
            2026, 1, 1, tzinfo=timezone.utc
        )

    def test_naive_timestamp_is_assumed_utc(self):
        assert repo._parse_timestamp("2026-01-01T00:00:00") == datetime(
            2026, 1, 1, tzinfo=timezone.utc
        )


# ── list_tags / list_stargazers tests ──


class TestSimpleRepoLists:
    def test_tags_are_mapped_to_tree_urls(self):
        captured: dict = {}

        async def fake_get_paginated(api, url, **kwargs):
            captured.update(url=url, **kwargs)
            return [{"name": "v1.0.0"}]

        with (
            mock.patch.object(repo, "get_api", lambda *a, **kw: object()),
            mock.patch.object(repo, "get_paginated", fake_get_paginated),
        ):
            result = asyncio.run(
                GithubListTagsBlock.list_tags(TEST_CREDENTIALS, REPO_URL, 5)
            )
        assert captured["url"] == f"{REPO_URL}/tags"
        assert result == [{"name": "v1.0.0", "url": f"{REPO_URL}/tree/v1.0.0"}]

    def test_stargazers_are_mapped(self):
        captured: dict = {}

        async def fake_get_paginated(api, url, **kwargs):
            captured.update(url=url, **kwargs)
            return [{"login": "octocat", "html_url": "https://github.com/octocat"}]

        with (
            mock.patch.object(repo, "get_api", lambda *a, **kw: object()),
            mock.patch.object(repo, "get_paginated", fake_get_paginated),
        ):
            result = asyncio.run(
                GithubListStargazersBlock.list_stargazers(TEST_CREDENTIALS, REPO_URL, 5)
            )
        assert captured["url"] == f"{REPO_URL}/stargazers"
        assert result == [{"username": "octocat", "url": "https://github.com/octocat"}]


# ── list_discussions (GraphQL) tests ──

GRAPHQL_URL = "https://api.github.com/graphql"


class _GraphqlApi:
    """Replays a queue of GraphQL responses and snapshots each request."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    async def post(self, url, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["json"]))
        payload = self.responses.pop(0)
        return mock.Mock(json=lambda: payload)


def _discussions_page(nodes: list[dict], has_next: bool = False, cursor=None) -> dict:
    return {
        "data": {
            "repository": {
                "discussions": {
                    "nodes": nodes,
                    "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                }
            }
        }
    }


def _list_discussions(responses: list[dict], **kwargs):
    api = _GraphqlApi(responses)
    input_data = GithubListDiscussionsBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "repo_url": REPO_URL, **kwargs}
    )
    with mock.patch.object(repo, "get_api", lambda *a, **kw: api):
        result = asyncio.run(
            GithubListDiscussionsBlock.list_discussions(TEST_CREDENTIALS, input_data)
        )
    return result, api.requests


def _discussion(title: str) -> dict:
    return {"title": title, "url": f"https://github.com/owner/repo/discussions/{title}"}


class TestListDiscussions:
    def test_defaults_send_no_filters(self):
        _, requests = _list_discussions([_discussions_page([])])
        variables = requests[0]["variables"]
        assert variables["owner"] == "owner"
        assert variables["repo"] == "repo"
        assert variables["categoryId"] is None
        assert variables["answered"] is None
        assert variables["states"] is None
        assert variables["orderBy"] == {"field": "UPDATED_AT", "direction": "DESC"}

    def test_page_size_is_capped_at_100(self):
        _, requests = _list_discussions([_discussions_page([])], num_discussions=250)
        assert requests[0]["variables"]["num"] == 100

    def test_filters_are_translated_to_graphql_arguments(self):
        _, requests = _list_discussions(
            [_discussions_page([])],
            answered="unanswered",
            state="closed",
            order_by="created",
            direction="asc",
        )
        variables = requests[0]["variables"]
        assert variables["answered"] is False
        assert variables["states"] == ["CLOSED"]
        assert variables["orderBy"] == {"field": "CREATED_AT", "direction": "ASC"}

    def test_pagination_follows_the_cursor_and_truncates(self):
        result, requests = _list_discussions(
            [
                _discussions_page(
                    [_discussion("a"), _discussion("b")], has_next=True, cursor="CUR"
                ),
                _discussions_page([_discussion("c"), _discussion("d")]),
            ],
            num_discussions=3,
        )
        assert [d["title"] for d in result] == ["a", "b", "c"]
        assert requests[0]["variables"]["after"] is None
        assert requests[1]["variables"]["after"] == "CUR"

    def test_category_name_is_resolved_to_an_id(self):
        categories = {
            "data": {
                "repository": {
                    "discussionCategories": {"nodes": [{"id": "DIC_1", "name": "Q&A"}]}
                }
            }
        }
        _, requests = _list_discussions(
            [categories, _discussions_page([])], category="q&a"
        )
        assert requests[1]["variables"]["categoryId"] == "DIC_1"

    def test_unknown_category_raises(self):
        categories = {
            "data": {
                "repository": {
                    "discussionCategories": {"nodes": [{"id": "DIC_1", "name": "Q&A"}]}
                }
            }
        }
        with pytest.raises(ValueError, match="does not exist"):
            _list_discussions([categories], category="Ideas")


# ── get_user tests ──


class TestGetUser:
    def _call(self, username: str) -> str:
        api = _FakeApi()
        with mock.patch.object(users, "get_api", lambda *a, **kw: api):
            asyncio.run(GithubGetUserInfoBlock.get_user(TEST_CREDENTIALS, username))
        return api.calls[0][1]

    def test_named_user(self):
        assert self._call("octocat") == "https://api.github.com/users/octocat"

    def test_empty_username_means_the_authenticated_user(self):
        assert self._call("") == "https://api.github.com/user"
