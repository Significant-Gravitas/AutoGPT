import pytest

from backend.blocks.github._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.github.commits import (
    FileOperation,
    GithubMultiFileCommitBlock,
)
from backend.blocks.github.pull_requests import (
    GithubMergePullRequestBlock,
    prepare_pr_api_url,
)
from backend.util.test import execute_block_test


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


def _raise(exc: Exception):
    """Helper that returns a callable which raises the given exception."""

    def _raiser(*args, **kwargs):
        raise exc

    return _raiser


@pytest.mark.asyncio
async def test_merge_pr_error_path():
    block = GithubMergePullRequestBlock()
    block.test_input = {
        "pr_url": "https://github.com/owner/repo/pull/1",
        "merge_method": "squash",
        "commit_title": "",
        "commit_message": "",
        "credentials": TEST_CREDENTIALS_INPUT,
    }
    block.test_credentials = TEST_CREDENTIALS
    block.test_output = [("error", "PR not mergeable")]
    block.test_mock = {"merge_pr": _raise(RuntimeError("PR not mergeable"))}
    await execute_block_test(block)


@pytest.mark.asyncio
async def test_multi_file_commit_error_path():
    block = GithubMultiFileCommitBlock()
    block.test_input = {
        "repo_url": "https://github.com/owner/repo",
        "branch": "feature",
        "commit_message": "test",
        "files": [{"path": "a.py", "content": "x", "operation": "upsert"}],
        "credentials": TEST_CREDENTIALS_INPUT,
    }
    block.test_credentials = TEST_CREDENTIALS
    block.test_output = [("error", "ref update failed")]
    block.test_mock = {
        "multi_file_commit": _raise(RuntimeError("ref update failed"))
    }
    await execute_block_test(block)


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
