"""Regression tests for GitHub's granted-scope parsing.

The original expression was::

    scopes=(
        token_data.get("scope", "").split(",")
        or (current_credentials.scopes if current_credentials else [])
    )

``"".split(",")`` is ``[""]``, which is truthy, so the ``or`` fallback was
dead code. Two real failures came out of that:

* a **refresh** (whose ``scope`` is documented as empty) silently replaced the
  credential's real scopes with ``[""]``;
* a **zero-scope authorization** — GitHub reusing a prior grant and handing
  back nothing — stored a credential claiming one nameless scope instead of
  an honest empty grant, so no coverage check could ever notice.
"""

from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.github import GitHubOAuthHandler


def _existing(scopes: list[str]) -> OAuth2Credentials:
    return OAuth2Credentials(
        id="github-cred-1",
        provider="github",
        title="My GitHub",
        access_token=SecretStr("ghp_old"),
        refresh_token=SecretStr("ghr_old"),
        scopes=scopes,
        username="alice",
    )


class TestGitHubResolveScopes:
    def test_empty_scope_on_first_exchange_is_no_scopes_not_one_empty_scope(self):
        """The incident. A zero-scope token must be recorded as zero scopes so
        the post-connect check can flag it."""
        assert GitHubOAuthHandler._resolve_scopes({"scope": ""}, None) == []

    def test_missing_scope_key_on_first_exchange_is_no_scopes(self):
        assert GitHubOAuthHandler._resolve_scopes({}, None) == []

    def test_empty_scope_on_refresh_keeps_the_existing_scopes(self):
        """GitHub documents refresh responses as carrying an empty `scope`;
        the credential's real grant must survive it."""
        current = _existing(["repo", "workflow"])
        assert GitHubOAuthHandler._resolve_scopes({"scope": ""}, current) == [
            "repo",
            "workflow",
        ]

    def test_missing_scope_key_on_refresh_keeps_the_existing_scopes(self):
        current = _existing(["repo"])
        assert GitHubOAuthHandler._resolve_scopes({}, current) == ["repo"]

    def test_comma_separated_scopes_are_split(self):
        assert GitHubOAuthHandler._resolve_scopes(
            {"scope": "repo,read:org,workflow"}, None
        ) == ["repo", "read:org", "workflow"]

    def test_space_separated_scopes_are_split(self):
        """Some GitHub flows space-delimit; the old code only split on comma,
        so the whole string read as one exotic scope."""
        assert GitHubOAuthHandler._resolve_scopes({"scope": "repo read:org"}, None) == [
            "repo",
            "read:org",
        ]

    def test_granted_scopes_win_over_existing_on_refresh(self):
        current = _existing(["repo"])
        assert GitHubOAuthHandler._resolve_scopes(
            {"scope": "repo,workflow"}, current
        ) == ["repo", "workflow"]

    def test_null_scope_value_is_tolerated(self):
        assert GitHubOAuthHandler._resolve_scopes({"scope": None}, None) == []

    def test_handler_declares_that_it_reports_granted_scopes(self):
        """Licenses the callback to read an empty grant as "nothing granted"
        rather than "the provider said nothing"."""
        assert GitHubOAuthHandler.REPORTS_GRANTED_SCOPES is True
