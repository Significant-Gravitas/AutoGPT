"""Tests for granted-vs-requested OAuth scope reconciliation."""

import pytest

from backend.integrations.oauth.scopes import (
    ScopeCoverage,
    evaluate_scope_coverage,
    normalize_scopes,
)


class TestNormalizeScopes:
    def test_comma_separated_single_string_is_split(self):
        """GitHub's authorization-code exchange comma-delimits `scope`."""
        assert normalize_scopes(["repo,read:org,workflow"]) == [
            "repo",
            "read:org",
            "workflow",
        ]

    def test_space_separated_single_string_is_split(self):
        """RFC 6749 / Linear / Discord space-delimit `scope`."""
        assert normalize_scopes(["repo read:org"]) == ["repo", "read:org"]

    def test_mixed_comma_and_space_separators(self):
        assert normalize_scopes(["repo, read:org  workflow"]) == [
            "repo",
            "read:org",
            "workflow",
        ]

    def test_already_split_list_passes_through(self):
        """Google (via oauthlib) hands us a real list."""
        assert normalize_scopes(["openid", "email"]) == ["openid", "email"]

    def test_empty_string_yields_no_scopes(self):
        """`"".split(",") == [""]` is the bug this exists to kill: an empty
        scope field means zero scopes, not one scope named ``""``."""
        assert normalize_scopes([""]) == []

    def test_empty_input_yields_empty_list(self):
        assert normalize_scopes([]) == []

    def test_duplicates_are_dropped_and_order_preserved(self):
        assert normalize_scopes(["repo", "repo,workflow", "repo"]) == [
            "repo",
            "workflow",
        ]

    def test_stray_separators_do_not_produce_empty_scopes(self):
        assert normalize_scopes([" , repo , "]) == ["repo"]


class TestEvaluateScopeCoverage:
    def test_exact_match_is_covered(self):
        result = evaluate_scope_coverage(
            ["repo"], ["repo"], provider_reports_scopes=True
        )
        assert result.coverage is ScopeCoverage.COVERED
        assert result.missing == []
        assert result.is_shortfall is False

    def test_superset_grant_is_covered(self):
        """A provider that grants more than we asked for is not a problem."""
        result = evaluate_scope_coverage(
            ["repo"], ["repo", "workflow"], provider_reports_scopes=True
        )
        assert result.coverage is ScopeCoverage.COVERED
        assert result.missing == []

    def test_partial_grant_lists_only_the_missing_scopes(self):
        result = evaluate_scope_coverage(
            ["repo", "workflow", "read:org"],
            ["repo"],
            provider_reports_scopes=True,
        )
        assert result.coverage is ScopeCoverage.PARTIAL
        assert result.missing == ["workflow", "read:org"]
        assert result.is_shortfall is True

    def test_empty_grant_from_a_reporting_provider_is_none_granted(self):
        """The reported incident: OAuth succeeded, token carried nothing."""
        result = evaluate_scope_coverage(["repo"], [], provider_reports_scopes=True)
        assert result.coverage is ScopeCoverage.NONE_GRANTED
        assert result.missing == ["repo"]
        assert result.is_shortfall is True

    def test_github_empty_scope_string_is_none_granted_not_one_scope(self):
        result = evaluate_scope_coverage(["repo"], [""], provider_reports_scopes=True)
        assert result.coverage is ScopeCoverage.NONE_GRANTED
        assert result.granted == []
        assert result.missing == ["repo"]

    def test_non_reporting_provider_is_unknown_not_a_failure(self):
        """Notion hardcodes ``scopes=[]``; alarming on it would false-alarm
        on every successful Notion connect."""
        result = evaluate_scope_coverage(["repo"], [], provider_reports_scopes=False)
        assert result.coverage is ScopeCoverage.UNKNOWN
        assert result.missing == []
        assert result.is_shortfall is False

    def test_non_reporting_provider_stays_unknown_even_with_a_real_shortfall(self):
        result = evaluate_scope_coverage(
            ["repo", "workflow"], ["repo"], provider_reports_scopes=False
        )
        assert result.coverage is ScopeCoverage.UNKNOWN
        assert result.missing == []

    def test_no_requested_scopes_is_covered(self):
        result = evaluate_scope_coverage([], [], provider_reports_scopes=True)
        assert result.coverage is ScopeCoverage.COVERED

    @pytest.mark.parametrize(
        "granted",
        [["repo,workflow"], ["repo workflow"], ["repo", "workflow"]],
    )
    def test_separator_style_does_not_change_the_verdict(self, granted):
        result = evaluate_scope_coverage(
            ["repo", "workflow"], granted, provider_reports_scopes=True
        )
        assert result.coverage is ScopeCoverage.COVERED

    def test_requested_side_is_normalized_too(self):
        """`/login?scopes=` arrives comma-joined and round-trips through the
        state token, so the requested side needs the same flattening."""
        result = evaluate_scope_coverage(
            ["repo,workflow"], ["repo", "workflow"], provider_reports_scopes=True
        )
        assert result.coverage is ScopeCoverage.COVERED
        assert result.requested == ["repo", "workflow"]
