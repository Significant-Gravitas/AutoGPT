"""Trust settings must pin tokens to one workload; claims match them exactly."""

import pytest

from backend.data.auth.workload_trust import (
    ClaimCondition,
    TrustPolicyError,
    check_trust_policy,
    claims_match,
)

GITHUB = "https://token.actions.githubusercontent.com"
AUDIENCE = "https://backend.agpt.co"


def _cond(claim: str, pattern: str) -> ClaimCondition:
    return ClaimCondition(claim=claim, pattern=pattern)


@pytest.mark.parametrize(
    "issuer, conditions",
    [
        (GITHUB, [_cond("repository", "acme/api")]),
        (GITHUB, [_cond("sub", "repo:acme/api:ref:refs/heads/main")]),
        (GITHUB, [_cond("repository_owner", "acme"), _cond("ref", "*")]),
        ("https://oidc.vercel.com", [_cond("project_id", "prj_123")]),
        ("https://oidc.vercel.com/acme", [_cond("environment", "production")]),
        ("https://kubernetes.example.com", [_cond("sub", "system:serviceaccount:*")]),
    ],
)
def test_a_policy_pinned_to_one_workload_is_accepted(issuer, conditions):
    check_trust_policy(issuer, AUDIENCE, conditions)


@pytest.mark.parametrize(
    "issuer, conditions, message",
    [
        (GITHUB, [_cond("ref", "refs/heads/main")], "every account"),
        (GITHUB, [_cond("repository", "*")], "literal value"),
        (GITHUB, [_cond("repository", "*acme*")], "literal value"),
        ("https://gitlab.com", [_cond("ref", "main")], "every account"),
        ("https://accounts.google.com", [_cond("aud", AUDIENCE)], "cannot use"),
        ("https://custom.example.com", [], "at least one"),
        ("http://custom.example.com", [_cond("sub", "x")], "https URL"),
        ("https://custom.example.com?x=1", [_cond("sub", "x")], "https URL"),
        ("https://user:pw@custom.example.com", [_cond("sub", "x")], "https URL"),
        ("not a url", [_cond("sub", "x")], "https URL"),
    ],
)
def test_a_policy_that_accepts_strangers_tokens_is_refused(issuer, conditions, message):
    with pytest.raises(TrustPolicyError, match=message):
        check_trust_policy(issuer, AUDIENCE, conditions)


def test_an_empty_audience_is_refused():
    with pytest.raises(TrustPolicyError, match="Audience"):
        check_trust_policy(GITHUB, "  ", [_cond("repository", "acme/api")])


@pytest.mark.parametrize(
    "conditions, claims, expected",
    [
        ([_cond("repository", "acme/api")], {"repository": "acme/api"}, True),
        ([_cond("repository", "acme/api")], {"repository": "acme/api-fork"}, False),
        ([_cond("repository", "acme/api")], {"repository": "Acme/api"}, False),
        ([_cond("repository", "acme/api")], {}, False),
        ([_cond("ref", "refs/heads/*")], {"ref": "refs/heads/main"}, True),
        ([_cond("ref", "refs/heads/*")], {"ref": "refs/tags/v1"}, False),
        ([_cond("groups", "deployers")], {"groups": ["dev", "deployers"]}, True),
        ([_cond("groups", "deployers")], {"groups": ["dev"]}, False),
        ([_cond("repository_id", "42")], {"repository_id": 42}, True),
        ([_cond("protected", "true")], {"protected": True}, True),
        ([_cond("protected", "true")], {"protected": False}, False),
        ([_cond("nested", "x")], {"nested": {"x": 1}}, False),
        (
            [_cond("repository", "acme/api"), _cond("ref", "refs/heads/main")],
            {"repository": "acme/api", "ref": "refs/heads/dev"},
            False,
        ),
    ],
)
def test_claims_match_only_when_every_condition_holds(conditions, claims, expected):
    assert claims_match(conditions, claims) is expected
