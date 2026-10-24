"""Which OIDC tokens a workload identity accepts: issuer, audience and claims."""

import fnmatch
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

# Platforms whose single issuer signs tokens for every one of their customers.
# Trusting the issuer alone would let any of them act as the owner, so an
# identity for one of these must pin a claim that names the owner's own
# repository, project or account.
SHARED_ISSUER_BINDING_CLAIMS: dict[str, frozenset[str]] = {
    "https://token.actions.githubusercontent.com": frozenset(
        {
            "sub",
            "repository",
            "repository_id",
            "repository_owner",
            "repository_owner_id",
            "job_workflow_ref",
        }
    ),
    "https://gitlab.com": frozenset(
        {"sub", "project_path", "project_id", "namespace_path", "namespace_id"}
    ),
    "https://oidc.vercel.com": frozenset(
        {"sub", "owner", "owner_id", "project", "project_id"}
    ),
    "https://accounts.google.com": frozenset({"sub", "email", "azp"}),
}

# Checked by the token verifier itself; a condition on them adds nothing.
_VERIFIED_CLAIMS = frozenset({"iss", "aud", "exp", "iat", "nbf"})
_WILDCARD_STARTS = ("*", "?", "[")


class ClaimCondition(BaseModel):
    claim: str = Field(min_length=1, max_length=128)
    pattern: str = Field(
        min_length=1,
        max_length=512,
        description="Exact value, or a glob such as `repo:acme/api:ref:refs/heads/*`",
    )


class TrustPolicyError(ValueError):
    """A workload identity's trust settings would accept the wrong tokens."""


def check_trust_policy(
    issuer: str, audience: str, conditions: list[ClaimCondition]
) -> None:
    """Raise TrustPolicyError unless the settings pin tokens to one workload."""
    _check_issuer(issuer)
    if not audience.strip():
        raise TrustPolicyError("Audience must not be empty")
    if not conditions:
        raise TrustPolicyError("Add at least one claim condition")
    if any(c.claim in _VERIFIED_CLAIMS for c in conditions):
        raise TrustPolicyError(
            f"Conditions cannot use {', '.join(sorted(_VERIFIED_CLAIMS))}; "
            "issuer, audience and expiry are always checked"
        )
    anchored = [c for c in conditions if not c.pattern.startswith(_WILDCARD_STARTS)]
    if not anchored:
        raise TrustPolicyError(
            "At least one condition must start with a literal value, not a wildcard"
        )
    binding_claims = SHARED_ISSUER_BINDING_CLAIMS.get(issuer)
    if binding_claims and not any(c.claim in binding_claims for c in anchored):
        raise TrustPolicyError(
            f"{issuer} signs tokens for every account on that platform. Add a "
            f"condition on one of: {', '.join(sorted(binding_claims))}"
        )


def claims_match(conditions: list[ClaimCondition], claims: dict[str, Any]) -> bool:
    """Whether every condition holds. A list claim holds if any item matches."""
    return all(_condition_holds(c, claims) for c in conditions)


def _check_issuer(issuer: str) -> None:
    parts = urlsplit(issuer)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or len(issuer) > 512
    ):
        raise TrustPolicyError(
            "Issuer must be an https URL without credentials, query or fragment"
        )


def _condition_holds(condition: ClaimCondition, claims: dict[str, Any]) -> bool:
    value = claims.get(condition.claim)
    values = value if isinstance(value, list) else [value]
    texts = [_as_text(v) for v in values]
    return any(
        text is not None and fnmatch.fnmatchcase(text, condition.pattern)
        for text in texts
    )


def _as_text(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    return None
