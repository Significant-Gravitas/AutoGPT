"""Reconcile the scopes an OAuth provider *granted* with the ones we *asked for*.

Two provider quirks make this less trivial than a set difference:

* **Separators.** RFC 6749 says the ``scope`` response field is
  space-delimited, but real providers ship comma-delimited (GitHub, in some
  flows), space-delimited (GitHub in others, Linear, Discord, Stripe Link) or
  already-split lists (Google via oauthlib).  A single scope string that was
  never split looks like one exotic scope and makes every real scope read as
  missing.  :func:`normalize_scopes` flattens all of those shapes.

* **Silence vs. refusal.** A provider that omits ``scope`` from its token
  response has told us *nothing* — alarming on that would false-alarm on
  every Notion connect (its handler hardcodes ``scopes=[]``).  A provider
  that is contractually required to report the grant and returns an empty
  one has told us *nothing was granted* — which is exactly the failure we
  need to shout about.  The two are indistinguishable from the credential
  object alone, so the distinction is declared per handler via
  ``BaseOAuthHandler.REPORTS_GRANTED_SCOPES``.
"""

import re
from enum import Enum
from typing import Iterable

from pydantic import BaseModel

# Split on any run of commas and/or whitespace, so one helper covers every
# separator convention we have seen in the wild.
_SCOPE_SPLIT_PATTERN = re.compile(r"[,\s]+")


class ScopeCoverage(str, Enum):
    """How well the granted grant covers what was requested."""

    UNKNOWN = "unknown"
    """The provider does not report its grant; we cannot tell. Not a failure."""

    COVERED = "covered"
    """Every requested scope is present in the grant."""

    PARTIAL = "partial"
    """Some scopes were granted, but not all of the requested ones."""

    NONE_GRANTED = "none_granted"
    """The provider reported a grant and it was empty."""


class ScopeCoverageResult(BaseModel):
    coverage: ScopeCoverage
    requested: list[str]
    granted: list[str]
    missing: list[str]

    @property
    def is_shortfall(self) -> bool:
        """True when the user must re-authorize for the connection to work."""
        return self.coverage in (ScopeCoverage.PARTIAL, ScopeCoverage.NONE_GRANTED)


def normalize_scopes(raw: Iterable[str]) -> list[str]:
    """Flatten a provider's scope field into individual, deduplicated scopes.

    Handles ``["repo,workflow"]``, ``["repo workflow"]``, ``["repo",
    "workflow"]`` and mixtures of them identically.  Empty fragments are
    dropped, which is what turns GitHub's ``"".split(",") == [""]`` into the
    empty list it always meant.  Order is preserved so the stored scope list
    still reads the way the provider sent it.
    """
    seen: set[str] = set()
    result: list[str] = []
    for entry in raw:
        for scope in _SCOPE_SPLIT_PATTERN.split(entry or ""):
            if scope and scope not in seen:
                seen.add(scope)
                result.append(scope)
    return result


def evaluate_scope_coverage(
    requested: Iterable[str],
    granted: Iterable[str],
    *,
    provider_reports_scopes: bool,
) -> ScopeCoverageResult:
    """Diff *requested* against *granted*, honouring the silence/refusal split.

    ``provider_reports_scopes`` mirrors the handler's
    ``REPORTS_GRANTED_SCOPES``: when False the result is always
    :attr:`ScopeCoverage.UNKNOWN` and ``missing`` is empty, because an absent
    grant report is not evidence of a narrow grant.
    """
    requested_scopes = normalize_scopes(requested)
    granted_scopes = normalize_scopes(granted)

    if not provider_reports_scopes:
        return ScopeCoverageResult(
            coverage=ScopeCoverage.UNKNOWN,
            requested=requested_scopes,
            granted=granted_scopes,
            missing=[],
        )

    granted_set = set(granted_scopes)
    missing = [scope for scope in requested_scopes if scope not in granted_set]

    if not missing:
        coverage = ScopeCoverage.COVERED
    elif granted_scopes:
        coverage = ScopeCoverage.PARTIAL
    else:
        coverage = ScopeCoverage.NONE_GRANTED

    return ScopeCoverageResult(
        coverage=coverage,
        requested=requested_scopes,
        granted=granted_scopes,
        missing=missing,
    )
