"""Shared block test fixtures.

Kept in its own module so block files never import test data from each other.
"""

from ._types import (
    AllQuietEntity,
    AllQuietUser,
    Incident,
    IncidentSeverity,
    IncidentStatus,
    OnCallAvailability,
    OnCallShift,
    Team,
)

TEST_INCIDENT = Incident(
    id="a1b2c3d4-0000-4000-8000-000000000001",
    title="Checkout latency above SLO",
    status=IncidentStatus.OPEN,
    severity=IncidentSeverity.CRITICAL,
    createdAt="2026-08-16T23:42:17.274Z",
    lastUpdatedAt="2026-08-16T23:42:17.274Z",
    allowedIntents=["Investigated", "Resolved", "Escalated"],
)

TEST_MARKDOWN = (
    "# Checkout latency above SLO\n\n"
    "- **Status**: Open\n"
    "- **Severity**: Critical\n"
)

TEST_TEAM = Team(
    id="7da9d74c-0000-4000-8000-000000000003",
    displayName="Platform",
    timeZoneId="UTC",
    labels=["Engineering"],
)

TEST_USER = AllQuietUser(
    id="b7c8d9e0-0000-4000-8000-000000000002",
    displayName="Ada Lovelace",
    email="ada@example.com",
)

TEST_SHIFT = OnCallShift(
    user=TEST_USER,
    team=AllQuietEntity(id=TEST_TEAM.id, displayName=TEST_TEAM.display_name),
    availabilities=[OnCallAvailability(tier=1, isOnline=True, fillUp=False)],
)
