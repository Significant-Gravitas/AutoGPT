"""Tests for tiered-memory plumbing: budget merge, read-target resolution,
membership + write governance."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from . import tiers
from .tiers import (
    MemoryTier,
    TierError,
    hold_buffer_enabled,
    is_org_admin,
    merge_tiered,
    resolve_search_targets,
    resolve_store_team,
    resolve_warm_targets,
    team_label,
)

# ---------------------------------------------------------------------------
# Budget merge: personal keeps >= half; shared tiers share the rest
# ---------------------------------------------------------------------------


class TestMergeTiered:
    def test_personal_keeps_at_least_half_when_all_tiers_full(self) -> None:
        personal = [f"p{i}" for i in range(20)]
        shared = [("org memory", [f"o{i}" for i in range(20)])]
        merged = merge_tiered(personal, shared, total=20)

        assert len(merged) == 20
        personal_count = sum(1 for _, label in merged if label is None)
        # Floor is total // 2 = 10; personal must never drop below it.
        assert personal_count >= 10
        assert personal_count == 10  # shared is plentiful, so exactly the floor

    def test_personal_absorbs_unused_shared_budget(self) -> None:
        # Only 3 shared facts available, so personal should take the other 17.
        personal = [f"p{i}" for i in range(20)]
        shared = [("org memory", ["o0", "o1", "o2"])]
        merged = merge_tiered(personal, shared, total=20)

        personal_count = sum(1 for _, label in merged if label is None)
        assert personal_count == 17
        assert sum(1 for _, label in merged if label == "org memory") == 3

    def test_personal_below_half_takes_all_and_shared_fills(self) -> None:
        personal = ["p0", "p1"]  # fewer than half of 10
        shared = [("org memory", [f"o{i}" for i in range(20)])]
        merged = merge_tiered(personal, shared, total=10)

        assert sum(1 for _, label in merged if label is None) == 2
        assert sum(1 for _, label in merged if label == "org memory") == 8

    def test_shared_labels_attached_and_round_robin_interleaved(self) -> None:
        personal: list[str] = []
        shared = [
            ("org memory", ["o0", "o1", "o2"]),
            ("team memory (X)", ["t0", "t1", "t2"]),
        ]
        merged = merge_tiered(personal, shared, total=6)

        labels = [label for _, label in merged]
        # Round-robin: org, team, org, team, ... so both tiers' top hits lead.
        assert labels[0] == "org memory"
        assert labels[1] == "team memory (X)"
        assert set(labels) == {"org memory", "team memory (X)"}

    def test_zero_budget_returns_empty(self) -> None:
        assert merge_tiered(["p"], [("org", ["o"])], total=0) == []


def test_team_label_formats_with_and_without_name() -> None:
    assert team_label("Platform") == "team memory (Platform)"
    assert team_label(None) == "team memory"


# ---------------------------------------------------------------------------
# Warm-context target resolution
# ---------------------------------------------------------------------------


class TestResolveWarmTargets:
    @pytest.mark.asyncio
    async def test_personal_only_without_org(self) -> None:
        targets = await resolve_warm_targets("u1", None, None)
        assert [t.tier for t in targets] == [MemoryTier.personal]
        assert targets[0].group_id == "user_u1"
        assert targets[0].label is None

    @pytest.mark.asyncio
    async def test_personal_and_org_without_team(self) -> None:
        with patch.object(
            tiers, "is_org_member", new_callable=AsyncMock, return_value=True
        ):
            targets = await resolve_warm_targets("u1", "org-1", None)
        tiers_seen = [t.tier for t in targets]
        assert tiers_seen == [MemoryTier.personal, MemoryTier.org]
        assert targets[1].group_id == "org_org-1"
        assert targets[1].label == "org memory"

    @pytest.mark.asyncio
    async def test_org_tier_excluded_when_not_org_member(self) -> None:
        # A stale/revoked org membership must not reach org memory: the org
        # tier is re-checked here, not trusted from session.organization_id.
        with patch.object(
            tiers, "is_org_member", new_callable=AsyncMock, return_value=False
        ):
            targets = await resolve_warm_targets("u1", "org-1", None)
        assert [t.tier for t in targets] == [MemoryTier.personal]

    @pytest.mark.asyncio
    async def test_includes_session_team_when_active_member(self) -> None:
        with (
            patch.object(
                tiers, "is_org_member", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                tiers,
                "get_user_team_ids",
                new_callable=AsyncMock,
                return_value=["team-1"],
            ),
            patch.object(
                tiers,
                "resolve_team_name",
                new_callable=AsyncMock,
                return_value="Platform",
            ),
        ):
            targets = await resolve_warm_targets("u1", "org-1", "team-1")

        assert [t.tier for t in targets] == [
            MemoryTier.personal,
            MemoryTier.org,
            MemoryTier.team,
        ]
        team_target = targets[2]
        assert team_target.group_id == "team_team-1"
        assert team_target.label == "team memory (Platform)"

    @pytest.mark.asyncio
    async def test_skips_session_team_when_not_active_member(self) -> None:
        with (
            patch.object(
                tiers, "is_org_member", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                tiers,
                "get_user_team_ids",
                new_callable=AsyncMock,
                return_value=["other-team"],
            ),
        ):
            targets = await resolve_warm_targets("u1", "org-1", "team-1")

        # Non-member of the team: team tier is dropped, org still present.
        assert [t.tier for t in targets] == [MemoryTier.personal, MemoryTier.org]

    @pytest.mark.asyncio
    async def test_untagged_session_skips_team(self) -> None:
        # team_id None → team tier never considered (no membership query).
        with (
            patch.object(
                tiers, "is_org_member", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                tiers, "get_user_team_ids", new_callable=AsyncMock
            ) as get_teams,
        ):
            targets = await resolve_warm_targets("u1", "org-1", None)
        get_teams.assert_not_awaited()
        assert [t.tier for t in targets] == [MemoryTier.personal, MemoryTier.org]

    @pytest.mark.asyncio
    async def test_invalid_org_id_preserves_personal(self) -> None:
        # A malformed org id must skip the org tier, not sink personal.
        with patch.object(
            tiers, "derive_org_group_id", side_effect=ValueError("bad org id")
        ):
            targets = await resolve_warm_targets("u1", "bad-org", None)
        assert [t.tier for t in targets] == [MemoryTier.personal]


# ---------------------------------------------------------------------------
# Search target resolution
# ---------------------------------------------------------------------------


class TestResolveSearchTargets:
    @pytest.mark.asyncio
    async def test_all_without_org_is_personal_only(self) -> None:
        targets = await resolve_search_targets("u1", None, "all")
        assert [t.tier for t in targets] == [MemoryTier.personal]

    @pytest.mark.asyncio
    async def test_unknown_tier_raises(self) -> None:
        # An unrecognized tier must be rejected, not silently return no
        # targets (which memory_search would report as "no memories").
        with pytest.raises(TierError, match="Unknown memory tier"):
            await resolve_search_targets("u1", "org-1", "bogus")

    @pytest.mark.asyncio
    async def test_all_unions_personal_org_and_active_teams(self) -> None:
        with (
            patch.object(
                tiers, "is_org_member", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                tiers,
                "get_user_team_ids",
                new_callable=AsyncMock,
                return_value=["team-1", "team-2"],
            ),
            patch.object(
                tiers,
                "resolve_team_names",
                new_callable=AsyncMock,
                return_value={"team-1": "Platform", "team-2": "Growth"},
            ),
        ):
            targets = await resolve_search_targets("u1", "org-1", "all")

        assert [t.tier for t in targets] == [
            MemoryTier.personal,
            MemoryTier.org,
            MemoryTier.team,
            MemoryTier.team,
        ]
        team_labels = {t.label for t in targets if t.tier == MemoryTier.team}
        assert team_labels == {"team memory (Platform)", "team memory (Growth)"}

    @pytest.mark.asyncio
    async def test_org_tier_excluded_for_non_member(self) -> None:
        # tier='org' by a non-member yields no targets (memory_search then
        # reports "no memories") — never the org group's contents.
        with patch.object(
            tiers, "is_org_member", new_callable=AsyncMock, return_value=False
        ):
            targets = await resolve_search_targets("u1", "org-1", "org")
        assert targets == []

    @pytest.mark.asyncio
    async def test_team_tier_only_queries_active_memberships(self) -> None:
        # get_user_team_ids is THE active-membership source; tier='team'
        # returns exactly those teams and nothing else.
        with (
            patch.object(
                tiers,
                "get_user_team_ids",
                new_callable=AsyncMock,
                return_value=["team-1"],
            ),
            patch.object(
                tiers,
                "resolve_team_names",
                new_callable=AsyncMock,
                return_value={"team-1": "Platform"},
            ) as names,
        ):
            targets = await resolve_search_targets("u1", "org-1", "team")

        names.assert_awaited_once_with(["team-1"])
        assert [t.tier for t in targets] == [MemoryTier.team]
        assert targets[0].group_id == "team_team-1"

    @pytest.mark.asyncio
    async def test_org_tier_without_org_raises(self) -> None:
        with pytest.raises(TierError):
            await resolve_search_targets("u1", None, "org")

    @pytest.mark.asyncio
    async def test_team_tier_without_org_raises(self) -> None:
        with pytest.raises(TierError):
            await resolve_search_targets("u1", None, "team")

    @pytest.mark.asyncio
    async def test_personal_tier_is_personal_only(self) -> None:
        targets = await resolve_search_targets("u1", "org-1", "personal")
        assert [t.tier for t in targets] == [MemoryTier.personal]


# ---------------------------------------------------------------------------
# Write governance: membership + hold buffer
# ---------------------------------------------------------------------------


class TestIsOrgAdmin:
    @pytest.mark.asyncio
    async def test_admin_true(self) -> None:
        with patch("backend.data.db.prisma") as mock_prisma:
            mock_prisma.orgmember.find_first = AsyncMock(
                return_value=SimpleNamespace(isAdmin=True, isOwner=False)
            )
            assert await is_org_admin("u1", "org-1") is True

    @pytest.mark.asyncio
    async def test_owner_true(self) -> None:
        with patch("backend.data.db.prisma") as mock_prisma:
            mock_prisma.orgmember.find_first = AsyncMock(
                return_value=SimpleNamespace(isAdmin=False, isOwner=True)
            )
            assert await is_org_admin("u1", "org-1") is True

    @pytest.mark.asyncio
    async def test_plain_member_false(self) -> None:
        with patch("backend.data.db.prisma") as mock_prisma:
            mock_prisma.orgmember.find_first = AsyncMock(
                return_value=SimpleNamespace(isAdmin=False, isOwner=False)
            )
            assert await is_org_admin("u1", "org-1") is False

    @pytest.mark.asyncio
    async def test_non_member_false(self) -> None:
        with patch("backend.data.db.prisma") as mock_prisma:
            mock_prisma.orgmember.find_first = AsyncMock(return_value=None)
            assert await is_org_admin("u1", "org-1") is False


class TestHoldBufferEnabled:
    @pytest.mark.asyncio
    async def test_defaults_true_when_setting_absent(self) -> None:
        with patch("backend.data.db.prisma") as mock_prisma:
            mock_prisma.organization.find_unique = AsyncMock(
                return_value=SimpleNamespace(settings={})
            )
            assert await hold_buffer_enabled("org-1") is True

    @pytest.mark.asyncio
    async def test_false_when_explicitly_disabled(self) -> None:
        with patch("backend.data.db.prisma") as mock_prisma:
            mock_prisma.organization.find_unique = AsyncMock(
                return_value=SimpleNamespace(settings={"memory": {"holdBuffer": False}})
            )
            assert await hold_buffer_enabled("org-1") is False

    @pytest.mark.asyncio
    async def test_true_when_explicitly_enabled(self) -> None:
        with patch("backend.data.db.prisma") as mock_prisma:
            mock_prisma.organization.find_unique = AsyncMock(
                return_value=SimpleNamespace(settings={"memory": {"holdBuffer": True}})
            )
            assert await hold_buffer_enabled("org-1") is True

    @pytest.mark.asyncio
    async def test_defaults_true_when_org_missing(self) -> None:
        with patch("backend.data.db.prisma") as mock_prisma:
            mock_prisma.organization.find_unique = AsyncMock(return_value=None)
            assert await hold_buffer_enabled("org-1") is True

    @pytest.mark.asyncio
    async def test_parses_settings_stored_as_json_string(self) -> None:
        with patch("backend.data.db.prisma") as mock_prisma:
            mock_prisma.organization.find_unique = AsyncMock(
                return_value=SimpleNamespace(
                    settings='{"memory": {"holdBuffer": false}}'
                )
            )
            assert await hold_buffer_enabled("org-1") is False


class TestResolveStoreTeam:
    @pytest.mark.asyncio
    async def test_explicit_team_id_validates_membership(self) -> None:
        membership = SimpleNamespace(teamId="team-1", isAdmin=True)
        with patch.object(
            tiers,
            "get_team_membership",
            new_callable=AsyncMock,
            return_value=membership,
        ) as get_mem:
            result = await resolve_store_team("u1", "org-1", None, "team-1")
        get_mem.assert_awaited_once_with("u1", "team-1", "org-1")
        assert result.teamId == "team-1"

    @pytest.mark.asyncio
    async def test_falls_back_to_single_team(self) -> None:
        membership = SimpleNamespace(teamId="only-team", isAdmin=False)
        with (
            patch.object(
                tiers,
                "get_user_team_ids",
                new_callable=AsyncMock,
                return_value=["only-team"],
            ),
            patch.object(
                tiers,
                "get_team_membership",
                new_callable=AsyncMock,
                return_value=membership,
            ),
        ):
            result = await resolve_store_team("u1", "org-1", None, None)
        assert result.teamId == "only-team"

    @pytest.mark.asyncio
    async def test_no_teams_raises_clear_error(self) -> None:
        with patch.object(
            tiers, "get_user_team_ids", new_callable=AsyncMock, return_value=[]
        ):
            with pytest.raises(TierError, match="not an active member of any team"):
                await resolve_store_team("u1", "org-1", None, None)

    @pytest.mark.asyncio
    async def test_multiple_teams_requires_explicit_id(self) -> None:
        with patch.object(
            tiers,
            "get_user_team_ids",
            new_callable=AsyncMock,
            return_value=["a", "b"],
        ):
            with pytest.raises(TierError, match="multiple teams"):
                await resolve_store_team("u1", "org-1", None, None)

    @pytest.mark.asyncio
    async def test_non_member_of_target_team_raises(self) -> None:
        with patch.object(
            tiers,
            "get_team_membership",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(
                TierError, match="not an active member of the specified"
            ):
                await resolve_store_team("u1", "org-1", "some-team", "some-team")
