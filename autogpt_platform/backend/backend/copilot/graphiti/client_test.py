"""Tests for Graphiti client management — derive_group_id and evict_client."""

import pytest

from .client import derive_group_id, evict_client


class TestDeriveGroupId:
    def test_empty_user_id_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            derive_group_id("")

    def test_all_invalid_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="empty group_id after sanitization"):
            derive_group_id("!!!")

    def test_user_id_with_stripped_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid characters"):
            derive_group_id("abc.def")

    def test_valid_uuid_passthrough(self) -> None:
        uid = "883cc9da-fe37-4863-839b-acba022bf3ef"
        result = derive_group_id(uid)
        assert result == f"user_{uid}"

    def test_simple_alphanumeric_id(self) -> None:
        result = derive_group_id("user123")
        assert result == "user_user123"

    def test_hyphens_and_underscores_allowed(self) -> None:
        result = derive_group_id("a-b_c")
        assert result == "user_a-b_c"


class TestEvictClient:
    @pytest.mark.asyncio
    async def test_evict_nonexistent_group_id_does_not_raise(self) -> None:
        await evict_client("no-such-group-id")


class TestHyphenInGroupIdRegression:
    """Regression coverage for upstream Graphiti issue #1483.

    `AutoGPTFalkorDriver.build_fulltext_query` interpolates the group_id
    into a Redisearch tag filter as ``(@group_id:user_abc-def)``. In
    Redisearch query syntax ``-`` means NOT — so the literal hyphen
    inside a UUID-derived group_id is interpreted as a negation, which
    causes silent search misses on every user whose UUID contains
    hyphens (i.e. every user).

    Upstream tracking: https://github.com/getzep/graphiti/issues/1483

    The xfail test below documents what we *want* the contract to be.
    Once upstream resolves the bug — or we mitigate locally by
    converting hyphens to underscores in ``derive_group_id`` — flip
    the marker off and the regression suite catches a re-introduction.
    """

    def test_derive_group_id_preserves_hyphens(self) -> None:
        """Sanity check: hyphenated UUIDs round-trip through derivation."""
        uid = "a1b2c3d4-e5f6-7890-1234-567890abcdef"
        result = derive_group_id(uid)
        assert result == f"user_{uid}"
        assert "-" in result

    @pytest.mark.xfail(
        reason=(
            "Graphiti #1483: AutoGPTFalkorDriver.build_fulltext_query produces a "
            "Redisearch tag filter that interprets hyphens in the group_id as NOT. "
            "Mitigation deferred — either upstream fix or sanitize in derive_group_id."
        ),
        strict=False,
    )
    def test_build_fulltext_query_escapes_hyphens_in_group_id(self) -> None:
        from .falkordb_driver import AutoGPTFalkorDriver

        # Instantiate without connecting — we only need build_fulltext_query.
        driver = AutoGPTFalkorDriver.__new__(AutoGPTFalkorDriver)
        result = driver.build_fulltext_query(
            query="alice",
            group_ids=["user_a1b2c3d4-e5f6-7890-1234-567890abcdef"],
        )

        # The contract we want: hyphens are escaped (Redisearch backtick
        # form) or otherwise rendered safe for tag-filter matching.
        # Today's output is the raw interpolation, which Redisearch treats
        # as a NOT — hence xfail.
        # Acceptable forms include backtick-escaped values
        #   (@group_id:`user_…-…`)
        # or substituting hyphens with underscores in the filter.
        unsafe_pattern = "user_a1b2c3d4-e5f6-7890"  # raw hyphenated form
        assert unsafe_pattern not in result, (
            "Hyphenated group_id is interpolated raw into the Redisearch tag "
            "filter — Redisearch treats `-` as NOT, causing silent search misses."
        )
