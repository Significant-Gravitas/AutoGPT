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
