"""Tests for MemoryScope: ids and Redis keys must match the legacy derivations."""

import pytest
from pydantic import ValidationError

from backend.copilot.dream.scheduling import (
    COMMUNITY_REBUILD_REGISTRATION_PREFIX,
    NIGHTLY_BATCH_REGISTRATION_PREFIX,
)

from .client import derive_group_id, derive_memory_group_id, derive_memory_scope_key
from .scope import (
    DREAM_LOCK_KEY_PREFIX,
    HIT_TRACKER_KEY_PREFIX,
    LAST_COMPLETED_KEY_PREFIX,
    REBUILD_LOCK_KEY_PREFIX,
    MemoryScope,
)

USER_ID = "883cc9da-fe37-4863-839b-acba022bf3ef"
EXPERT_ID = "11111111-2222-3333-4444-555555555555"

BOTH_SCOPES = pytest.mark.parametrize("expert_id", [None, EXPERT_ID])


class TestDerivedIds:
    def test_account_scope_matches_legacy_derivations(self) -> None:
        scope = MemoryScope.for_user(USER_ID)

        assert scope.owner_user_id == USER_ID
        assert scope.expert_id is None
        assert not scope.is_expert
        assert scope.group_id == derive_memory_group_id(USER_ID)
        assert scope.group_id == derive_group_id(USER_ID)
        assert scope.scope_key == derive_memory_scope_key(USER_ID)
        assert scope.scope_key == USER_ID

    def test_expert_scope_matches_legacy_derivations(self) -> None:
        scope = MemoryScope.for_expert(USER_ID, EXPERT_ID)

        assert scope.owner_user_id == USER_ID
        assert scope.expert_id == EXPERT_ID
        assert scope.is_expert
        assert scope.group_id == derive_memory_group_id(USER_ID, EXPERT_ID)
        assert scope.scope_key == derive_memory_scope_key(USER_ID, EXPERT_ID)
        assert scope.scope_key == scope.group_id

    def test_expert_and_account_never_share_a_graph(self) -> None:
        account = MemoryScope.for_user(USER_ID)
        expert = MemoryScope.for_expert(USER_ID, EXPERT_ID)

        assert expert.group_id != account.group_id
        assert expert.scope_key != account.scope_key

    @BOTH_SCOPES
    def test_build_dispatches_on_expert_id(self, expert_id: str | None) -> None:
        expected = (
            MemoryScope.for_user(USER_ID)
            if expert_id is None
            else MemoryScope.for_expert(USER_ID, expert_id)
        )
        assert MemoryScope.build(USER_ID, expert_id) == expected

    def test_scope_is_frozen_and_hashable(self) -> None:
        scope = MemoryScope.for_user(USER_ID)

        with pytest.raises(ValidationError):
            scope.group_id = "user_someone_else"
        assert {scope: 1}[MemoryScope.for_user(USER_ID)] == 1


class TestRedisKeys:
    """Each family must equal the key its owning module built before
    ``MemoryScope`` existed, for the account and for an expert."""

    @BOTH_SCOPES
    def test_dream_lock_keys_on_scope_key(self, expert_id: str | None) -> None:
        scope = MemoryScope.build(USER_ID, expert_id)
        legacy = f"{DREAM_LOCK_KEY_PREFIX}{derive_memory_scope_key(USER_ID, expert_id)}"
        assert scope.redis_key("dream_lock") == legacy

    @BOTH_SCOPES
    def test_last_completed_keys_on_scope_key(self, expert_id: str | None) -> None:
        scope = MemoryScope.build(USER_ID, expert_id)
        scope_key = derive_memory_scope_key(USER_ID, expert_id)
        legacy = f"{LAST_COMPLETED_KEY_PREFIX}{scope_key}"
        assert scope.redis_key("last_completed") == legacy

    @BOTH_SCOPES
    def test_hits_key_on_scope_key_and_edge(self, expert_id: str | None) -> None:
        scope = MemoryScope.build(USER_ID, expert_id)
        scope_key = derive_memory_scope_key(USER_ID, expert_id)
        legacy = f"{HIT_TRACKER_KEY_PREFIX}:{scope_key}:edge-1"
        assert scope.redis_key("hits", edge_uuid="edge-1") == legacy

    @BOTH_SCOPES
    def test_rebuild_lock_keys_on_group_id(self, expert_id: str | None) -> None:
        scope = MemoryScope.build(USER_ID, expert_id)
        group_id = derive_memory_group_id(USER_ID, expert_id)
        legacy = f"{REBUILD_LOCK_KEY_PREFIX}{group_id}"
        assert scope.redis_key("rebuild_lock") == legacy

    @BOTH_SCOPES
    @pytest.mark.parametrize(
        "prefix",
        [COMMUNITY_REBUILD_REGISTRATION_PREFIX, NIGHTLY_BATCH_REGISTRATION_PREFIX],
    )
    def test_registration_keys_on_owner_user(
        self, expert_id: str | None, prefix: str
    ) -> None:
        scope = MemoryScope.build(USER_ID, expert_id)
        legacy = f"{prefix}:{USER_ID}"
        assert scope.redis_key("registration", registration_prefix=prefix) == legacy

    def test_hits_key_requires_edge_uuid(self) -> None:
        with pytest.raises(ValueError, match="edge_uuid"):
            MemoryScope.for_user(USER_ID).redis_key("hits")

    def test_registration_key_requires_prefix(self) -> None:
        with pytest.raises(ValueError, match="prefix"):
            MemoryScope.for_user(USER_ID).redis_key("registration")


class TestValidation:
    def test_empty_user_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="user_id must be non-empty"):
            MemoryScope.for_user("")
        with pytest.raises(ValueError, match="user_id must be non-empty"):
            MemoryScope.for_expert("", EXPERT_ID)
        with pytest.raises(ValueError, match="user_id must be non-empty"):
            MemoryScope.build("")

    def test_empty_expert_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="expert_id must be non-empty"):
            MemoryScope.for_expert(USER_ID, "")
        with pytest.raises(ValueError, match="expert_id must be non-empty"):
            MemoryScope.build(USER_ID, "")

    def test_invalid_characters_raise_the_legacy_error(self) -> None:
        with pytest.raises(ValueError, match="invalid characters"):
            MemoryScope.for_user("abc.def")
        with pytest.raises(ValueError, match="invalid characters"):
            MemoryScope.for_expert(USER_ID, "expert/1")

    def test_direct_construction_rejects_empty_ids(self) -> None:
        account = MemoryScope.for_user(USER_ID)
        with pytest.raises(ValidationError):
            MemoryScope(
                owner_user_id="",
                expert_id=None,
                group_id=account.group_id,
                scope_key=account.scope_key,
            )
        with pytest.raises(ValidationError):
            MemoryScope(
                owner_user_id=USER_ID,
                expert_id="",
                group_id=account.group_id,
                scope_key=account.scope_key,
            )

    def test_direct_construction_rejects_ids_from_another_scope(self) -> None:
        """An expert scope carrying the account's ids is the silent-fallback
        bug this type exists to prevent."""
        account = MemoryScope.for_user(USER_ID)
        with pytest.raises(ValidationError, match="must be derived"):
            MemoryScope(
                owner_user_id=USER_ID,
                expert_id=EXPERT_ID,
                group_id=account.group_id,
                scope_key=account.scope_key,
            )
