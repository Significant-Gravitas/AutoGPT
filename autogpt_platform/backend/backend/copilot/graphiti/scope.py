"""Which memory an operation reads and writes: the account's or one expert's.

A ``MemoryScope`` is built once where a request enters the memory code (API
routes, chat tools, ingestion, the dream and community-rebuild entry points)
and passed down. Internal code reads ``group_id`` / ``scope_key`` off it
instead of re-deriving them from a loose ``(user_id, expert_id)`` pair, so no
caller can land in the account's graph by forgetting the ``expert_id``.

The derivations stay in ``client.py``; this module only carries their
results and owns the Redis key layouts keyed on them.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .client import derive_memory_group_id, derive_memory_scope_key

RedisKeyFamily = Literal[
    "dream_lock", "last_completed", "hits", "rebuild_lock", "registration"
]

# One in-flight dream per scope (dream/locks.py).
DREAM_LOCK_KEY_PREFIX = "dream:inflight:"
# When the scope's last sync dream pass completed (dream/orchestrator.py).
LAST_COMPLETED_KEY_PREFIX = "dream:last_completed:"
# Warm-context hit counters for tentative edges, one key per edge so INCR
# never contends and each TTL cleans up on its own (dream/ratification_hits.py).
HIT_TRACKER_KEY_PREFIX = "mem:hits"
# One community rebuild per graph (graphiti/communities.py).
REBUILD_LOCK_KEY_PREFIX = "graphiti:community_rebuild_lock:"


class MemoryScope(BaseModel):
    """The owner, the expert (if any), and the ids derived from them.

    ``group_id`` is the Graphiti namespace and FalkorDB database;
    ``scope_key`` keys queues, locks and Redis markers. Both are validated
    against the derivations, so a scope can never carry an id that does not
    belong to its owner and expert.
    """

    model_config = ConfigDict(frozen=True)

    owner_user_id: str = Field(min_length=1)
    expert_id: str | None = Field(min_length=1)
    group_id: str
    scope_key: str

    @classmethod
    def for_user(cls, user_id: str) -> "MemoryScope":
        """The account (Otto) scope. Raises ``ValueError`` for an invalid id."""
        return cls(
            owner_user_id=user_id,
            expert_id=None,
            group_id=derive_memory_group_id(user_id),
            scope_key=derive_memory_scope_key(user_id),
        )

    @classmethod
    def for_expert(cls, user_id: str, expert_id: str) -> "MemoryScope":
        """One expert's scope. Raises ``ValueError`` for an invalid id."""
        return cls(
            owner_user_id=user_id,
            expert_id=expert_id,
            group_id=derive_memory_group_id(user_id, expert_id),
            scope_key=derive_memory_scope_key(user_id, expert_id),
        )

    @classmethod
    def build(cls, user_id: str, expert_id: str | None = None) -> "MemoryScope":
        """``for_user`` when ``expert_id`` is None, else ``for_expert``."""
        if expert_id is None:
            return cls.for_user(user_id)
        return cls.for_expert(user_id, expert_id)

    @property
    def is_expert(self) -> bool:
        return self.expert_id is not None

    def redis_key(
        self,
        family: RedisKeyFamily,
        *,
        edge_uuid: str | None = None,
        registration_prefix: str | None = None,
    ) -> str:
        """This scope's Redis key for ``family``, exactly as it was keyed
        before this class existed.

        Most families key on ``scope_key`` (the raw user id for the account,
        the group id for an expert). Two do not, and keep their layouts:
        ``rebuild_lock`` keys on ``group_id``, and ``registration`` keys on
        the owner's user id even for an expert scope, because the dream
        crons are registered per user. ``hits`` needs the counted
        ``edge_uuid``; ``registration`` needs the cron's marker prefix (see
        ``dream/scheduling.py``).
        """
        match family:
            case "dream_lock":
                return f"{DREAM_LOCK_KEY_PREFIX}{self.scope_key}"
            case "last_completed":
                return f"{LAST_COMPLETED_KEY_PREFIX}{self.scope_key}"
            case "hits":
                if edge_uuid is None:
                    raise ValueError("the hits key needs an edge_uuid")
                return f"{HIT_TRACKER_KEY_PREFIX}:{self.scope_key}:{edge_uuid}"
            case "rebuild_lock":
                return f"{REBUILD_LOCK_KEY_PREFIX}{self.group_id}"
            case "registration":
                if registration_prefix is None:
                    raise ValueError("the registration key needs a prefix")
                return f"{registration_prefix}:{self.owner_user_id}"

    @model_validator(mode="after")
    def _ids_match_derivation(self) -> "MemoryScope":
        derived = (
            derive_memory_group_id(self.owner_user_id, self.expert_id),
            derive_memory_scope_key(self.owner_user_id, self.expert_id),
        )
        if (self.group_id, self.scope_key) != derived:
            raise ValueError(
                "group_id and scope_key must be derived from owner_user_id "
                "and expert_id"
            )
        return self
