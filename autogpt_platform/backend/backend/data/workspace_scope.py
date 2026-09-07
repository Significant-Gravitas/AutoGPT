"""Trusted resource scope for expert chat sessions.

Resolved from persisted session attribution — never from tool arguments. An
expert session may read and write under its own sessions and its own skills
folder, and read under sessions it delegated. ``None`` scope means
unrestricted: the account owner acting through personal AutoPilot, REST
endpoints, or system paths.
"""

import posixpath
from typing import cast

import prisma
from prisma.enums import ResourceVisibility
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import Expert as PrismaExpert
from pydantic import BaseModel, Field

SESSIONS_ROOT = "/sessions/"
EXPERTS_ROOT = "/experts/"

EXPERT_FILE_ACCESS_DENIED = (
    "This file is outside this expert's scope. Experts can only access files "
    "from their own sessions and their own skills. Open personal AutoPilot to "
    "work with other files."
)
EXPERT_SKILL_SCOPE_DENIED = (
    "Experts can only use and manage their own skills. Open personal AutoPilot "
    "to manage another expert's skills or the account's skills."
)


class WorkspaceAccessDeniedError(PermissionError):
    """A workspace operation fell outside the caller's resolved scope."""


def session_path_prefix(session_id: str) -> str:
    return f"{SESSIONS_ROOT}{session_id}/"


def expert_skills_folder(expert_id: str) -> str:
    return f"{EXPERTS_ROOT}{expert_id}/skills"


class WorkspaceScope(BaseModel):
    """Grants for one expert. Safe for RPC transport."""

    expert_id: str
    session_ids: list[str] = Field(default_factory=list)
    delegated_session_ids: list[str] = Field(default_factory=list)

    def with_session(self, session_id: str) -> "WorkspaceScope":
        if session_id in self.session_ids:
            return self
        return self.model_copy(update={"session_ids": [*self.session_ids, session_id]})

    @property
    def skills_prefix(self) -> str:
        return f"{expert_skills_folder(self.expert_id)}/"

    @property
    def write_prefixes(self) -> list[str]:
        return [session_path_prefix(s) for s in self.session_ids] + [self.skills_prefix]

    @property
    def read_prefixes(self) -> list[str]:
        return self.write_prefixes + [
            session_path_prefix(s) for s in self.delegated_session_ids
        ]

    def allows_path(self, path: str, *, write: bool = False) -> bool:
        if not path.startswith("/") or "\\" in path or posixpath.normpath(path) != path:
            return False
        prefixes = self.write_prefixes if write else self.read_prefixes
        return any(path.startswith(prefix) for prefix in prefixes)


async def resolve_expert_workspace_scope(
    user_id: str, expert_id: str
) -> WorkspaceScope:
    """Resolve the grants for *expert_id* owned by *user_id*.

    Fails closed: a missing, archived, or foreign expert yields a scope with
    no session grants at all (callers add the current session explicitly).
    """
    expert = await PrismaExpert.prisma().find_first(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            "visibility": ResourceVisibility.PRIVATE,
        }
    )
    if expert is None:
        return _NoGrants(expert_id=expert_id)

    own_sessions = await PrismaChatSession.prisma().find_many(
        where={"userId": user_id, "expertId": expert_id}
    )
    delegated_sessions = await PrismaChatSession.prisma().find_many(
        where={
            "userId": user_id,
            "metadata": cast(
                prisma.types.JsonFilter,
                {"path": ["delegated_by_expert_id"], "equals": prisma.Json(expert_id)},
            ),
        }
    )
    return WorkspaceScope(
        expert_id=expert_id,
        session_ids=[row.id for row in own_sessions],
        delegated_session_ids=[
            row.id for row in delegated_sessions if row.expertId != expert_id
        ],
    )


class _NoGrants(WorkspaceScope):
    """Scope for an expert that no longer exists for this owner."""

    @property
    def write_prefixes(self) -> list[str]:
        return [session_path_prefix(s) for s in self.session_ids]
