"""Trusted file scope for expert chat sessions.

Resolved from persisted session attribution, never from tool arguments. An
expert session may read and write under its own conversations and its own
skills folder, read under conversations it delegated, and read the owner's
own files. ``None`` scope means unrestricted: the account owner acting
through personal Otto, REST endpoints, or system paths.
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
SKILLS_ROOT = "/skills/"

# Roots a WorkspaceManager takes literally instead of resolving under the
# calling session: a skill package is shared across every session, so a
# session prefix would make its files unreachable from the turn that read it.
SHARED_ROOTS = (SESSIONS_ROOT, SKILLS_ROOT, EXPERTS_ROOT)

EXPERT_FILE_ACCESS_DENIED = (
    "This file is outside this expert's scope. Experts can read your own files "
    "and their own conversations, and can only write to their own "
    "conversations and skills. Open a chat with Otto to work with another "
    "expert's files."
)
EXPERT_SKILL_SCOPE_DENIED = (
    "Experts can only use and manage their own skills. Open a chat with Otto "
    "to manage another expert's skills or the account's skills."
)


class WorkspaceAccessDeniedError(PermissionError):
    """A workspace operation fell outside the caller's resolved scope."""


def session_path_prefix(session_id: str) -> str:
    return f"{SESSIONS_ROOT}{session_id}/"


def expert_skills_folder(expert_id: str) -> str:
    return f"{EXPERTS_ROOT}{expert_id}/skills"


def is_user_file_path(path: str) -> bool:
    """Whether *path* is one of the owner's own files.

    Everything outside the three managed roots: what the user uploads on the
    Files page, at the workspace root or in a folder. A predicate over the
    roots rather than a ``/`` prefix, which ``allows_path``'s ``startswith``
    would match against every path in the workspace.
    """
    return not path.startswith(SHARED_ROOTS)


class WorkspaceScope(BaseModel):
    """Grants for one acting session. Safe for RPC transport.

    ``expert_id`` is ``None`` when nobody could be attributed; such a scope
    only carries the sessions a caller added explicitly.
    """

    expert_id: str | None = None
    session_ids: list[str] = Field(default_factory=list)
    delegated_session_ids: list[str] = Field(default_factory=list)
    # Carried grants, never inferred from ``expert_id``: this model crosses an
    # RPC boundary and is rebuilt as this class, so a subclass that withheld a
    # grant would come back granting it. Defaulting to False also keeps the
    # unattributed scope (``WorkspaceScope(session_ids=[sid])``) fail-closed.
    owns_skills_folder: bool = False
    # Read-only: the owner's own files, shared by every expert they hire.
    reads_user_files: bool = False

    def with_session(self, session_id: str) -> "WorkspaceScope":
        if session_id in self.session_ids:
            return self
        return self.model_copy(update={"session_ids": [*self.session_ids, session_id]})

    @property
    def skills_prefix(self) -> str | None:
        if self.expert_id is None or not self.owns_skills_folder:
            return None
        return f"{expert_skills_folder(self.expert_id)}/"

    @property
    def write_prefixes(self) -> list[str]:
        own = [session_path_prefix(s) for s in self.session_ids]
        return own if self.skills_prefix is None else own + [self.skills_prefix]

    @property
    def read_prefixes(self) -> list[str]:
        return self.write_prefixes + [
            session_path_prefix(s) for s in self.delegated_session_ids
        ]

    def allows_path(self, path: str, *, write: bool = False) -> bool:
        if not path.startswith("/") or "\\" in path or posixpath.normpath(path) != path:
            return False
        if not write and self.reads_user_files and is_user_file_path(path):
            return True
        prefixes = self.write_prefixes if write else self.read_prefixes
        return any(path.startswith(prefix) for prefix in prefixes)


async def resolve_expert_workspace_scope(
    user_id: str, expert_id: str
) -> WorkspaceScope:
    """Resolve the grants for *expert_id* owned by *user_id*.

    Own conversations are every session attributed to the expert, so a new
    conversation keeps reaching files from earlier ones, and the owner's own
    files are readable so an expert can work on what the owner uploaded.
    Fails closed: a missing, archived, or foreign expert yields no grants at
    all, user files included (callers add the current session explicitly,
    which grants nothing beyond that one conversation). The ``visibility`` filter
    mirrors ``experts_db.get_expert``: hired experts are PRIVATE in v1, and
    a TEAM/ORG expert must not read files until sharing rules exist for it.
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
        return WorkspaceScope(expert_id=expert_id)

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
        owns_skills_folder=True,
        reads_user_files=True,
        session_ids=[row.id for row in own_sessions],
        delegated_session_ids=[
            row.id for row in delegated_sessions if row.expertId != expert_id
        ],
    )
