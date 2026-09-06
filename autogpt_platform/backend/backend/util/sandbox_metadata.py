"""Metadata stamped on every E2B sandbox the platform creates.

E2B sandbox metadata is the one record that outlives Redis keys and process
restarts, and it is what ``SandboxQuery(metadata=...)`` filters on.  Every
creation site builds its metadata here, so a box seen in the E2B dashboard,
or by a reaper, can be traced to the user, the session / expert / graph run,
the block and the deployment that made it.

Keys are prefixed ``autogpt_`` and values are always strings (E2B's
``dict[str, str]`` contract).  ``owner`` + ``kind`` are the identity CoPilot
looks boxes up by; everything else is provenance.
"""

from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, ConfigDict

from backend.util.settings import Config

if TYPE_CHECKING:
    from backend.data.execution import ExecutionContext

METADATA_PREFIX = "autogpt_"

SandboxSource = Literal["copilot", "block"]
SandboxUse = Literal["shell", "desktop", "code", "claude_code"]
MountState = Literal["attached", "none"]


class SandboxMetadata(BaseModel):
    """Typed view of the ``autogpt_*`` keys; ``as_e2b()`` flattens it for the SDK."""

    model_config = ConfigDict(frozen=True)

    owner: str
    kind: SandboxUse
    source: SandboxSource
    env: str
    user: Optional[str] = None
    session: Optional[str] = None
    expert: Optional[str] = None
    graph: Optional[str] = None
    graph_exec: Optional[str] = None
    node_exec: Optional[str] = None
    block: Optional[str] = None
    template: Optional[str] = None
    mounts: Optional[MountState] = None

    @classmethod
    def for_copilot(
        cls,
        owner: str,
        kind: SandboxUse,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        expert_id: Optional[str] = None,
        template: Optional[str] = None,
        mounts: Optional[MountState] = None,
    ) -> "SandboxMetadata":
        """Metadata for a CoPilot box; *owner* is ``"<kind>:<id>"`` from ``SandboxOwner``."""
        return cls(
            owner=owner,
            kind=kind,
            source="copilot",
            env=deployment_env(),
            user=user_id,
            session=session_id,
            expert=expert_id,
            template=template,
            mounts=mounts,
        )

    @classmethod
    def for_block(
        cls,
        context: "ExecutionContext | None",
        kind: SandboxUse,
        block_id: str,
        template: Optional[str] = None,
    ) -> "SandboxMetadata":
        """Metadata for a box a graph block creates, traced to its node execution."""
        if context is None:
            return cls(
                owner=f"block:{block_id}",
                kind=kind,
                source="block",
                env=deployment_env(),
                block=block_id,
                template=template or None,
            )
        return cls(
            owner=_block_owner(context, block_id),
            kind=kind,
            source="block",
            env=deployment_env(),
            user=context.user_id,
            session=context.session_id,
            expert=context.expert_id,
            graph=context.graph_id,
            graph_exec=context.graph_exec_id,
            node_exec=context.node_exec_id,
            block=block_id,
            template=template or None,
        )

    def as_e2b(self) -> dict[str, str]:
        """Flatten to E2B's ``dict[str, str]``: unset fields dropped, keys prefixed."""
        return {
            f"{METADATA_PREFIX}{key}": value
            for key, value in self.model_dump(exclude_none=True).items()
        }


def deployment_env() -> str:
    """The ``APP_ENV`` this process runs as (``local`` / ``dev`` / ``prod``)."""
    return Config().app_env.value


def _block_owner(context: "ExecutionContext", block_id: str) -> str:
    if context.graph_exec_id:
        return f"graph_exec:{context.graph_exec_id}"
    if context.user_id:
        return f"user:{context.user_id}"
    return f"block:{block_id}"
