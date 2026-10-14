"""Metadata stamped on every E2B sandbox the platform creates.

E2B sandbox metadata is the one record that outlives Redis keys and process
restarts, and it is what ``SandboxQuery(metadata=...)`` filters on.  Every
creation site builds its metadata here, so a box seen in the E2B dashboard,
or by a reaper, can be traced to the user, the session / expert / graph run,
the block and the deployment that made it.

Keys are prefixed ``autogpt_`` and values are always strings (E2B's
``dict[str, str]`` contract).  ``owner`` + ``kind`` are the identity CoPilot
looks boxes up by; everything else is provenance.

Several products share one E2B team, each with its own metadata schema.  The
one key they have in common is ``service``: a box is only parsed as ours when
``service == "autogpt-platform"`` (see ``SandboxMetadata.parse``), so another
product's keys are never misread and it can ignore ours the same way.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, ConfigDict, ValidationError

from backend.util.settings import Config

if TYPE_CHECKING:
    from backend.data.execution import ExecutionContext

METADATA_PREFIX = "autogpt_"
SERVICE_FIELD = "service"
SERVICE = "autogpt-platform"

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
    user: str | None = None
    session: str | None = None
    expert: str | None = None
    graph: str | None = None
    graph_exec: str | None = None
    node_exec: str | None = None
    block: str | None = None
    template: str | None = None
    mounts: MountState | None = None

    @classmethod
    def for_copilot(
        cls,
        owner: str,
        kind: SandboxUse,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        expert_id: str | None = None,
        template: str | None = None,
        mounts: MountState | None = None,
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
        template: str | None = None,
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

    @classmethod
    def parse(cls, metadata: Mapping[str, str]) -> Optional["SandboxMetadata"]:
        """Read a box's metadata back; ``None`` unless the box is this service's.

        Unknown ``autogpt_*`` keys (from a newer or older build) are ignored;
        a box missing the identity keys is treated as not ours.
        """
        if metadata.get(SERVICE_FIELD) != SERVICE:
            return None
        fields = {
            key.removeprefix(METADATA_PREFIX): value
            for key, value in metadata.items()
            if key.startswith(METADATA_PREFIX)
            and key.removeprefix(METADATA_PREFIX) in cls.model_fields
        }
        try:
            return cls.model_validate(fields)
        except ValidationError:
            return None

    def as_e2b(self) -> dict[str, str]:
        """Flatten to E2B's ``dict[str, str]``: unset fields dropped, keys prefixed."""
        return {SERVICE_FIELD: SERVICE} | {
            f"{METADATA_PREFIX}{key}": value
            for key, value in self.model_dump(exclude_none=True).items()
        }


def owned_by_user(metadata: Mapping[str, str], user_id: str | None) -> bool:
    """Whether a box's metadata says it was created for *user_id*.

    Used before attaching to a sandbox id a caller supplied.  Under the
    platform's E2B key every sandbox id connects, so the stamped user is the
    only thing that makes "reconnect" mean "reconnect to your own box".  A
    box with no parseable metadata is not ours, or predates stamping, and is
    refused either way.
    """
    parsed = SandboxMetadata.parse(metadata)
    return bool(user_id) and parsed is not None and parsed.user == user_id


def deployment_env() -> str:
    """The ``APP_ENV`` this process runs as (``local`` / ``dev`` / ``prod``)."""
    return Config().app_env.value


def _block_owner(context: "ExecutionContext", block_id: str) -> str:
    if context.graph_exec_id:
        return f"graph_exec:{context.graph_exec_id}"
    if context.user_id:
        return f"user:{context.user_id}"
    return f"block:{block_id}"
