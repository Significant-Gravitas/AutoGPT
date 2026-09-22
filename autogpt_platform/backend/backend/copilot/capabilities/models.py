"""Data model for the copilot capability registry.

One :class:`CapabilityEntry` describes one thing the copilot can do: a
platform tool, a block, an MCP server from the catalog, a skill of the
session's owner, or (phase 2) a library agent.  The registry indexes
entries by name, description and tags; argument schemas are fetched on
demand and never indexed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CapabilityKindName = Literal[
    "tool", "block", "mcp_server", "mcp_tool", "agent", "skill"
]
CapabilityClass = Literal["service", "primitive"]
CapabilityContext = Literal["direct", "graph", "both"]
ConnectionKeyType = Literal["provider", "server_url", "host", "none"]

PURPOSE_MAX_CHARS = 160
# Bounds the index-only description: room for any real block, tool or catalog
# text (the longest block description is under 1,000 chars), and a ceiling on
# what a runaway one can add to the index and its memory.
DESCRIPTION_MAX_CHARS = 4000
# The tool a skill runs through.  A turn that may not call it may not see
# skills either, and a ``skill:`` dispatch is a call to it.
SKILL_TOOL = "read_skill"


class Implementation(BaseModel):
    """One way to run a capability.

    ``bash_exec`` (a tool) and ``ExecuteCodeBlock`` (a block) are one
    capability with two implementations: the tool for direct use, the
    block inside an agent graph.
    """

    kind: CapabilityKindName
    ref: str
    name: str | None = None
    context: CapabilityContext = "both"


class Connection(BaseModel):
    """What the capability needs before it can run, and whether it has it.

    ``connected`` is ``None`` until the caller resolves it against the
    user's credentials (``key_type`` says which set to look in).  Host-keyed
    primitives are resolved from the request URL at describe/run time, so
    their ``key`` is ``None`` here.
    """

    required: bool = False
    key_type: ConnectionKeyType = "none"
    key: str | None = None
    connected: bool | None = None


class CapabilityEntry(BaseModel):
    id: str
    kind: CapabilityKindName
    klass: CapabilityClass = "service"
    name: str
    purpose: str = Field(max_length=PURPOSE_MAX_CHARS)
    # Everything the source says about the capability, for the index only:
    # ``purpose`` is the first sentence or two of it, sized for a listing,
    # and the sentence that names what a block does in CoPilot ("saves to
    # workspace") is often the one clipped away.  Never listed.
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    context: CapabilityContext = "both"
    implementations: list[Implementation] = Field(default_factory=list)
    connection: Connection = Field(default_factory=Connection)
    # Argument names give the index a lexical signal ("subject", "channel")
    # without indexing the full schema.
    argument_names: list[str] = Field(default_factory=list)
    schema_ref: str | None = None
    # Already in the model's tool list; a search still returns it so the
    # model is pointed back at the tool it has instead of a block copy.
    eager: bool = False
    # Blocks the platform hides for a reason (disabled / graph-only) keep an
    # entry for graph context but stay out of direct results.
    sensitive: bool = False

    def listing(self) -> dict[str, object]:
        """Compact form for a search result (about 40-60 tokens)."""
        out: dict[str, object] = {
            "id": self.id,
            "name": self.name,
            "purpose": self.purpose,
            "kind": self.kind,
        }
        if self.klass == "primitive":
            out["class"] = "primitive"
        if self.connection.required:
            out["connected"] = self.connection.connected
        if self.eager:
            out["eager"] = True
        return out

    def available_in(self, context: CapabilityContext) -> bool:
        return self.context == "both" or context == "both" or self.context == context


def normalize_text(text: str | None, limit: int = DESCRIPTION_MAX_CHARS) -> str:
    """*text* with its whitespace collapsed and cut at *limit* on a word
    boundary, for the index."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    # One past the limit, so a word that ends exactly there is kept whole.
    cut = text[: limit + 1]
    idx = cut.rfind(" ")
    return (cut[:idx] if idx >= limit // 2 else cut[:limit]).rstrip()


def clip_purpose(text: str | None, limit: int = PURPOSE_MAX_CHARS) -> str:
    """First sentence-ish of *text*, at most *limit* characters."""
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    for sep in (". ", "; ", ", "):
        idx = cut.rfind(sep)
        if idx >= limit // 4:
            return cut[: idx + (1 if sep == ". " else 0)].rstrip()
    idx = cut.rfind(" ")
    return (cut[:idx] if idx >= limit // 2 else cut).rstrip() + "…"
