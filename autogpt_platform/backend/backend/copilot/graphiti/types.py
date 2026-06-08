"""Custom Graphiti entity + edge types for the CoPilot memory domain.

Per the integration audit (dream/dreaming-graphiti.md §6.4), passing
``entity_types`` and ``edge_types`` to ``Graphiti.add_episode()`` so that
``MemoryEnvelope`` metadata (``status``, ``confidence``, ``source_kind``,
``scope``, ``provenance``) survives the LLM extraction step and lives on
the durable ``:RELATES_TO`` edge — not only in the ``:Episodic.content``
JSON blob, which Cypher cannot filter without parsing.

After this lands, search filters can do ``WHERE e.status = 'active'``
natively and ratification can flip ``status`` with a single SET.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from .memory_model import MemoryStatus, SourceKind


class MemoryFact(BaseModel):
    """Custom Graphiti edge type for :RELATES_TO edges in our domain.

    Carries the structured metadata fields that the LLM extractor would
    otherwise discard. Each field maps 1:1 to ``MemoryEnvelope`` so that
    round-trips between the envelope JSON (on the episode) and the edge
    properties preserve the same shape.
    """

    status: MemoryStatus = Field(
        default=MemoryStatus.active,
        description="Lifecycle state — active, tentative, superseded, contradicted.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in [0, 1]; None means unspecified.",
    )
    source_kind: SourceKind = Field(
        default=SourceKind.user_asserted,
        description="Whether the fact came from the user, the assistant, or a tool observation.",
    )
    scope: str = Field(
        default="real:global",
        description="Namespace — 'real:global', 'project:<name>', 'book:<title>', 'session:<id>'.",
    )
    provenance: str | None = Field(
        default=None,
        description="Origin reference — 'session:{id}#msg:{sequence}', tool_call_id, or URL.",
    )
    web_verified_at: datetime | None = Field(
        default=None,
        description="When (if ever) this fact was verified against external web sources.",
    )
    ratified_at: datetime | None = Field(
        default=None,
        description="When a tentative fact was promoted to active via the ratification loop.",
    )
    expiration_reason: str | None = Field(
        default=None,
        description=(
            "Short reason recorded when a fact is retracted or superseded. Examples: "
            "'stale_fact', 'contradicted_by:{uuid}', 'entity_invalidated:{uuid}', "
            "'web_contradicted:{url}', 'user_signal', 'unratified'."
        ),
    )


# Entity types — kept narrow at v1. Expand only when extraction quality
# issues prove that a missing type is the cause (per audit §6.3).
class Person(BaseModel):
    role: str | None = Field(
        default=None, description="Role or title at an organization."
    )
    email: str | None = Field(default=None, description="Contact email if mentioned.")


class Organization(BaseModel):
    industry: str | None = Field(default=None, description="Industry or sector.")


class Project(BaseModel):
    # Named ``project_status`` (not ``status``) to disambiguate from
    # ``MemoryFact.status`` (lifecycle of a fact) — different Cypher
    # namespaces but readers see the same field name with conflicting
    # types otherwise.
    project_status: str | None = Field(
        default=None,
        description="Project status — active, completed, paused, cancelled.",
    )


class Concept(BaseModel):
    """Domain concepts that aren't people, organizations, or projects."""


class Preference(BaseModel):
    """User preferences and stated likes/dislikes."""


class Rule(BaseModel):
    """Standing instructions (matches MemoryKind.rule)."""


ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "Person": Person,
    "Organization": Organization,
    "Project": Project,
    "Concept": Concept,
    "Preference": Preference,
    "Rule": Rule,
}

EDGE_TYPES: dict[str, type[BaseModel]] = {"MemoryFact": MemoryFact}

# Allow MemoryFact between any pair of our entity types. Graphiti uses
# this to constrain edge labels during extraction; allowing all pairs
# keeps the LLM free to extract any meaningful relationship.
EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    (src, tgt): ["MemoryFact"] for src in ENTITY_TYPES for tgt in ENTITY_TYPES
}
