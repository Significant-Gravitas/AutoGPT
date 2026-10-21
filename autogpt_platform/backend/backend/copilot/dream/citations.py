"""What a dream write rests on, checked against what its pass read.

Every consolidated fact and proposal must cite the facts and episodes it was
drawn from. apply keeps a citation only when it names a fact or an episode
the pass read (its input bundle), filed under the kind the bundle says it
is, in the model's order and once each; a write left citing nothing is
dropped before it is queued and counted in ``uncited_writes_dropped``. So a
made-up uuid never reaches the graph, and no dream write rests on nothing.

Scope binds the fact citations, as the prompts tell the model ("group by
scope", a finding stays in the scope of the facts it cites). A fact's scope
is its own, ``UNSCOPED`` (the default everywhere) when it names none. A
write citing a fact of another scope is dropped whole, not trimmed to its
own scope's citations: trimmed, it could restate the other fact and escape
that fact's forget. It is counted in ``uncited_writes_dropped`` too, and
each such citation in ``cross_scope_citations_dropped``. Episode citations
are not scoped: a chat turn is raw material that can hold facts of any
scope, a project fact consolidated from one is legitimate, and the cascade
reaches it through ``derived_from_episodes`` when a forget hides that turn.
A source in the right scope that the write does not truly rest on is not
caught: no deterministic check can tell.

What survives goes to the ingestion worker, which checks it against the
forgets made since the pass read the graph (``graphiti/recall_citations.py``)
and then records it on the dream's episode and on the facts only dream
episodes state (``graphiti/recall_derivation.py``). A later forget of
anything it cites retracts those facts (``graphiti/recall_cascade.py``). The
episode's free-text ``source_description`` no longer lists what it cites:
the marker and the records carry that, and only the derivation backfill
reads the lists older descriptions hold
(``graphiti/migrations/legacy_citations.py``).
"""

from collections.abc import Collection, Mapping, Sequence

from pydantic import BaseModel, ValidationError

from backend.copilot.graphiti.recall_citations import Citations

from .fetch import DreamInput

# The scope of a fact or an envelope that names none: ``MemoryFact``'s and
# ``MemoryEnvelope``'s default, and how the prompts list an unscoped fact.
UNSCOPED = "real:global"


class CheckedCitations(BaseModel):
    """What apply keeps of a write's citations: ``None`` when the write is
    dropped, for citing nothing the pass read or a fact of another scope;
    ``cross_scope`` counts its citations of a fact of another scope."""

    citations: Citations | None = None
    cross_scope: int = 0


def validated_citations(
    fact_uuids: Sequence[str],
    episode_uuids: Sequence[str],
    *,
    scope: str,
    known_facts: Collection[str],
    known_episodes: Collection[str],
    fact_scopes: Mapping[str, str],
) -> CheckedCitations:
    """The write's citations that name something its pass read, each under
    the kind the pass read it as; none at all when one names a fact of
    another scope than the write's ``scope``. ``fact_scopes`` gives each
    fact's scope, ``UNSCOPED`` when absent; an episode's is not checked."""
    cited = list(dict.fromkeys([*fact_uuids, *episode_uuids]))
    facts = [uuid for uuid in cited if uuid in known_facts]
    episodes = [uuid for uuid in cited if uuid in known_episodes]
    wanted = scope_key(scope)
    elsewhere = [uuid for uuid in facts if scope_key(fact_scopes.get(uuid)) != wanted]
    if elsewhere or not (facts or episodes):
        return CheckedCitations(cross_scope=len(elsewhere))
    return CheckedCitations(
        citations=Citations(fact_uuids=facts, episode_uuids=episodes)
    )


def scope_key(scope: str | None) -> str:
    """``scope`` as the prompts list it: whitespace collapsed, ``UNSCOPED``
    when it names none."""
    return " ".join((scope or "").split()) or UNSCOPED


def fact_scopes(input_bundle: DreamInput) -> dict[str, str]:
    """The scope of every fact a pass read, ``UNSCOPED`` when it names none."""
    return {fact.uuid: scope_key(fact.scope) for fact in input_bundle.facts}


class _EnvelopeScope(BaseModel):
    scope: str | None = None


def envelope_scope(content: str | None) -> str:
    """The scope a ``MemoryEnvelope`` body declares; ``UNSCOPED`` for any
    other episode (a chat turn is plain text). The derivation backfill reads
    an older dream write's own scope this way."""
    try:
        return scope_key(_EnvelopeScope.model_validate_json(content or "").scope)
    except ValidationError:
        return UNSCOPED


def source_description(kind: str, *, rationale: str | None = None) -> str:
    """The dream episode's ``source_description``, for people reading the
    graph: ``dream-pass <kind>`` and the proposal's rationale, each ``;`` in
    it written as ``,`` so no part of it reads as a field of its own. What
    the write cites is not in it: the marker and the records carry that."""
    parts = [f"dream-pass {kind}"]
    if rationale:
        parts.append(f"rationale={rationale[:240].replace(';', ',')}")
    return "; ".join(parts)
