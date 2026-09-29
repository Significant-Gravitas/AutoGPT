"""What a dream write rests on, checked against what its pass read.

Every consolidated fact and proposal must cite the facts and episodes it was
drawn from. apply keeps a citation only when it names a fact or an episode
the pass read (its input bundle), filed under the kind the bundle says it
is, in the model's order and once each, and only when that source is in the
write's own scope: the prompts' rule ("group by scope", "proposed findings
live in the same scope as their evidence"), enforced here. A fact's scope is
its own, an episode's the one its ``MemoryEnvelope`` declares, and a source
naming none (a chat turn, a fact from before scopes) is ``UNSCOPED``, the
default everywhere. A citation to another scope is dropped and counted in
``cross_scope_citations_dropped``; a write left citing nothing is dropped
before it is queued and counted in ``uncited_writes_dropped``. So a made-up
uuid never reaches the graph, a write never rests on a source outside its
scope, and no dream write rests on nothing. A source in the right scope that
the write does not truly rest on is not caught: no deterministic check can
tell.

What survives goes to the ingestion worker, which checks it against the
forgets made since the pass read the graph (``graphiti/recall_citations.py``)
and then records it on the dream's episode and on the facts only dream
episodes state (``graphiti/recall_derivation.py``). A later forget of
anything it cites retracts those facts (``graphiti/recall_cascade.py``).
"""

import re
from collections.abc import Collection, Mapping, Sequence

from pydantic import BaseModel, ValidationError

from backend.copilot.graphiti.recall_citations import Citations

from .fetch import DreamInput

# The scope of a fact or episode that names none: ``MemoryFact``'s and
# ``MemoryEnvelope``'s default, and how the prompts list an unscoped fact.
UNSCOPED = "real:global"

# How many uuids of each kind the episode's free-text ``source_description``
# repeats, for people reading the graph. The complete lists are recorded on
# the episode and its facts; a dream episode written before them has only
# these (``graphiti/migrations/backfill_derivations.py`` reads them back).
DESCRIBED_CITATIONS = 5

_DESCRIBED = re.compile(r"(?:^|;)\s*(src_episodes|src_facts)=([^;]*)")


class CheckedCitations(BaseModel):
    """What apply keeps of a write's citations (``None``: nothing the pass
    read in the write's scope), and how many it dropped for naming a source
    in another scope."""

    citations: Citations | None = None
    cross_scope: int = 0


def validated_citations(
    fact_uuids: Sequence[str],
    episode_uuids: Sequence[str],
    *,
    scope: str,
    known_facts: Collection[str],
    known_episodes: Collection[str],
    source_scopes: Mapping[str, str],
) -> CheckedCitations:
    """The write's citations that name something its pass read in the
    write's ``scope``, each under the kind the pass read it as;
    ``source_scopes`` gives each source's scope, ``UNSCOPED`` when absent."""
    cited = list(dict.fromkeys([*fact_uuids, *episode_uuids]))
    known = [uuid for uuid in cited if uuid in known_facts or uuid in known_episodes]
    wanted = scope_key(scope)
    kept = [uuid for uuid in known if scope_key(source_scopes.get(uuid)) == wanted]
    facts = [uuid for uuid in kept if uuid in known_facts]
    episodes = [uuid for uuid in kept if uuid in known_episodes]
    cross_scope = len(known) - len(kept)
    if not facts and not episodes:
        return CheckedCitations(cross_scope=cross_scope)
    citations = Citations(fact_uuids=facts, episode_uuids=episodes)
    return CheckedCitations(citations=citations, cross_scope=cross_scope)


def scope_key(scope: str | None) -> str:
    """``scope`` as the prompts list it: whitespace collapsed, ``UNSCOPED``
    when it names none."""
    return " ".join((scope or "").split()) or UNSCOPED


def source_scopes(input_bundle: DreamInput) -> dict[str, str]:
    """The scope of every fact and episode a pass read: a fact's own, an
    episode's envelope's, else ``UNSCOPED``."""
    scopes = {fact.uuid: scope_key(fact.scope) for fact in input_bundle.facts}
    for episode in input_bundle.episodes:
        scopes[episode.uuid] = _envelope_scope(episode.content)
    return scopes


class _EnvelopeScope(BaseModel):
    scope: str | None = None


def _envelope_scope(content: str | None) -> str:
    """The scope a ``MemoryEnvelope`` body declares; ``UNSCOPED`` for any
    other episode (a chat turn is plain text)."""
    try:
        return scope_key(_EnvelopeScope.model_validate_json(content or "").scope)
    except ValidationError:
        return UNSCOPED


def source_description(
    kind: str, citations: Citations, *, rationale: str | None = None
) -> str:
    """The dream episode's ``source_description``: ``dream-pass <kind>``, the
    proposal's rationale, and the first few uuids of each kind it cites."""
    parts = [f"dream-pass {kind}"]
    if rationale:
        parts.append(f"rationale={rationale[:240]}")
    for label, uuids in (
        ("src_episodes", citations.episode_uuids),
        ("src_facts", citations.fact_uuids),
    ):
        if uuids:
            parts.append(f"{label}={','.join(uuids[:DESCRIBED_CITATIONS])}")
    return "; ".join(parts)


def described_citations(description: str | None) -> tuple[list[str], list[str]]:
    """``(facts, episodes)`` a dream episode's ``source_description`` lists,
    as ``source_description`` writes it and older dream writes did (a
    consolidation listed episodes only, a proposal facts only); a key given
    twice (a rationale quoting one) counts its last value."""
    found: dict[str, list[str]] = {}
    for key, value in _DESCRIBED.findall(description or ""):
        found[key] = [uuid for uuid in value.strip().split(",") if uuid]
    return found.get("src_facts", []), found.get("src_episodes", [])
