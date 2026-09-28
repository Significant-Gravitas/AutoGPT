"""What a dream write rests on, checked against what its pass read.

Every consolidated fact and proposal must cite the facts and episodes it was
drawn from. apply keeps a citation only when it names a fact or an episode
the pass read (its input bundle), filed under the kind the bundle says it
is, in the model's order and once each; a write left citing nothing is
dropped before it is queued and counted in ``uncited_writes_dropped``. So a
made-up uuid never reaches the graph, and no dream write rests on nothing.

What survives goes to the ingestion worker, which checks it against the
forgets made since the pass read the graph (``graphiti/recall_citations.py``)
and then records it on the dream's episode and on the facts only dream
episodes state (``graphiti/recall_derivation.py``). A later forget of
anything it cites retracts those facts (``graphiti/recall_cascade.py``).
"""

import re
from collections.abc import Collection, Sequence

from backend.copilot.graphiti.recall_citations import Citations

# How many uuids of each kind the episode's free-text ``source_description``
# repeats, for people reading the graph. The complete lists are recorded on
# the episode and its facts; a dream episode written before them has only
# these (``graphiti/migrations/backfill_derivations.py`` reads them back).
DESCRIBED_CITATIONS = 5

_DESCRIBED = re.compile(r"(?:^|;)\s*(src_episodes|src_facts)=([^;]*)")


def validated_citations(
    fact_uuids: Sequence[str],
    episode_uuids: Sequence[str],
    *,
    known_facts: Collection[str],
    known_episodes: Collection[str],
) -> Citations | None:
    """The write's citations that name something its pass read, each under
    the kind the pass read it as, or ``None`` when none does."""
    cited = list(dict.fromkeys([*fact_uuids, *episode_uuids]))
    facts = [uuid for uuid in cited if uuid in known_facts]
    episodes = [uuid for uuid in cited if uuid in known_episodes]
    if not facts and not episodes:
        return None
    return Citations(fact_uuids=facts, episode_uuids=episodes)


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
