"""What a dream episode written before derivation records cites, read back
from its ``source_description`` for the derivation backfill
(``backfill_derivations.py``), and checked before it is trusted.

Before records, the dream wrote a write's first five citations of each kind
into its description, after the model's free-text rationale and with the
same ``;`` delimiter, so a rationale could forge one (Codex's probe:
``rationale="ordinary; src_facts=<uuid>"``). A description is read only in
the shapes the dream wrote, its keys ending it: ``dream-pass consolidation;
src_episodes=<ids>``, ``dream-pass proposal[; rationale=<text>][;
src_facts=<ids>]``, and this branch's first form, which could end with both
keys. A description naming a ``src_`` key twice, or anywhere before those
final keys (a rationale quoting one), or listing an id with a space in it,
is ambiguous and attributes nothing. Each uuid read back must then name a
fact or an episode the graph has, a fact in the dream episode's own scope
(episode citations are not scoped: ``dream/citations.py``); the others are
dropped and counted. Unlike a new write, an older one citing a fact of
another scope cannot be dropped: it keeps its other citations, so it could
escape that fact's forget. A shape a forged rationale could still take (it
ends the description with one key, naming a real source in the right
scope) cannot be told from a real one; new writes never put citations or a
raw ``;`` in their description.
"""

import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

from backend.copilot.dream.citations import scope_key
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver

from .backfill_pages import BATCH_SIZE, rows

_SUFFIX = re.compile(
    r"(?:;\s*src_episodes=(?P<episodes>[^;]*))?"
    r"(?:;\s*src_facts=(?P<facts>[^;]*))?\s*$"
)
_KEYS = ("src_facts=", "src_episodes=")


class LegacyCitations(BaseModel):
    """What one description lists: ``facts`` and ``episodes``, or nothing
    and ``ambiguous`` when its shape is not one the dream wrote."""

    facts: list[str] = Field(default_factory=list)
    episodes: list[str] = Field(default_factory=list)
    ambiguous: bool = False


def described_citations(description: str | None) -> LegacyCitations:
    """The citations a pre-record dream description lists at its end."""
    text = description or ""
    if any(text.count(key) > 1 for key in _KEYS):
        return LegacyCitations(ambiguous=True)
    found = _SUFFIX.search(text)  # always, possibly empty, at the end
    if found is None:
        return LegacyCitations()
    if "src_" in text[: found.start()]:
        return LegacyCitations(ambiguous=True)
    facts = _ids(found.group("facts"))
    episodes = _ids(found.group("episodes"))
    if facts is None or episodes is None:
        return LegacyCitations(ambiguous=True)
    return LegacyCitations(facts=facts, episodes=episodes)


def _ids(listed: str | None) -> list[str] | None:
    """The ids a key lists, or None when one of them has a space in it."""
    ids = [part.strip() for part in (listed or "").split(",") if part.strip()]
    return None if any(re.search(r"\s", uuid) for uuid in ids) else ids


class Checked(BaseModel):
    """The citations of the episodes read back that name a source the graph
    has (a fact in the episode's own scope), by episode uuid; how many it
    ``rejected``."""

    cited: dict[str, LegacyCitations] = Field(default_factory=dict)
    rejected: int = 0


async def checked(
    driver: AutoGPTFalkorDriver,
    described: dict[str, LegacyCitations],
    scopes: dict[str, str],
) -> Checked:
    """Keep each citation in ``described`` (by dream episode) whose source
    the graph has: a fact in the scope ``scopes`` gives its dream episode,
    an episode of any scope."""
    cited_facts = [uuid for c in described.values() for uuid in c.facts]
    cited_episodes = [uuid for c in described.values() for uuid in c.episodes]
    facts = await _found(driver, FACT_SCOPES_QUERY, cited_facts)
    episodes = await _found(driver, CITED_EPISODES_QUERY, cited_episodes)
    done = Checked()
    for uuid, citations in described.items():
        wanted = scope_key(scopes.get(uuid))
        kept = LegacyCitations(
            facts=[f for f in citations.facts if facts.get(f) == wanted],
            episodes=[e for e in citations.episodes if e in episodes],
        )
        done.rejected += len(citations.facts) + len(citations.episodes)
        done.rejected -= len(kept.facts) + len(kept.episodes)
        done.cited[uuid] = kept
    return done


async def _found(
    driver: AutoGPTFalkorDriver, query: str, uuids: Iterable[str]
) -> dict[str, str]:
    """Each of ``uuids`` the graph has, with the scope its row gives (a
    fact's; ``UNSCOPED`` for a row that gives none)."""
    listed = list(dict.fromkeys(uuids))
    found: dict[str, str] = {}
    for start in range(0, len(listed), BATCH_SIZE):
        result = await driver.execute_query(
            query, uuids=listed[start : start + BATCH_SIZE]
        )
        found |= {row["uuid"]: scope_key(row.get("scope")) for row in rows(result)}
    return found


FACT_SCOPES_QUERY = """
MATCH ()-[e:RELATES_TO]->()
WHERE e.uuid IN $uuids
RETURN e.uuid AS uuid, e.scope AS scope
"""

CITED_EPISODES_QUERY = """
MATCH (ep:Episodic)
WHERE ep.uuid IN $uuids
RETURN ep.uuid AS uuid
"""
