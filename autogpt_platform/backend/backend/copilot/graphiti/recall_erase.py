"""A hard forget erases the sentences the dream derived from what it forgot.

A soft forget keeps every sentence it hides for audit: a fact's in its
``fact_redacted`` / ``name_redacted`` copies (``recall_hide.py``), a dream
episode's in its body. A hard forget asks for the text itself to go. It
deletes the facts the user named and empties the episodes only they kept
(``recall_orphans.py``); what the dream derived from them its cascade
retracts softly, as always (``recall_cascade.py``), keeping the edges, their
markers, reasons and records, so the walk and a retry still find them, and
erases here:

- every derived fact the walk reached, retracted or walked through: its
  ``fact`` and ``name`` read ``recall.FORGOTTEN_FACT``, as the scrub leaves
  them, both audit copies are blank, and its ``fact_embedding``, the vector
  graphiti computed from the sentence, is removed. FalkorDB's vector
  similarity reads a missing embedding as no score, so graphiti's searches
  and its dedup simply pass the edge by. Nothing else on the edge holds the
  sentence: its other properties are uuids, times and ``MemoryFact`` fields;
- every dream episode it hid: its body, and the rationale and citations its
  ``source_description`` listed after the kind (``dream-pass proposal``).
  Only an episode carrying a dream's record (``recall_derivation.py``) is the
  dream's, so a user's episode the cascade hides keeps its text.

Each write is idempotent, so a hard forget repeated after a failure erases
what the first one left.
"""

from graphiti_core.driver.driver import GraphDriver

from .recall import FORGOTTEN_FACT


async def erase(driver: GraphDriver, facts: list[str], episodes: list[str]) -> None:
    """Blank ``facts``' text and audit copies (their entities already
    scrubbed, ``recall_hide.scrub``), then the dream episodes among
    ``episodes``."""
    if facts:
        await driver.execute_query(
            ERASE_FACTS_QUERY, uuids=facts, placeholder=FORGOTTEN_FACT
        )
    if episodes:
        await driver.execute_query(ERASE_DREAM_EPISODES_QUERY, uuids=episodes)


ERASE_FACTS_QUERY = """
MATCH ()-[e:RELATES_TO]->()
WHERE e.uuid IN $uuids
SET e.fact = $placeholder,
    e.name = $placeholder,
    e.fact_redacted = '',
    e.name_redacted = '',
    e.fact_embedding = NULL
"""

# The description keeps what precedes its first ``;``: the kind of write.
ERASE_DREAM_EPISODES_QUERY = """
MATCH (ep:Episodic)
WHERE ep.uuid IN $uuids AND ep.derived_from_facts IS NOT NULL
SET ep.content = '',
    ep.source_description = split(ep.source_description, ';')[0]
"""
