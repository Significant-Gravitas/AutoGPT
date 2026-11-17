"""graphiti's extraction seeds a new fact's attributes and never changes a
stored fact's.

graphiti-core 0.30.2 saves a fact (a ``RELATES_TO`` edge) with
``SET e = $edge_data``, or ``SET r = edge`` in ``add_episode``'s bulk save,
which removes every property its map does not hold. When its model merges a
statement into a stored fact, the map holds the attributes graphiti works
out for the fact afresh (``edge_operations.resolve_extracted_edge``): only
what its attribute model answered (``merge_mode='replace'``) for a fact
named after an edge type, and none for one with any other name. A restated
fact therefore lost everything the platform had written on it: the recall
stamps, the ratification stamp and the dream's envelope metadata. The
model's answer also set its ``status`` afresh, so a ratified fact could go
back to tentative.

Two rules keep a stored fact's attributes as they are:

- The FalkorDB driver runs graphiti's two fact saves as ``+=``
  (``merging_save``, ``EdgeMergingSession``): a property graphiti's map
  does not hold keeps its stored value. The map always holds graphiti's own
  fields (``fact``, ``name``, ``episodes``, ``expired_at``, ``invalid_at``
  and the rest), nulls included, and FalkorDB's ``+=`` removes a property
  set to null, so those are saved as before. Only the two queries graphiti
  generates are rewritten, matched by their exact text.
- graphiti's attribute prompt for a fact lists its stored attributes
  (``EXISTING ATTRIBUTES``), empty for a new fact. For a stored one
  (``stored_fact``), ``recall_ingest.ForgetAwareLLMClient`` answers no
  attributes without asking the model, so with ``+=`` the stored ones stay.
  A prompt whose attributes cannot be read counts as a stored fact's,
  logged at error once per reason: a new fact then gets no attributes (it
  reads as an unstamped fact, live), and a stored one never gets new values.

A user's own restatement still moves a tentative fact on: the worker
promotes it once the chat turn is written (``ingest._promote_restated``).
A fact saved with no attributes at all (before ``MemoryFact`` was wired)
reads as new, so a statement merged into it still seeds them.

A forgotten fact (one a forget stamped ``forgotten_at``) takes nothing from
either save but its episode list: each save keeps the stored properties and
embedding of an edge carrying the marker, in the same statement. graphiti
saves what it read when its ``add_episode`` began, and without the write
lock (``scope_lock.py``: Redis unreachable, or a lease lost) an ingestion
that read the fact before a forget saves it after. A merge would put the
sentence back under the marker, and graphiti's dedup, which knows a
forgotten fact only by its placeholder text (``recall_ingest.py``), would
then merge a later statement of it into the forgotten fact, where it stays
hidden. Nothing else ever has to be saved on a forgotten fact: graphiti's
model never merges a statement into one or invalidates one
(``recall_ingest.ForgetAwareLLMClient``), and the exact-text match that runs
before it only adds the episode of a statement repeating its ``[forgotten]``
text.
"""

import logging
import re
from typing import Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb_driver import FalkorDriverSession
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_save_bulk_query,
    get_entity_edge_save_query,
)
from graphiti_core.prompts.models import Message
from pydantic import TypeAdapter, ValidationError

logger = logging.getLogger(__name__)


def _save_clauses(edge: str, data: str) -> dict[str, str]:
    """The two clauses of a fact save writing map ``data`` on ``edge``, each
    with the clause the driver runs instead: a merge of the map, taking only
    ``episodes`` on an edge a forget marked, and the embedding, kept on one."""
    live = f"{edge}.forgotten_at IS NULL"
    return {
        f"SET {edge} = {data}": (
            f"SET {edge} += CASE WHEN {live} THEN {data}"
            f" ELSE {{episodes: {data}.episodes}} END"
        ),
        f"SET {edge}.fact_embedding = vecf32({data}.fact_embedding)": (
            f"SET {edge}.fact_embedding = CASE WHEN {live}"
            f" THEN vecf32({data}.fact_embedding) ELSE {edge}.fact_embedding END"
        ),
    }


# graphiti's fact saves on FalkorDB, each with its clauses that write the
# edge (graphiti's map replacing its properties, then its embedding) and
# what the driver runs in their place.
SAVE_CLAUSES: dict[str, dict[str, str]] = {
    get_entity_edge_save_query(GraphProvider.FALKORDB): _save_clauses(
        "e", "$edge_data"
    ),
    get_entity_edge_save_bulk_query(GraphProvider.FALKORDB): _save_clauses("r", "edge"),
}


def _rewritten(query: str, clauses: dict[str, str]) -> str:
    for clause, replacement in clauses.items():
        query = query.replace(clause, replacement)
    return query


# Each save as the driver runs it.
MERGING_SAVES: dict[str, str] = {
    query: _rewritten(query, clauses) for query, clauses in SAVE_CLAUSES.items()
}

_EXISTING = re.compile(
    r"<EXISTING ATTRIBUTES>\s*(.*?)\s*</EXISTING ATTRIBUTES>", re.DOTALL
)
_ATTRIBUTES = TypeAdapter(dict[str, Any])
_REPORTED: set[str] = set()


def merging_save(query: str) -> str:
    """``query``, as a merge when it is one of graphiti's fact saves
    (``MERGING_SAVES``)."""
    return MERGING_SAVES.get(query, query)


class EdgeMergingSession(FalkorDriverSession):
    """graphiti's FalkorDB session (``add_episode``'s bulk save runs in it),
    running graphiti's fact saves as merges."""

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        merges = [merge for save, merge in MERGING_SAVES.items() if query == save]
        return await super().run(merges[0] if merges else query, **kwargs)


def stored_fact(messages: list[Message]) -> bool:
    """Whether graphiti's attribute prompt for a fact is about a stored one:
    the ``EXISTING ATTRIBUTES`` it shows are not an empty object, or cannot
    be read."""
    text = "\n".join(message.content for message in messages)
    shown = _EXISTING.findall(text)
    if len(shown) != 1:
        _report(f"it shows {len(shown)} attribute lists, not one")
        return True
    try:
        existing = _ATTRIBUTES.validate_json(shown[0])
    except ValidationError:
        _report("its attributes are not a JSON object")
        return True
    return existing != {}


def _report(why: str) -> None:
    """Log, at error and once per process for each reason, an attribute
    prompt ``stored_fact`` cannot read."""
    if why in _REPORTED:
        return
    _REPORTED.add(why)
    logger.error(
        "graphiti's fact attribute prompt no longer reads as fact_attributes "
        f"expects: {why}. While it does not, no fact gets attributes from "
        "extraction"
    )
