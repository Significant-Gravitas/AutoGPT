"""The Cypher a forget's cascade runs (``recall_cascade.py``).

A dream's record is ``derived_from_facts`` and ``derived_from_episodes`` on
its episode and on the facts only dream episodes state
(``recall_derivation.py``); ``$facts`` and ``$episodes`` are the frontier,
what the cascade last retracted and hid, and ``$seen`` everything it has
reached.
"""

from .recall import live_fact_predicate

_LIVE = live_fact_predicate("e")

# The facts an earlier try of this forget retracted.
EARLIER_QUERY = """
MATCH ()-[e:RELATES_TO]->()
WHERE e.expiration_reason IN $reasons
RETURN e.uuid AS uuid, e.expiration_reason AS reason
"""

# A fact derived from the frontier, not reached yet, that no user's episode
# states (every episode it names carries a dream's record), and whether it
# is live.
DERIVED_FACTS_QUERY = f"""
MATCH ()-[e:RELATES_TO]->()
WHERE NOT e.uuid IN $seen
  AND (any(x IN coalesce(e.derived_from_facts, []) WHERE x IN $facts)
       OR any(x IN coalesce(e.derived_from_episodes, []) WHERE x IN $episodes))
OPTIONAL MATCH (stated:Episodic)
WHERE stated.uuid IN coalesce(e.episodes, [])
  AND stated.derived_from_facts IS NULL
WITH e, count(stated) AS independent
WHERE independent = 0
RETURN e.uuid AS uuid,
       [x IN coalesce(e.derived_from_facts, []) WHERE x IN $facts]
       + [x IN coalesce(e.derived_from_episodes, []) WHERE x IN $episodes] AS via,
       ({_LIVE}) AS live
ORDER BY uuid
LIMIT $limit
"""

# A dream episode derived from the frontier, hidden or not, not yet reached.
DERIVED_EPISODES_QUERY = """
MATCH (ep:Episodic)
WHERE NOT ep.uuid IN $seen
  AND (any(x IN coalesce(ep.derived_from_facts, []) WHERE x IN $facts)
       OR any(x IN coalesce(ep.derived_from_episodes, []) WHERE x IN $episodes))
RETURN ep.uuid AS uuid,
       [x IN coalesce(ep.derived_from_facts, []) WHERE x IN $facts]
       + [x IN coalesce(ep.derived_from_episodes, []) WHERE x IN $episodes] AS via
ORDER BY uuid
LIMIT $limit
"""

# A soft forget's retraction (``recall_forget._RETRACT_EDGE_QUERY``) of each
# target still live, with its own reason.
RETRACT_QUERY = f"""
UNWIND $targets AS target
MATCH ()-[e:RELATES_TO {{uuid: target.uuid}}]->()
WHERE (e.group_id = $group_id OR e.group_id IS NULL) AND {_LIVE}
SET e.forgotten_at = coalesce(e.forgotten_at, $now),
    e.expired_at = coalesce(e.expired_at, $now),
    e.status = $status,
    e.expiration_reason = target.reason
RETURN e.uuid AS uuid
"""

REDACT_DERIVED_QUERY = """
MATCH (ep:Episodic)
WHERE ep.uuid IN $uuids
SET ep.redacted_at = coalesce(ep.redacted_at, $now)
WITH ep
OPTIONAL MATCH (ep)-[:MENTIONS]->(entity:Entity)
RETURN collect(DISTINCT entity.uuid) AS mentioned
"""
