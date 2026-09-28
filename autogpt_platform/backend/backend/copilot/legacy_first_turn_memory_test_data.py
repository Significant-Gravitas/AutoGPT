"""Old first messages and CLI session files for the tests of
``legacy_first_turn_memory.py``, ``first_turn_memory_backfill.py`` and the
engines' restores."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode

from backend.copilot.graphiti.context import _format_context

NOW = datetime(2025, 6, 1, tzinfo=timezone.utc)
SKILLS = (
    "Skills are reusable procedures.\n"
    "- name: deploy — Ship the app — triggers: deploy, ship"
)
SKILLS_BLOCK = f"<available_skills>\n{SKILLS}\n</available_skills>\n\n"
REST = (
    "<session_context>\nsession_id: s-1; pending_followups: 0\n"
    "</session_context>\n\n"
    "<env_context>\nworking_dir: /tmp/copilot-s-1\n</env_context>\n\n"
    "what is Alice working on"
)
# The query-only blocks the SDK engine put in front of the first message when
# it sent it, as a CLI session file recorded them.
BUDGET_BLOCK = (
    "<budget_status>\nTask budget: $0.10 spent of $5.00, 1 of 8 agents.\n"
    "</budget_status>\n\n"
)
BUILDER_BLOCK = (
    '<builder_context>\n<graph id="g-1" version="3" node_count="1" '
    'edge_count="0"/>\n<nodes>\n- n1: AgentInputBlock\n</nodes>\n'
    "</builder_context>\n\n"
)


def edge(fact: str) -> EntityEdge:
    return EntityEdge(
        uuid=str(uuid4()),
        group_id="user_abc",
        source_node_uuid="alice",
        target_node_uuid="atlas",
        created_at=NOW,
        name="works_on",
        fact=fact,
        valid_at=NOW,
        attributes={"status": "active"},
    )


def episode(content: str) -> EpisodicNode:
    return EpisodicNode(
        name="ep",
        group_id="user_abc",
        source=EpisodeType.text,
        source_description="chat",
        content=content,
        created_at=NOW,
        valid_at=NOW,
    )


def warm(facts: tuple[str, ...] = (), episodes: tuple[str, ...] = ()) -> str:
    """A warm-context block as graphiti renders it."""
    block = _format_context(
        [edge(fact) for fact in facts], [episode(body) for body in episodes]
    )
    assert block is not None
    return block


def legacy_first_message(
    warm_block: str, *, skills: bool = True, rest: str = REST
) -> str:
    """A first message as ``inject_user_context(warm_ctx=...)`` stored it."""
    prefix = SKILLS_BLOCK if skills else ""
    return f"{prefix}<memory_context>\n{warm_block}\n</memory_context>\n\n{rest}"


def built_to_backtrack(pairs: int) -> str:
    """A first query that opens with the engine's budget block and holds no
    memory block, whose user text repeats a ``<budget_status>`` closing and
    opening ``pairs`` times: it splits into leading blocks in 2**pairs ways,
    and a pattern that tried them all would take that long to give up."""
    pair = "\n</budget_status>\n\n<budget_status>\n"
    return BUDGET_BLOCK + REST + pair * pairs + "."


def session_file(*entries: tuple[str, str | list[dict]]) -> bytes:
    """A CLI session file: one JSONL entry per ``(role, content)``, chained
    by ``parentUuid`` the way the CLI writes them."""
    lines: list[str] = []
    parent: str | None = None
    for role, content in entries:
        uid = str(uuid4())
        if role == "assistant":
            message: dict = {
                "role": "assistant",
                "model": "claude-test",
                "id": f"msg_{uid[:8]}",
                "type": "message",
                "content": [{"type": "text", "text": content}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
            }
        else:
            message = {"role": "user", "content": content}
        entry = {"type": role, "uuid": uid, "parentUuid": parent, "message": message}
        lines.append(json.dumps(entry, separators=(",", ":")))
        parent = uid
    return ("\n".join(lines) + "\n").encode()


def bucket_storage(content: bytes) -> MagicMock:
    """Bucket storage holding one uploaded CLI session file, covering two
    messages, for the real ``download_transcript`` to restore."""
    meta = json.dumps({"message_count": 2, "mode": "sdk"}).encode()
    storage = MagicMock()
    storage.retrieve = AsyncMock(
        side_effect=lambda path: content if path.endswith(".jsonl") else meta
    )
    return storage


ALICE = warm(("Alice works on Atlas",))
