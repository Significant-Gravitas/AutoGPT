"""Old first messages and CLI session files for the tests of
``legacy_first_turn_memory.py``, ``legacy_session_file.py``,
``first_turn_memory_backfill.py`` and the engines' restores."""

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
# The review's counterexample: a body the old renderer could never have
# written (a fact line with no validity stamp), which an earlier matcher
# stripped.
USER_AUTHORED_BLOCK = (
    "<memory_context>\n<temporal_context>\n<FACTS>\n"
    "  - user-authored bare fact with no renderer validity suffix\n"
    "</FACTS>\n</temporal_context>\n</memory_context>\n\n"
    "keep this request"
)
_STAMP = f"{NOW} — present"
_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
_FULLWIDTH = str.maketrans("0123456789", "０１２３４５６７８９")
_LINES = "line\n" * 101


def _fact_stamped(stamp: str) -> str:
    return f"<FACTS>\n  - Alice works on Atlas ({stamp})\n</FACTS>"


def _episode(body: str, created_at: str = str(NOW)) -> str:
    return f"<RECENT_EPISODES>\n  - [{created_at}] {body}\n</RECENT_EPISODES>"


# ``<temporal_context>`` bodies no renderer wrote, each one line, time or
# section away from one that did.
RENDERER_IMPOSSIBLE = {
    "bare-fact": "<FACTS>\n  - Alice works on Atlas\n</FACTS>",
    "date-only-stamp": _fact_stamped("2025-06-01 — present"),
    "words-for-a-time": _fact_stamped("valid: yesterday — present"),
    "no-such-retirement": _fact_stamped(f"forgotten {NOW}"),
    "one-bare-fact-of-two": f"<FACTS>\n  - Alice ({_STAMP})\n  - Bob leads Atlas\n</FACTS>",
    "bare-episode": "<RECENT_EPISODES>\n  - asked about Atlas\n</RECENT_EPISODES>",
    "date-only-episode": _episode("asked", "2025-06-01"),
    "episodes-before-facts": (
        f"{_episode('asked')}\n\n<FACTS>\n  - Alice ({_STAMP})\n</FACTS>"
    ),
    "no-item-marker": f"<FACTS>\nAlice works on Atlas ({_STAMP})\n</FACTS>",
    "empty-section": "<FACTS>\n</FACTS>",
    "unknown-section": f"<FACTS>\n  - Alice ({_STAMP})\n</FACTS>\n\n<NOTES>\n  - x\n</NOTES>",
    # Times: the renderer wrote ``str()`` of a datetime, in ASCII digits.
    "arabic-indic-digits": _fact_stamped(_STAMP.translate(_ARABIC_INDIC)),
    "fullwidth-digits": _fact_stamped(_STAMP.translate(_FULLWIDTH)),
    "no-such-month": _fact_stamped("2025-13-01 00:00:00+00:00 — present"),
    "no-such-day": _fact_stamped("2025-02-31 00:00:00+00:00 — present"),
    "no-such-hour": _fact_stamped("2025-06-01 24:00:00 — present"),
    "no-such-offset": _fact_stamped("2025-06-01 00:00:00+99:99 — present"),
    "no-such-end": _fact_stamped("unknown — 2025-06-31 00:00:00"),
    "no-such-retirement-time": _fact_stamped("superseded 2025-00-10 00:00:00"),
    "zero-offset-written-negative": _fact_stamped(
        "2025-06-01 00:00:00-00:00 — present"
    ),
    "zero-microseconds-written": _fact_stamped("2025-06-01 00:00:00.000000 — present"),
    "episode-arabic-indic-digits": _episode("asked", str(NOW).translate(_ARABIC_INDIC)),
    "episode-no-such-time": _episode("asked", "2025-99-99 99:99:99+99:99"),
    # Episodes: the renderer cut the body to 500 characters.
    "episode-past-the-cut": _episode("e" * 501),
    "episode-far-past-the-cut": _episode("e" * 100_000),
    "multiline-episode-past-the-cut": _episode(_LINES),
    # One renderer per block, and only this stack's renderer writes a body
    # past the cut: by neutralising tag starts, in every line.
    "two-renderers-stamps": (
        f"<FACTS>\n  - Alice ({_STAMP})\n  - Bob (valid: {_STAMP})\n</FACTS>"
    ),
    "neutralised-past-the-cut-beside-production-stamps": (
        f"<FACTS>\n  - Alice ({_STAMP})\n</FACTS>\n\n{_episode('<!b>' + 'e' * 497)}"
    ),
    "neutralised-past-the-cut-beside-a-tag-left-as-is": (
        f"<FACTS>\n  - <b>Alice</b> (valid: {_STAMP})\n</FACTS>\n\n"
        f"{_episode('<!b>' + 'e' * 497)}"
    ),
    "past-the-cut-with-a-tag-left-as-is": _episode("<b>" + "e" * 498),
    "past-the-cut-even-without-the-marks": _episode("<!b>" + "e" * 498),
}


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
    """A warm-context block as this stack's renderer writes it."""
    block = _format_context(
        [edge(fact) for fact in facts], [episode(body) for body in episodes]
    )
    assert block is not None
    return block


def master_warm(facts: tuple[str, ...] = (), episodes: tuple[str, ...] = ()) -> str:
    """A warm-context block as production wrote it: master's
    ``graphiti/context._format_context`` from #12720 on, with no ``valid:``
    label and no tag neutralising."""
    sections = []
    if facts:
        lines = "\n".join(f"  - {fact} ({NOW} — present)" for fact in facts)
        sections.append(f"<FACTS>\n{lines}\n</FACTS>")
    if episodes:
        lines = "\n".join(f"  - [{NOW}] {body}" for body in episodes)
        sections.append(f"<RECENT_EPISODES>\n{lines}\n</RECENT_EPISODES>")
    return "<temporal_context>\n" + "\n\n".join(sections) + "\n</temporal_context>"


def legacy_first_message(
    warm_block: str, *, skills: bool = True, rest: str = REST
) -> str:
    """A first message as ``inject_user_context(warm_ctx=...)`` stored it."""
    prefix = SKILLS_BLOCK if skills else ""
    return f"{prefix}<memory_context>\n{warm_block}\n</memory_context>\n\n{rest}"


def folded_query(first_message: str) -> str:
    """The first turn's query when a message the user sent meanwhile was
    folded in front of the stored first message."""
    return BUDGET_BLOCK + "one more thing while you start\n\n" + first_message


def history_query(first_message: str, current: str = "and now?") -> str:
    """A query rebuilt from the database, as a turn without ``--resume`` sent
    it: its history opens with the stored first message."""
    return (
        f"<conversation_history>\nUser: {first_message}\nYou responded: done\n"
        f"</conversation_history>\n\nNow, the user says:\n{BUDGET_BLOCK}{current}"
    )


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
