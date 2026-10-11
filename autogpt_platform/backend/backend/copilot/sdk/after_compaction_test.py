"""The context re-sent after a compaction never exceeds the CLI's hook cap."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.experts.models import (
    PROTECTED_SOUL_RULES,
    Expert,
    ExpertRoutine,
    ExpertWorkflowRef,
)
from backend.copilot.expert_context import (
    _routines_block,
    render_expert_workflows_block,
    render_team_context,
)

from .after_compaction import (
    AFTER_COMPACTION_MAX_PARTS,
    AFTER_COMPACTION_PART_BUDGET,
    ContextBlock,
    fit_after_compaction,
    shorten_block,
    split_blocks,
    utf16_len,
)
from .service import _AFTER_COMPACTION_LEAD, _context_after_compaction

_LEAD = "The conversation was just compacted; here are the blocks again:\n\n"
# The CLI 2.1.284 threshold: longer than this and the model gets a preview.
_CLI_HOOK_CAP = 10_000


def _block(tag: str, size: int, *, item: str = "x") -> str:
    """A ``<tag>`` block of exactly ``size`` characters, trailing blank line
    included, holding one line of ``item`` repeated."""
    frame = f"<{tag}>\n\n</{tag}>\n\n"
    return f"<{tag}>\n{item * (size - len(frame))}\n</{tag}>\n\n"


def _skills_block(n_entries: int) -> str:
    entries = "\n".join(
        f"- name: skill-{i:03d} — Does thing number {i} for the user — "
        f"triggers: thing {i}, chore {i}"
        for i in range(n_entries)
    )
    return (
        "<available_skills>\n"
        "Skills are reusable procedures loaded with `tool:read_skill`.\n"
        f"{entries}\n"
        "</available_skills>\n\n"
    )


def _prefix(*blocks: str) -> str:
    return "".join(blocks)


def _expert(index: int, *, workflows: int = 1) -> Expert:
    return Expert(
        id=f"exp-{index}",
        name=f"Expert {index}",
        avatar_url=None,
        role="Specialist",
        tagline=None,
        bio=None,
        skills=[],
        identity="You are a specialist.",
        voice_preferences="Direct.",
        boundaries="Ask first.",
        protected_soul_rules=list(PROTECTED_SOUL_RULES),
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=[
            ExpertWorkflowRef(
                id=f"wf-{index}-{w}",
                store_listing_version_id="slv-1",
                library_agent_id=f"la-{w}",
                graph_id=f"graph-{w}",
                name=f"Workflow {w}",
                description=f"First line of {w}\nsecond line of {w}",
            )
            for w in range(workflows)
        ],
    )


def _routine(index: int, source: str) -> ExpertRoutine:
    return ExpertRoutine(
        id=f"routine-{index}",
        expert_id=None,
        title=f"Routine {index}\nwith a second line",
        prompt="Read the week ahead.",
        crons=["0 8 * * 1"],
        source=source,
        enabled=False,
    )


async def _routines(source: str) -> str:
    db = MagicMock()
    db.list_routines = AsyncMock(return_value=[_routine(i, source) for i in range(40)])
    with patch("backend.copilot.expert_context.experts_db", return_value=db):
        return await _routines_block("user-1", None)


class TestSplitBlocks:
    def test_blocks_come_back_in_order_with_their_text(self):
        prefix = _prefix(
            _skills_block(2),
            _block("team_context", 300),
            _block("session_context", 120),
            _block("user_context", 200),
        )

        blocks = split_blocks(prefix)

        assert [b.tag for b in blocks] == [
            "available_skills",
            "team_context",
            "session_context",
            "user_context",
        ]
        assert "".join(b.text for b in blocks) == prefix

    def test_text_that_is_not_a_block_is_kept_whole(self):
        prefix = _block("env_context", 80) + "loose text <not a block"

        blocks = split_blocks(prefix)

        assert [b.tag for b in blocks] == ["env_context", ""]
        assert blocks[1].text == "loose text <not a block"


class TestFitUnderBudget:
    def test_small_context_is_one_unchanged_part(self):
        prefix = _prefix(_skills_block(3), _block("user_context", 400))

        fitted = fit_after_compaction(_LEAD, prefix)

        assert fitted.parts == [_LEAD + prefix]
        assert fitted.dropped == [] and fitted.shortened == []
        assert fitted.delivered_chars == fitted.rebuilt_chars

    def test_nothing_rebuilt_sends_nothing(self):
        assert fit_after_compaction(_LEAD, "").parts == []


class TestFitOverBudget:
    def test_splits_between_blocks_and_keeps_order(self):
        blocks = [
            _block("available_skills", 6_000),
            _block("team_context", 5_000),
            _block("session_context", 4_000),
            _block("user_context", 3_000),
        ]

        fitted = fit_after_compaction(_LEAD, _prefix(*blocks))

        assert len(fitted.parts) == 3
        assert fitted.parts[0] == _LEAD + blocks[0]
        assert fitted.parts[1] == blocks[1] + blocks[2]
        assert fitted.parts[2] == blocks[3]
        assert all(utf16_len(p) <= AFTER_COMPACTION_PART_BUDGET for p in fitted.parts)
        assert fitted.dropped == [] and fitted.shortened == []

    def test_prod_maximum_fits_in_the_parts_without_cuts(self):
        # The largest first-turn context seen in prod (22,762 characters)
        # in its largest blocks.
        blocks = [
            _block("available_skills", 9_000),
            _block("expert_workflows", 6_042),
            _block("team_context", 7_462),
            _block("session_context", 109),
            _block("user_context", 149),
        ]

        fitted = fit_after_compaction(_LEAD, _prefix(*blocks))

        assert len(fitted.parts) <= AFTER_COMPACTION_MAX_PARTS
        assert "".join(fitted.parts) == _LEAD + _prefix(*blocks)

    def test_counts_utf16_units_like_the_cli(self):
        # A non-BMP character is one Python character and two UTF-16 units.
        emoji_block = _block("user_context", 6_000, item="\U0001f600")
        other = _block("session_context", 4_000)

        fitted = fit_after_compaction(_LEAD, emoji_block + other)

        assert utf16_len(emoji_block) > 11_000
        assert all(utf16_len(p) <= _CLI_HOOK_CAP for p in fitted.parts)

    def test_drops_lowest_priority_blocks_when_parts_run_out(self):
        blocks = {
            "standing_work": _block("standing_work", 9_000),
            "available_skills": _block("available_skills", 9_000),
            "team_context": _block("team_context", 9_000),
            "routines": _block("routines", 9_000),
            "session_context": _block("session_context", 9_000),
            "user_context": _block("user_context", 9_000),
        }

        fitted = fit_after_compaction(_LEAD, _prefix(*blocks.values()))

        assert len(fitted.parts) == AFTER_COMPACTION_MAX_PARTS
        assert fitted.dropped == ["standing_work", "team_context"]
        kept = "".join(fitted.parts)
        for tag in ("available_skills", "routines", "session_context", "user_context"):
            assert blocks[tag] in kept
        # Survivors keep their original order.
        assert kept.index("<available_skills>") < kept.index("<routines>")
        assert kept.index("<routines>") < kept.index("<session_context>")
        assert kept.index("<session_context>") < kept.index("<user_context>")

    def test_unknown_blocks_go_before_known_ones(self):
        blocks = [
            _block("session_context", 9_000),
            _block("future_block", 9_000),
            _block("standing_work", 9_000),
        ]

        fitted = fit_after_compaction(_LEAD, _prefix(*blocks), max_parts=2)

        assert fitted.dropped == ["future_block"]


class TestOversizeBlock:
    def test_skills_index_is_shortened_by_whole_entries(self):
        skills = _skills_block(150)
        assert utf16_len(skills) > _CLI_HOOK_CAP
        user = _block("user_context", 500)

        fitted = fit_after_compaction(_LEAD, skills + user)

        assert fitted.shortened == ["available_skills"]
        assert fitted.dropped == []
        assert all(utf16_len(p) <= AFTER_COMPACTION_PART_BUDGET for p in fitted.parts)
        sent = "".join(fitted.parts)
        assert user in sent
        (sent_skills,) = [
            b for b in split_blocks(sent[len(_LEAD) :]) if b.tag == "available_skills"
        ]
        lines = sent_skills.text.split("\n")
        assert lines[1].startswith("Skills are reusable procedures")
        entries = [line for line in lines if line.startswith("- name: ")]
        # Every kept entry is whole and in its original order.
        for i, entry in enumerate(entries):
            assert entry.startswith(f"- name: skill-{i:03d} ")
            assert entry.endswith(f"chore {i}")
        closing = lines[len(entries) + 2]
        assert closing == (
            f"{150 - len(entries)} more skills are not listed here: call "
            "`tool:list_skills` for the full index."
        )
        assert sent_skills.text.endswith("</available_skills>\n\n")

    def test_block_without_items_is_dropped(self):
        big = _block("user_context", 12_000)
        session = _block("session_context", 200)

        fitted = fit_after_compaction(_LEAD, big + session)

        assert fitted.dropped == ["user_context"]
        assert fitted.parts == [_LEAD + session]

    def test_shorten_keeps_head_and_tail_lines(self):
        block = ContextBlock(
            tag="team_context",
            text=(
                "<team_context>\nThe user has hired these experts:\n"
                + "\n".join(f"- Expert {i} — role (expert id: e{i})" for i in range(50))
                + "\nWhen a request clearly matches an expert's domain, say so."
                + "\n</team_context>\n\n"
            ),
        )

        short = shorten_block(block, 600)

        assert short is not None
        assert utf16_len(short.text) <= 600
        lines = short.text.split("\n")
        assert lines[1] == "The user has hired these experts:"
        assert lines[-4] == "When a request clearly matches an expert's domain, say so."
        assert lines[-5].endswith("call `tool:list_team` for the full roster.")

    def test_a_dropped_last_item_takes_its_other_lines_with_it(self):
        entries = "\n".join(
            f"- name: skill-{i} — Does thing {i}.\nSecond line of {i}."
            for i in range(6)
        )
        block = ContextBlock(
            tag="available_skills",
            text=f"<available_skills>\n{entries}\n</available_skills>\n\n",
        )

        short = shorten_block(block, 230)

        assert short is not None
        assert short.text == (
            "<available_skills>\n"
            "- name: skill-0 — Does thing 0.\nSecond line of 0.\n"
            "- name: skill-1 — Does thing 1.\nSecond line of 1.\n"
            "4 more skills are not listed here: call `tool:list_skills` for "
            "the full index.\n"
            "</available_skills>\n\n"
        )

    def test_hiring_roster_is_not_shortened(self):
        # Its items are templates to hire, which `list_team` does not list.
        hiring = (
            "<team_context>\nThe user has not hired any experts yet. You are "
            "their Head of AI.\nRoster:\n"
            + "\n".join(f"- Template {i} (template_id: t{i})" for i in range(400))
            + "\n</team_context>\n\n"
        )
        session = _block("session_context", 200)

        fitted = fit_after_compaction(_LEAD, hiring + session)

        assert fitted.dropped == ["team_context"]
        assert fitted.shortened == []
        assert fitted.parts == [_LEAD + session]

    def test_not_shortened_when_the_turn_hides_the_listing_tool(self):
        skills = _skills_block(150)
        session = _block("session_context", 200)

        fitted = fit_after_compaction(
            _LEAD, skills + session, hidden_tools={"list_skills"}
        )

        assert fitted.dropped == ["available_skills"]
        assert fitted.shortened == []
        assert fitted.parts == [_LEAD + session]


class TestRenderedBlocksShorten:
    """The real renderers' blocks lose whole items and keep their last line,
    even when an item's text runs over several lines."""

    @staticmethod
    def _shorten(rendered: str) -> list[str]:
        (block,) = split_blocks(rendered)
        short = shorten_block(block, utf16_len(block.text) // 2)
        assert short is not None
        return short.text.split("\n")

    @pytest.mark.parametrize("delegation_enabled", [True, False])
    @pytest.mark.parametrize("exclude_expert_id", [None, "exp-0"])
    def test_team_roster(self, delegation_enabled, exclude_expert_id):
        experts = [_expert(i) for i in range(40)]
        rendered = render_team_context(
            experts,
            delegation_enabled=delegation_enabled,
            exclude_expert_id=exclude_expert_id,
        )

        lines = self._shorten(rendered)

        assert rendered.split("\n")[-4] == lines[-4]
        assert lines[-5].endswith("call `tool:list_team` for the full roster.")
        assert lines[-6].startswith("- Expert ")

    def test_expert_workflows(self):
        rendered = render_expert_workflows_block(_expert(0, workflows=40))

        lines = self._shorten(rendered)

        assert lines[-4].startswith("If the user skips or declines")
        assert rendered.split("\n")[-4] == lines[-4]
        assert lines[-5].endswith("call `tool:list_expert_workflows` for all of them.")
        # The last kept workflow is whole: both lines of its description.
        assert lines[-6].startswith("second line of ")
        assert lines[-7].startswith("- Workflow ")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("source", ["TEMPLATE", "OWNER"])
    async def test_routines(self, source):
        rendered = await _routines(source)

        lines = self._shorten(rendered)

        closing = -5 if source == "TEMPLATE" else -4
        assert lines[closing].endswith("call `tool:list_routines` for all of them.")
        assert lines[closing - 1].startswith("with a second line (id: routine-")
        if source == "TEMPLATE":
            assert lines[-4].startswith("The routines marked (proposal)")


def test_a_block_of_exactly_the_budget_goes_out_whole():
    assert AFTER_COMPACTION_PART_BUDGET == 9_500 < _CLI_HOOK_CAP
    block = _block("session_context", 9_500)

    fitted = fit_after_compaction("", block)

    assert fitted.parts == [block]
    assert fitted.dropped == []


def test_a_block_one_over_the_budget_that_cannot_be_shortened_is_dropped():
    fitted = fit_after_compaction("", _block("session_context", 9_501))

    assert fitted.parts == []
    assert fitted.dropped == ["session_context"]


class TestContextAfterCompaction:
    """``_context_after_compaction`` hands the hook parts and says when it cut."""

    async def _parts(
        self, prefix: str, hidden_tools: frozenset[str] = frozenset()
    ) -> list[str]:
        with patch(
            "backend.copilot.sdk.service._rebuild_first_turn_prefix",
            AsyncMock(return_value=prefix),
        ):
            return await _context_after_compaction(
                MagicMock(),
                "u1",
                sdk_cwd="/tmp/copilot-x",
                use_e2b=False,
                log_prefix="[test]",
                hidden_tools=hidden_tools,
            )

    @pytest.mark.asyncio
    async def test_fitting_context_goes_out_whole(self, caplog):
        prefix = _prefix(_skills_block(3), _block("user_context", 400))

        with caplog.at_level(logging.WARNING):
            parts = await self._parts(prefix)

        assert parts == [_AFTER_COMPACTION_LEAD + prefix]
        assert not caplog.records

    @pytest.mark.asyncio
    async def test_a_cut_is_logged_with_its_sizes(self, caplog):
        prefix = _prefix(
            _skills_block(150),
            *(_block("team_context", 9_000) for _ in range(4)),
            _block("user_context", 300),
        )

        with caplog.at_level(logging.WARNING):
            parts = await self._parts(prefix)

        assert 1 <= len(parts) <= AFTER_COMPACTION_MAX_PARTS
        assert all(utf16_len(p) <= AFTER_COMPACTION_PART_BUDGET for p in parts)
        assert parts[0].startswith(_AFTER_COMPACTION_LEAD)
        (record,) = caplog.records
        assert record.levelno == logging.WARNING
        message = record.getMessage()
        delivered = sum(len(p) for p in parts)
        rebuilt = len(_AFTER_COMPACTION_LEAD) + len(prefix)
        assert f"{delivered} of {rebuilt} chars in {len(parts)} parts" in message
        assert "dropped=['team_context'" in message
        assert "shortened=['available_skills']" in message

    @pytest.mark.asyncio
    async def test_a_hidden_listing_tool_is_not_pointed_at(self, caplog):
        prefix = _prefix(_skills_block(150), _block("user_context", 300))

        with caplog.at_level(logging.WARNING):
            parts = await self._parts(prefix, frozenset({"list_skills"}))

        assert parts == [_AFTER_COMPACTION_LEAD + _block("user_context", 300)]
        (record,) = caplog.records
        assert "dropped=['available_skills']" in record.getMessage()

    @pytest.mark.asyncio
    async def test_nothing_rebuilt_sends_nothing(self):
        assert await self._parts("") == []
