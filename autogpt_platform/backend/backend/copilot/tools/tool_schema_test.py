"""Schema regression tests for all registered CoPilot tools.

Validates that every tool in TOOL_REGISTRY produces a well-formed schema:
- description is non-empty
- all `required` fields exist in `properties`
- every property has a `type` and `description`
- total schema character budget does not regress past threshold
"""

import json
from typing import Any, cast

import pytest

from backend.copilot.tools import TOOL_REGISTRY

from ._test_data import make_session

# Character budget (~4 chars/token heuristic, targeting ~8000 tokens).
# Bumped 32000 -> 32500 on PR #12699 to fit two pieces of load-bearing
# guidance: the wait_for_result dispatch-mode docs on run_agent
# (tells the LLM when to block vs fire-and-forget, and what each
# response shape carries) and the dry_run description. Keeps the
# regression gate effective while accepting a deliberate ~120-token
# spend on LLM-decision-critical copy.
# Bumped 32500 -> 32800 on PR #12871 for the new web_search tool
# (server-side Anthropic beta). Description already trimmed to the
# minimum viable copy; the bump absorbs the schema skeleton cost
# (~300 chars / ~75 tokens) for a new LLM-facing primitive.
# Bumped 32800 -> 33200 on PR #12873 for the web_search Perplexity
# Sonar refactor — adds a load-bearing `deep` boolean with explicit
# "~100x more expensive" cost warning the model must see to avoid
# accidentally triggering sonar-reasoning on ordinary lookups, plus
# synthesised-answer wording in the top-level description so the LLM
# reads the answer before reaching for `web_fetch`. Both are
# LLM-decision-critical copy, not bloat.
# Bumped 33200 -> 34000 when baseline gained the MCP `TodoWrite` tool
# for parity with the Claude Code SDK's built-in (PR #12879). The new
# schema adds ~600 chars; description already trimmed to the minimum
# viable copy.
# Bumped 34000 -> 35000 on PR #12740 for the schedule management tools
# (list_schedules, delete_schedule) needed by the trigger-agent flow.
# Bumped 35000 -> 35500 on PR #12740 for the list_agent_triggers tool
# (returns trigger agents + webhook presets for a parent agent so
# Otto can inspect/manage them).
# Bumped 35500 -> 36500 for the schedule_followup tool. Adds ~950 chars
# of LLM-decision-critical copy: delay_seconds vs cron disambiguation,
# explicit "ends your turn" caveat, and an example wake-up message.
# Bumped 36500 -> 37000 for the schedule_followup `session_id` override
# parameter — lets the model target a different conversation owned by
# the same user (parent autopilot → sub-session followups). The parameter
# description spends ~170 chars on the ownership-rejection semantics so
# the model doesn't try to wake up other users' sessions.
# Bumped 37000 -> 38500 for the skill registry (store_skill, read_skill,
# delete_skill, list_skills) — the four tools that back the new
# self-learning loop.  Descriptions are already trimmed to the minimum
# viable copy; the bump absorbs the four schema skeletons plus the
# canonical SKILL.md frontmatter callout the model needs to format
# distillations correctly.
# Bumped 38500 -> 39000 for the schedule_followup ``session_id=null``
# sentinel — its description spends ~170 chars explaining the "fire
# into a fresh chat" semantics so the model picks the right value
# (null vs omit vs target_session_id) for autopilot-style flows.
# Bumped 39000 -> 39500 for the create-time library-similarity gate:
# find_library_agent's new ``for_creation`` and ``goal_summary``
# parameters and create_agent's ``library_check_ack`` bypass — the
# extra ~270 chars on CI (env-flagged tool registrations push CI
# higher than local) carry the LLM-decision-critical copy for
# "search the library before building new" + "user-confirmed bypass".
# Bumped 39500 -> 40500 on PR #12731 for the decompose_goal tool.
# Adds ~1k chars: step-level schema (id/description/action/block_name),
# the require_approval gate, and the "STOP before building" caveat the
# model needs to halt for user approval instead of rushing into
# create_agent.
# Bumped 40500 -> 41000 when find_library_agent absorbed direct by-id lookup:
# a new ``agent_id`` parameter (library_agent_id / graph_id) that resolves the
# exact agent with no fuzzy name-search fallback, so the library "Chat" flow is
# reliable without a separate tool. Net smaller than a dedicated tool would add.
# Bumped 41000 -> 42500 for the setup_agent_webhook_trigger tool (OPEN-3152). Adds
# ~1.3k chars: identifier + trigger_config + explicit-credentials schema
# and the "manual webhooks return an exact URL / provider webhooks need
# an explicitly chosen account" copy the model needs to drive webhook
# trigger setup without inventing URLs or auto-picking credentials.
# Bumped 42500 -> 45000 for the preset-management tools (list_presets /
# update_preset / delete_preset) that complete the /presets lifecycle for
# Otto. Adds ~1.6k chars: three tool skeletons plus the "is_active
# pauses/resumes the trigger" + "inputs reconfigure & re-register the webhook"
# copy the model needs to manage triggers without re-running setup.
# Bumped 45000 -> 47000 on the dev merge: dev added the proactive chat-platform
# tools (post_to_chat_platform + list_chat_platform_channels, ~1.4k chars) on top
# of the trigger/preset tools above, so the merged registry needs both deltas.
# Bumped 47000 -> 47800 on the post-#13601 dev merge: the registry now carries
# the full merged tool set (webhook-trigger + preset lifecycle + docs/building
# tools) at 47461 chars; ~340 headroom so routine wording tweaks don't trip it.
# Bumped 47800 -> 51500 for OPEN-3188: the five agent-graph tools (create/edit/
# customize/validate/fix) replaced their bare ``{"type": "object"}`` agent_json
# with a structured schema (nodes/links/...) and gained an agent_json_ref string
# param. The structure is what stops constrained decoders collapsing the graph to
# ``{}`` and dropping it; nested props are kept type-only to minimise the spend.
# Merged registry measures 50915 chars (incl. find_library_agent's
# write_graph_to); ~580 headroom for wording tweaks.
# Includes the two-step Soul edit flow (update_expert_soul preview +
# confirm_expert_soul_update); registry measures ~52.2k chars locally, with
# ~800 headroom for CI env deltas and wording tweaks.
# Bumped 53000 -> 54_000 for the copilot tool-chain UI: ``ask_question`` is back
# in TOOL_REGISTRY as a first-class tool (docked clarifying-question flow), so
# its schema counts again on top of the Soul edit flow. Merged registry measures
# 53349 chars; ~650 headroom for CI env deltas and wording tweaks.
# Bumped 54_000 -> 59_000 for the expert team tools: delegate_to_expert plus
# the confirm-gated hire/raise pair, their shared confirm, and handoff_to_expert.
# No single session sees them all (hire/raise/confirm and handoff/soul gate on
# opposite sides of session.expert_id), but the registry total counts every
# tool. Merged registry measures 57814 chars; ~1.2k headroom for CI env deltas.
# Bumped 59_000 -> 61_000 for update_expert (the Otto-side soul edit,
# same confirm gate) and raise_expert's color palette enum + persona-name
# guidance. Merged registry measures 59625 chars; ~1.4k headroom.
# Bumped 61_000 -> 65_000. That 1.4k of headroom was gone 17 days later:
# nine tools grew 50-400 chars each with no single PR at fault, dev reached
# 60,984, and the next PR to add anything was ejected from the merge queue.
# Sized against what concurrent in-flight PRs add in AGGREGATE (the ten v0.7.5
# PRs add 1,763) rather than against whatever sits on dev today, because each
# branch's CI only ever sees its own delta. Registry measures 62,747 with all
# ten merged; 2,253 headroom.
# list_expert_chats / read_expert_chat (SECRT-2581) add 1,706 chars and fit
# under 65_000 without a bump of their own; merged registry measures 62,694.
# Bumped 65_000 -> 67_651 on 2026-09-09: dev's 62,747 plus the six expert PRs
# then in flight, which add 4,903 between them and blow the old ceiling by 2,650
# while each branch's own CI, measuring only its own delta, stays green.
# Measured per branch, not estimated:
#     #14443 fix-expert-credential-grant-paths  +3,744
#     #14207 multi-expert-teams                   +981
#     #11220 input-blocks-alongside-trigger       +178
#     #14244 agent-collab-architecture              +0
#     #14209 autopilot-auto-mode-v2                 +0
#     #14432 secrt-2593-publish                     +0
#     #14365 sandbox-e2b-desktop (start_desktop)  +670  (branch measures 63,417)
# Bumped 68_238 -> 68_997 on 2026-09-16 when #14365 merged dev (twice in one
# morning: #14416's pause/resume_schedule landed between the two). Dev's
# registry measures 68,503 with start_desktop removed, so start_desktop's
# delta is +493 once merged (the +670 above was against an older dev).
# Merged registry measures 68,996; the ceiling is that plus one.
# Bumped 68_997 -> 69_063 for #14382 (one box per owner): start_desktop's
# description now says it is the same machine bash_exec runs in, +66.
# Merged registry measures 69,062.
# Lowered 69_063 -> 69_056 on the same PR: bash_exec's description no longer
# tells the model that only ~/workspace shows on the desktop, -7.  Merged
# registry measures 69,055.
# There is NO margin on top, deliberately. This limit is a brake: it exists to
# make every increase in what Otto pays per turn a decision someone took,
# so slack for growth nobody has measured is the one thing it must not carry.
# The assertion below is a strict <, so the ceiling is the measured total plus
# one — 67,651 admits exactly that aggregate and nothing beyond it.
# The next tool that does not fit raises this line itself, with its own measured
# number and its own row above.
#
# Bumped 67_651 -> 68_604 on 2026-09-16, by the line above: #14415 landed and
# dev alone now measures 67,622, leaving 29 chars — consult_teammate's measured
# +981 does not fit. Measured on the MERGED tree, which is what this PR's CI
# runs on, not on the branch tip:
#     dev 2ee0819cc5                             67,622 (81 tools)
#     + #14207 multi-expert-teams  +981          68,603 (82 tools)
# Still no margin: strict <, so the ceiling is that total plus one. dev's own
# 29 chars are the standing problem here — the next tool anyone adds trips this
# again, whatever this line says.
#
# Bumped 71_752 -> 73_003 for find_session and message_session, the two tools
# this PR adds: they measure 1,251 between them.
#     dev + this branch's consult_teammate           71,751 (86 tools)
#     + find_session, message_session  +1,251        73,002 (88 tools)
#
# Bumped 73_003 -> 73_030 for one sentence in find_session's description
# saying `task` searches recent sessions only — 27 chars. A description edit
# costs the same on both brakes (see below); only a tool's SHAPE makes them
# differ.
#
# ON CONFLICT, KEEP THE HIGHER VALUE. Two branches tuning this line independently
# both look correct: each one's CI only measures its own delta against dev, while
# the budget has to cover what every in-flight PR adds together. Taking the
# incoming side lowers a ceiling that has already ejected a green PR.
#
# Measure it the way this test does — one json.dumps over the whole list —
# not by summing per-tool lengths, which misses ~142 chars of array
# separators and overstates the headroom.
# Bumped 67_651 -> 68_238 for pause_schedule and resume_schedule, the two tools
# this PR adds. MEASURE ON THE PR'S MERGE REF, never on the branch tip: a stacked
# PR's CI builds this branch merged through its base into dev, so dev's own growth
# counts against this ceiling. The tip reads 68,235 and `refs/pull/14416/merge`
# 68,237 — dev widened raise_expert by two characters after this line was first
# set, which reddened three interpreters on a branch that had added nothing.
# Bumped 68_238 -> 69_219 on 2026-09-16, merging dev into #14207: dev's own
# 68,238 left one character of headroom (dev measures 68,237 with the schedule
# tools above), and consult_teammate's measured +981 does not fit. Re-measured
# on the MERGED tree per the rule above, not on either side's tip:
#     dev e45aa33600                             68,237 (83 tools)
#     + #14207 multi-expert-teams  +981          69,218 (84 tools)
# Higher of the two conflicting values wins and is then re-measured, which is
# what makes it 69,219 rather than this branch's earlier 68,604. #14476 landed
# mid-merge and is in here too; it moves API routes and no tools, so 69,218 holds.
# Bumped 69_056 -> 70_037 on 2026-09-16, merging dev into #14207 again. dev's
# ceiling of 69,056 sat one character above its own 69,055 — the third merge
# running where dev is on its limit — so consult_teammate's +981 does not fit.
# Measured on the MERGED tree, never on either tip, never reused from a prior
# merge:
#     dev 648c5ce6d5                             69,055 (84 tools)
#     + #14207 multi-expert-teams  +981          70,036 (85 tools)
# Keep the HIGHER of two conflicting values and then re-measure, per the rule
# above: this branch held 69,219 and dev 69,056, and neither is the answer.
# Bumped 69_056 -> 70_771 for SECRT-2605: edit_chat_platform_message (mirroring
# post_to_chat_platform's platform/target enums plus channel_id/ref_id/content)
# measures 1,460, and the line in post_to_chat_platform's description pointing
# at it 255. Measured on the branch merged with dev: 70,770, plus one.
# Bumped 70_771 -> 71_752 on 2026-09-16, merging dev into #14207 a fourth time.
# #14436 added edit_chat_platform_message and set 70,771 against dev's own
# 70,770 — one character, as every one of these bumps has left. Re-measured on
# the MERGED tree, against the ref merged rather than origin/dev afterwards:
#     dev 28d332fb36                             70,770 (85 tools)
#     + #14207 multi-expert-teams  +981          71,751 (86 tools)
# consult_teammate has measured +981 at every dev tip since 2026-09-09; what
# moves this line is dev, not this branch.
# Bumped 73_030 -> 75_260 for the two tools this PR adds: set_expert_routine
# measures 1,823 (nine parameters, because it both creates standing work and
# switches it on, and because the modes and the credential grant are each a
# decision the owner makes out loud) and list_expert_routines 403. Measured on
# the branch merged with dev, plus one — this line carries no margin by design.
# Bumped 75_261 -> 75_693 in the same PR, for two arguments those tools grew:
# `delay_seconds` (a routine can now be a one-shot, so the durable record
# covers "check the deploy at six" and not only work that repeats) and
# `session_id` (PINNED can name any chat the owner holds, which is the one
# thing schedule_followup could do that a routine could not). Every other
# description in both tools was cut first — that paid back 377 of the 808 —
# so what is left here is the two new arguments, not wording. Measured on the
# branch merged with dev at 75,692, plus one.
# Bumped 75_693 -> 75_839 after merging dev, which reworded `raise_expert`,
# `setup_agent_webhook_trigger` and `run_agent` — no tool was added and this
# branch's own delta did not move. Re-measured on the merged tree at 75,838,
# plus one, per the rule above about measuring on the merge ref.
# Bumped 75_839 -> 76_300 for the ``tool:`` capability ids in tool descriptions
# (SECRT-2667) and the routine/follow-up split in ``schedule_followup``.
# Measured on this branch merged with dev at 76,110, +272 over dev's 75,838.
# The margin over the measurement is deliberate, and the exception to the
# no-margin rule above: this lands during the v0.8.0 release while dev is
# still moving, and the two rewordings that cost 146 chars last week would
# each have reded this PR at the queue on a measured-plus-one ceiling.
# Bumped 76_300 -> 76_686 for list_workspace_files' folder_id and recursive
# arguments and the description rewrite that tells the model where the user's
# own uploads live, plus the matching sentence in read_workspace_file. No tool
# was added (89 either side). Measured on the branch merged with dev, which
# here is the branch itself — it already contains dev's tip 480c6f5509:
#     dev 480c6f5509                              76,245 (89 tools)
#     + this PR's two descriptions   +440         76,685 (89 tools)
# Plus one; that line carried no margin by design, and dev overtook it within
# the day: #14779 made the session's skills first-class find_capability
# candidates, growing find_capability and run_capability, and the merge ref
# measured 76,714 — 28 over — with none of this branch's delta having moved.
# Re-measured on this branch merged with dev b6b03f5e72:
#     merged tree                                 76,714 (89 tools)
#     + headroom                       +300       77,014
# The margin is deliberate and is the same exception the wire budget's #14476
# note names: this is queued while dev is still moving, and a measured-plus-one
# ceiling reds the queue's merge ref on the next reworded description.
_CHAR_BUDGET = 77_014


@pytest.fixture(scope="module")
def all_tool_schemas() -> list[tuple[str, Any]]:
    """Return (tool_name, openai_schema) pairs for every registered tool."""
    return [(name, tool.as_openai_tool()) for name, tool in TOOL_REGISTRY.items()]


def _get_parametrize_data() -> list[tuple[str, object]]:
    """Build parametrize data at collection time."""
    return [(name, tool.as_openai_tool()) for name, tool in TOOL_REGISTRY.items()]


@pytest.mark.parametrize(
    "tool_name,schema",
    _get_parametrize_data(),
    ids=[name for name, _ in _get_parametrize_data()],
)
class TestToolSchema:
    """Validate schema invariants for every registered tool."""

    def test_description_non_empty(self, tool_name: str, schema: dict) -> None:
        desc = schema["function"].get("description", "")
        assert desc, f"Tool '{tool_name}' has an empty description"

    def test_required_fields_exist_in_properties(
        self, tool_name: str, schema: dict
    ) -> None:
        params = schema["function"].get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])
        for field in required:
            assert field in properties, (
                f"Tool '{tool_name}': required field '{field}' "
                f"not found in properties {list(properties.keys())}"
            )

    def test_every_property_has_type_and_description(
        self, tool_name: str, schema: dict
    ) -> None:
        params = schema["function"].get("parameters", {})
        properties = params.get("properties", {})
        for prop_name, prop_def in properties.items():
            # ``anyOf`` is the JSON-Schema-compliant way to model a
            # nullable / union-typed parameter (e.g. ``session_id`` may
            # be ``string`` or ``null`` for the fresh-chat sentinel).
            # Accept either a top-level ``type`` OR an ``anyOf`` whose
            # branches each carry their own ``type``.
            has_type = "type" in prop_def or (
                isinstance(prop_def.get("anyOf"), list)
                and all(isinstance(b, dict) and "type" in b for b in prop_def["anyOf"])
            )
            assert (
                has_type
            ), f"Tool '{tool_name}', property '{prop_name}' is missing 'type' (or a typed 'anyOf')"
            assert (
                "description" in prop_def
            ), f"Tool '{tool_name}', property '{prop_name}' is missing 'description'"


def test_browser_act_action_enum_complete() -> None:
    """Assert browser_act action enum still contains all 14 supported actions.

    This prevents future PRs from accidentally dropping actions during description
    trimming. The enum is the authoritative list — this locks it at 14 values.
    """
    tool = TOOL_REGISTRY["browser_act"]
    schema = tool.as_openai_tool()
    fn_def = schema["function"]
    params = cast(dict[str, Any], fn_def.get("parameters", {}))
    actions = params["properties"]["action"]["enum"]
    expected = {
        "click",
        "dblclick",
        "fill",
        "type",
        "scroll",
        "hover",
        "press",
        "check",
        "uncheck",
        "select",
        "wait",
        "back",
        "forward",
        "reload",
    }
    assert set(actions) == expected, (
        f"browser_act action enum changed. Got {set(actions)}, expected {expected}. "
        "If you added/removed an action, update this test intentionally."
    )


def test_total_schema_char_budget() -> None:
    """Assert total tool schema size stays under the character budget.

    This locks in the 34% token reduction from #12398 and prevents future
    description bloat from eroding the gains. Uses character count with a
    ~4 chars/token heuristic; see ``_CHAR_BUDGET`` above for the current
    value and its change history.  Character count is tokenizer-agnostic
    — no dependency on GPT or Claude tokenizers — while still providing a
    stable regression gate.
    """
    schemas = [tool.as_openai_tool() for tool in TOOL_REGISTRY.values()]
    serialized = json.dumps(schemas)
    total_chars = len(serialized)
    assert total_chars < _CHAR_BUDGET, (
        f"Tool schemas use {total_chars} chars (~{total_chars // 4} tokens), "
        f"exceeding budget of {_CHAR_BUDGET} chars (~{_CHAR_BUDGET // 4} tokens). "
        f"Description bloat detected — trim descriptions or raise the budget intentionally."
    )


# The other half of the brake: what the LARGEST session actually declares, in
# the shape the SDK path sends (``mcp__copilot__`` name, no ``required``,
# compact separators).  ``_CHAR_BUDGET`` above sums the registry, which no
# session ever gets — so hiding a tool behind a context, as the interactive
# gate does, is worth nothing to it, and the SDK-only file tools it never
# counted are ones the user pays for on every turn.  This line is that number.
#
# The largest is a plain Otto chat with both flags on and E2B: it hides the
# five ``experts`` tools where an expert chat hides the eight ``expert_admin``
# ones.  ``is_available`` is deliberately NOT applied — the environment
# decides it, so applying it would make the ceiling differ between CI and a
# laptop; every tool the session's gates admit counts, which is the upper
# bound a ceiling wants.
#
# Set at the measured 62,003 plus one on 2026-09-16, the first time this line
# existed. No margin, for the reason _CHAR_BUDGET carries none.
# Raised 62_004 -> 63_610 the same day, on the dev merge that brought
# #14436's edit_chat_platform_message: the tool measures 1,351 here and the
# line added to post_to_chat_platform's description 255, so the largest
# session moves 62,003 -> 63,609. Measured on the branch merged with dev,
# which is what CI builds — the branch tip still read 62,003 and would have
# been ejected from the queue.
# Raised 63_610 -> 64_494 on 2026-09-16, merging dev into #14207:
# consult_teammate is in the ``delegation`` group, which an Otto chat does not
# hide, so it is declared here and the largest session moves 63,609 -> 64,493.
# Its wire cost is 884, not the 981 it adds to _CHAR_BUDGET above — this shape
# drops ``required`` and uses compact separators but prefixes each name.
#     dev d028684cce                             63,609 (85 tools)
#     + #14207 multi-expert-teams  +884          64,493 (86 tools)
# Raised 64_494 -> 65_603 for find_session and message_session. Both sit in the
# ``delegation`` group, which an Otto chat does not hide, so both are declared
# here: 65,602, +1,109. That is less than the 1,251 they add to _CHAR_BUDGET
# above — the two brakes never move in step, so measure each.
#
# Raised 65_603 -> 65_630 for the same sentence: 65,629, +27, exactly the
# registry's delta. Prose lands byte-for-byte in both, so the two brakes only
# diverge on tool shape — `required` and separators — not on wording.
#
# Raised 65_630 -> 67_767 for the same two tools. They ride the
# ``expert_resources`` group, so they are declared in every session that can
# manage an expert's resources — which is the largest one. Measured 67,767,
# plus one.
# Raised 67_768 -> 68_188 for the same two arguments as ``_CHAR_BUDGET`` above;
# both tools are in that largest session, so the whole delta lands here too.
# Measured 68,187, plus one.
# Raised 68_188 -> 68_329 for the same dev rewordings; all three tools are in
# the largest session, so the whole delta lands here too. Measured 68,328,
# plus one.
#
# Raised 68_329 -> 68_800 for the ``tool:`` capability ids in tool
# descriptions (SECRT-2667) and the routine/follow-up split in
# ``schedule_followup``: a deferred tool named bare is refused, so text
# pointing at one spells its id. Measured on this branch merged with dev at
# 68,595, +267 over dev's 68,328. Dev's 68,329 was measured without those
# prefixes, so the higher of the two conflicting values was the floor here,
# not the answer. Carries the same deliberate margin as ``_CHAR_BUDGET``.
#
# Raised 68_800 -> 69_156 for the same two descriptions as ``_CHAR_BUDGET``
# above. Both file tools are in the largest session, so the whole delta lands
# here too — but the wire form drops ``required`` and prefixes each name, so it
# is +430 here against +440 there. Measured on the branch merged with dev,
# which here is the branch itself (it contains dev's tip 480c6f5509):
#     dev 480c6f5509                              68,725
#     + this PR's two descriptions   +430         69,155
# Plus one. Dev's 68,725 sits 75 under the old 68,800, which was the margin
# that line took deliberately; this one takes none.
#
# ON CONFLICT, KEEP THE HIGHER VALUE — same rule, same reason: each branch's
# CI measures only its own delta while the ceiling has to cover every in-flight
# PR together. MEASURE ON THE PR'S MERGE REF, never the branch tip.
#
# Re-measured on this branch merged with dev b6b03f5e72, after #14779 grew
# find_capability and run_capability: the merge ref measured 69,183, 27 over
# the plus-one ceiling above. Same headroom, for the same reason.
#     merged tree                                 69,183
#     + headroom                       +300       69,483
_SESSION_WIRE_BUDGET = 69_483


def test_largest_declared_session_wire_budget() -> None:
    """Assert what one session declares stays under the wire budget.

    ``test_total_schema_char_budget`` measures the registry; this measures a
    turn. The two move independently: a tool added behind a context raises the
    first and not the second, and an SDK-only file tool raises the second and
    not the first.
    """
    assert _largest_session_wire_chars() < _SESSION_WIRE_BUDGET, (
        f"The largest session declares {_largest_session_wire_chars():,} chars "
        f"of tool schema, over the {_SESSION_WIRE_BUDGET:,} budget. Hide the "
        f"tool behind a context, trim it, or raise the budget intentionally."
    )


def _largest_session_wire_chars() -> int:
    """Wire chars an Otto chat declares with both flags on, E2B, interactive."""
    from backend.copilot.sdk.e2b_file_tools import E2B_FILE_TOOLS
    from backend.copilot.sdk.tool_adapter import (
        _READ_TOOL_DESCRIPTION,
        _READ_TOOL_NAME,
        _READ_TOOL_SCHEMA,
        BASELINE_ONLY_MCP_TOOLS,
        MCP_TOOL_PREFIX,
        _build_input_schema,
    )
    from backend.copilot.tools import (
        expert_tool_disabled_groups,
        origin_disabled_tools,
        tool_names_in_groups,
    )

    def wire(name: str, description: str, schema: dict) -> int:
        entry = {
            "name": f"{MCP_TOOL_PREFIX}{name}",
            "description": description,
            "input_schema": schema,
        }
        return len(json.dumps(entry, separators=(",", ":")))

    hidden = set(
        tool_names_in_groups(
            expert_tool_disabled_groups(experts_enabled=True, expert_id=None)
        )
    )
    hidden |= set(BASELINE_ONLY_MCP_TOOLS) | {"get_agent_building_guide"}
    hidden |= origin_disabled_tools("interactive")

    total = sum(
        wire(name, tool.description, _build_input_schema(tool))
        for name, tool in TOOL_REGISTRY.items()
        if name not in hidden
    )
    total += sum(wire(name, desc, schema) for name, desc, schema, _ in E2B_FILE_TOOLS)
    return total + wire(_READ_TOOL_NAME, _READ_TOOL_DESCRIPTION, _READ_TOOL_SCHEMA)


# ── Capability-group filtering (ToolGroup / disabled_groups) ───────────


def test_get_available_tools_hides_graphiti_when_disabled() -> None:
    """When the ``graphiti`` group is disabled, the memory_* tools must
    not appear in the OpenAI schema list — they'd just confuse the model
    and produce opaque runtime errors."""
    from backend.copilot.tools import get_available_tools

    memory_tool_names = {
        "memory_store",
        "memory_search",
        "memory_forget_search",
        "memory_forget_confirm",
    }

    default = {
        t["function"]["name"] for t in get_available_tools(include_deferred=True)
    }
    assert memory_tool_names.issubset(
        default
    ), "sanity: memory_* tools should be present when no groups disabled"

    filtered = {
        t["function"]["name"]
        for t in get_available_tools(
            include_deferred=True, disabled_groups=["graphiti"]
        )
    }
    assert not (
        memory_tool_names & filtered
    ), f"graphiti disabled but memory_* still present: {memory_tool_names & filtered}"
    # Non-graphiti tools stay visible.
    assert "find_capability" in filtered
    assert "TodoWrite" in filtered


def test_get_copilot_tool_names_hides_graphiti_when_disabled() -> None:
    """Same invariant for the SDK tool-name list."""
    from backend.copilot.sdk.tool_adapter import MCP_TOOL_PREFIX, get_copilot_tool_names

    memory_mcp_names = {
        f"{MCP_TOOL_PREFIX}memory_store",
        f"{MCP_TOOL_PREFIX}memory_search",
        f"{MCP_TOOL_PREFIX}memory_forget_search",
        f"{MCP_TOOL_PREFIX}memory_forget_confirm",
    }

    # ``memory_search`` is eager — the memory supplement orders a search by
    # name — and the other three are deferred, reached through
    # run_capability.  Disabling the group must hide all four either way.
    deferred_mcp_names = memory_mcp_names - {f"{MCP_TOOL_PREFIX}memory_search"}
    default = set(get_copilot_tool_names())
    assert not deferred_mcp_names & default

    filtered = set(get_copilot_tool_names(disabled_groups=["graphiti"]))
    assert not (
        memory_mcp_names & filtered
    ), f"graphiti disabled but memory MCP names still present: {memory_mcp_names & filtered}"
    # E2B path stays consistent.
    filtered_e2b = set(
        get_copilot_tool_names(use_e2b=True, disabled_groups=["graphiti"])
    )
    assert not (memory_mcp_names & filtered_e2b)


# ── Origin filtering (INTERACTIVE_ORIGIN_TOOLS / origin_disabled_tools) ──


def test_automation_origin_declares_no_interactive_origin_tools() -> None:
    """A machine-authored session is not offered what its guard would refuse.

    The baseline path passes the set as ``disabled_tools``; the SDK path
    unions it into the names it never registers, covered against the real
    MCP server in ``sdk/tool_adapter_test.py``.  A legacy ``origin=None``
    is treated as automation, as ``autopilot_session_guard`` treats it.
    """
    from backend.copilot.tools import (
        INTERACTIVE_ORIGIN_TOOLS,
        origin_disabled_tools,
        reachable_tool_names,
    )

    for origin in ("automation", None):
        hidden = origin_disabled_tools(origin)
        assert hidden == INTERACTIVE_ORIGIN_TOOLS

        # Reachable, not declared: most of these are deferred now, so the
        # schema list would read every one of them as gated whether the gate
        # works or not.  ``run_capability`` answers to the same hidden set.
        reachable = reachable_tool_names(disabled_tools=hidden)
        assert not (INTERACTIVE_ORIGIN_TOOLS & reachable), (
            f"origin={origin!r} can still reach "
            f"{sorted(INTERACTIVE_ORIGIN_TOOLS & reachable)}"
        )
        # The gate is narrow on purpose: an automation still does its work,
        # still reports through a chat platform, still wakes itself up.
        assert {
            "run_agent",
            "run_capability",
            "run_sub_session",
            "schedule_followup",
            "ask_question",
        } <= reachable


@pytest.mark.asyncio
async def test_a_deferred_tool_named_directly_is_refused() -> None:
    """The schema list is a presentation filter; this is the boundary.

    Deferred tools are absent from every schema list, but a model that names
    one anyway (replayed transcript, prompt injection) used to reach it here
    and run it — routing around ``run_capability`` and the permission and
    envelope gates it applies.
    """
    from unittest.mock import AsyncMock, patch

    from backend.copilot.tools import DEFERRED_TOOL_NAMES, execute_tool, get_tool
    from backend.copilot.tools.models import ErrorResponse

    name = "memory_store"
    assert name in DEFERRED_TOOL_NAMES, "test relies on this tool being deferred"
    tool = get_tool(name)
    assert tool is not None

    with patch.object(
        tool, "execute", new=AsyncMock(return_value="should never run")
    ) as ran:
        result = await execute_tool(
            tool_name=name,
            parameters={},
            user_id="user-1",
            session=make_session("user-1"),
            tool_call_id="call-1",
            # Nothing else gates it: the refusal has to come from deferral.
            disabled_groups=[],
            disabled_tools=(),
        )

    ran.assert_not_awaited()
    assert result.success is False
    assert ErrorResponse.model_validate_json(result.output).error == "tool_disabled"


def test_interactive_origin_reaches_every_tool_it_did_before() -> None:
    """An interactive session reaches exactly what it reached before.

    The counterpart to the test above, and what fails if the gate ever widens
    past ``origin`` into the sessions a person really is driving.
    """
    from backend.copilot.tools import (
        INTERACTIVE_ORIGIN_TOOLS,
        origin_disabled_tools,
        reachable_tool_names,
    )

    hidden = origin_disabled_tools("interactive")
    assert hidden == frozenset()

    reachable = reachable_tool_names(disabled_tools=hidden)
    assert (
        INTERACTIVE_ORIGIN_TOOLS <= reachable
    ), f"interactive session lost {sorted(INTERACTIVE_ORIGIN_TOOLS - reachable)}"


def test_set_matches_the_tools_that_call_the_origin_guard() -> None:
    """The set is only sound while it equals what the runtime refuses.

    Hiding a tool the guard does not refuse takes a capability away from
    automations; declaring one it does refuse is the waste this PR removes.
    A new staffing tool adds ``autopilot_session_guard`` and this fails until
    the name is listed — the check no other test in the tree performs.
    """
    import inspect

    from backend.copilot.tools import INTERACTIVE_ORIGIN_TOOLS

    guarded = {
        name
        for name, tool in TOOL_REGISTRY.items()
        if "autopilot_session_guard(" in inspect.getsource(type(tool))
    }
    assert guarded == INTERACTIVE_ORIGIN_TOOLS, (
        "INTERACTIVE_ORIGIN_TOOLS is out of step with the runtime guard: "
        f"guarded but declared {sorted(guarded - INTERACTIVE_ORIGIN_TOOLS)}, "
        f"hidden but unguarded {sorted(INTERACTIVE_ORIGIN_TOOLS - guarded)}"
    )


class TestPromptSupplementsNameRealTools:
    """A supplement that names a tool the turn was never given spends tokens
    telling the model to call something that does not exist. The role split
    is where this bites: the charters were written against task tools that
    ship in a later slice, so the names have to be checked against the
    registry, not against the text they were copied from."""

    @staticmethod
    def _tool_names_mentioned(text: str) -> set[str]:
        import re

        # `name(` or `name` in backticks — how every supplement writes one.
        return {
            m
            for m in re.findall(r"`([a-z_][a-z0-9_]*)\(?[^`]*`", text)
            if m.endswith("_expert")
            or m.endswith("_task")
            or m.startswith("memory_")
            or m in {"run_sub_session", "get_sub_session_result", "ask_question"}
        }

    @pytest.mark.parametrize("role", ["autopilot", "expert"])
    def test_every_tool_named_in_a_role_section_is_registered(self, role: str) -> None:
        from backend.copilot import prompting

        sections = (
            prompting.get_role_charter(role)
            + prompting.get_delegation_supplement(role)
            + prompting.get_graphiti_supplement(role)
        )
        named = self._tool_names_mentioned(sections)
        assert named, "matcher found no tool names — it has stopped working"
        assert named <= set(TOOL_REGISTRY), (
            f"{role} sections name unregistered tools: "
            f"{sorted(named - set(TOOL_REGISTRY))}"
        )
