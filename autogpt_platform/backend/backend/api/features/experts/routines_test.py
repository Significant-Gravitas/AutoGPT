"""Tests for expert routines: the roster invariants and the cadence spread.

The DB-backed lifecycle (install, enable, disable, archive) is covered in
``experts_db_test``; what lives here is everything checkable without one —
which is deliberately where the safety rules sit, so a roster edit that would
ship unattended standing work fails in a fast test rather than on someone's
account.
"""

from backend.api.features.experts import seed
from backend.api.features.experts.routines import _spread_cron
from backend.copilot.permissions import (
    CAPABILITY_GATE_NAMES,
    unattended_routine_disabled_tools,
)
from backend.copilot.tools import TOOL_REGISTRY

# Every routine the roster ships, as (expert, key). A routine fires unattended
# once its owner switches it on, and what it says is written by whoever wrote
# the roster entry — so adding one is a deliberate edit to this set, never a
# silent roster change.
EXPECTED_ROSTER_ROUTINES: set[tuple[str, str]] = set()

VALID_SESSION_MODES = {"FRESH", "HERE", "THREAD"}


def test_roster_routines_are_declared():
    assert {
        (entry["name"], routine["key"])
        for entry in seed.ROSTER
        for routine in entry["routines"]
    } == EXPECTED_ROSTER_ROUTINES


def test_roster_routine_keys_are_unique_per_expert():
    """``ExpertRoutine`` is unique on (expertId, key), so a duplicate key would
    make the second row silently overwrite the first at seed time."""
    for entry in seed.ROSTER:
        keys = [routine["key"] for routine in entry["routines"]]
        assert len(keys) == len(set(keys)), entry["name"]


def test_roster_routines_ship_a_cadence_and_a_valid_mode():
    for entry in seed.ROSTER:
        for routine in entry["routines"]:
            assert routine["crons"], (entry["name"], routine["key"])
            assert routine["session_mode"] in VALID_SESSION_MODES, (
                entry["name"],
                routine["key"],
            )


def test_roster_routine_crons_are_five_field():
    """A malformed cron would only surface when somebody switched the routine
    on, which is the worst place to find out."""
    for entry in seed.ROSTER:
        for routine in entry["routines"]:
            for cron in routine["crons"]:
                assert len(cron.split()) == 5, (entry["name"], routine["key"], cron)


def test_spread_moves_a_fixed_minute_within_its_hour():
    spread = _spread_cron("0 9 * * 1", seed="user-a:morning-sweep:0")
    minute, hour, dom, month, dow = spread.split()
    assert (hour, dom, month, dow) == ("9", "*", "*", "1")
    assert 0 <= int(minute) < 60


def test_spread_is_stable_for_the_same_owner_and_routine():
    """The owner has to be able to rely on it: a routine that lands on a
    different minute after every deploy is not a cadence."""
    first = _spread_cron("0 9 * * 1", seed="user-a:morning-sweep:0")
    second = _spread_cron("0 9 * * 1", seed="user-a:morning-sweep:0")
    assert first == second


def test_spread_separates_two_owners_on_the_same_cadence():
    """The whole point: five experts all say Monday 9am, and the turns that
    lose the race to run do not fail loudly, they simply never happen."""
    minutes = {
        _spread_cron("0 9 * * 1", seed=f"user-{n}:morning-sweep:0").split()[0]
        for n in range(20)
    }
    assert len(minutes) > 10


def test_spread_separates_two_routines_on_one_account():
    a = _spread_cron("0 9 * * 1", seed="user-a:queue-sweep:0")
    b = _spread_cron("0 9 * * 1", seed="user-a:pipeline-read:0")
    assert a != b


def test_spread_separates_the_fire_times_of_one_routine():
    """A routine with two crons in the same hour must not collapse onto one."""
    a = _spread_cron("30 8 * * *", seed="user-a:callback-sweep:0")
    b = _spread_cron("0 8 * * *", seed="user-a:callback-sweep:1")
    assert a != b


def test_spread_leaves_a_cadence_it_cannot_safely_rewrite():
    """``*/15`` and ``*`` are intents, not times; moving them would change what
    the routine means rather than when it runs."""
    for cron in ["*/15 * * * *", "* 9 * * 1", "0,30 9 * * 1", "not a cron"]:
        assert _spread_cron(cron, seed="user-a:x:0") == cron


def test_the_unattended_denylist_names_things_that_exist():
    """A name that has been renamed out from under this set denies nothing, and
    the routine would quietly gain the reach the denylist exists to remove."""
    for name in unattended_routine_disabled_tools():
        assert name in TOOL_REGISTRY or name in CAPABILITY_GATE_NAMES, name


def test_the_unattended_denylist_closes_both_capability_gates():
    """Blocks and MCP servers are reached through ``run_capability``, so only
    the gates withhold them — denying a tool name would leave both open."""
    assert CAPABILITY_GATE_NAMES <= unattended_routine_disabled_tools()
