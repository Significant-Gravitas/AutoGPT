"""Tests for expert routines: the roster invariants and the cadence spread.

The DB-backed lifecycle (install, enable, disable, archive) is covered in
``experts_db_test``; what lives here is everything checkable without one —
which is deliberately where the safety rules sit, so a roster edit that would
ship unattended standing work fails in a fast test rather than on someone's
account.
"""

from apscheduler.triggers.cron import CronTrigger

from backend.api.features.experts import seed
from backend.api.features.experts.routines import _session_mode, _spread_cron
from backend.copilot.permissions import (
    CAPABILITY_GATE_NAMES,
    unattended_routine_disabled_tools,
)
from backend.copilot.tools import TOOL_REGISTRY

# Every routine the roster ships, as (expert, key). A routine fires unattended
# once its owner switches it on, and what it says is written by whoever wrote
# the roster entry — so adding one is a deliberate edit to this set, never a
# silent roster change.
EXPECTED_ROSTER_ROUTINES: set[tuple[str, str]] = {
    ("Maria", "content-pipeline-check"),
    ("Jules", "repurposing-queue-check"),
    ("Nadia", "competitor-brief"),
    ("Remy", "lifecycle-performance-read"),
    ("Max", "weekday-prospecting-batch"),
    ("Max", "monday-list-top-up"),
    ("Max", "friday-pipeline-recap"),
    ("Frankie", "day-ahead-brief"),
    ("Frankie", "week-ahead-review"),
}

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


def test_roster_routine_crons_resolve_to_something_apscheduler_accepts():
    """A malformed cron — or an ``H`` nobody resolved — would only surface when
    somebody switched the routine on, which is the worst place to find out."""
    for entry in seed.ROSTER:
        for routine in entry["routines"]:
            for index, cron in enumerate(routine["crons"]):
                assert len(cron.split()) == 5, (entry["name"], routine["key"], cron)
                resolved = _spread_cron(cron, seed=f"u:{routine['key']}:{index}")
                CronTrigger.from_crontab(resolved, timezone="UTC")


def test_roster_routines_spread_their_hour():
    """Every one of these is "at its scheduled hour" from its source package, so
    the minute is an artefact of writing it on the hour. Left literal, the five
    that say 9am would land on one account together."""
    for entry in seed.ROSTER:
        for routine in entry["routines"]:
            for cron in routine["crons"]:
                assert cron.startswith("H "), (entry["name"], routine["key"], cron)


def test_spread_moves_a_fixed_minute_within_its_hour():
    spread = _spread_cron("H 9 * * 1", seed="user-a:morning-sweep:0")
    minute, hour, dom, month, dow = spread.split()
    assert (hour, dom, month, dow) == ("9", "*", "*", "1")
    assert 0 <= int(minute) < 60


def test_spread_is_stable_for_the_same_owner_and_routine():
    """The owner has to be able to rely on it: a routine that lands on a
    different minute after every deploy is not a cadence."""
    first = _spread_cron("H 9 * * 1", seed="user-a:morning-sweep:0")
    second = _spread_cron("H 9 * * 1", seed="user-a:morning-sweep:0")
    assert first == second


def test_spread_separates_two_owners_on_the_same_cadence():
    """The whole point: five experts all say Monday 9am, and the turns that
    lose the race to run do not fail loudly, they simply never happen."""
    minutes = {
        _spread_cron("H 9 * * 1", seed=f"user-{n}:morning-sweep:0").split()[0]
        for n in range(20)
    }
    assert len(minutes) > 10


def test_spread_separates_two_routines_on_one_account():
    a = _spread_cron("H 9 * * 1", seed="user-a:queue-sweep:0")
    b = _spread_cron("H 9 * * 1", seed="user-a:pipeline-read:0")
    assert a != b


def test_spread_separates_the_fire_times_of_one_routine():
    """A routine with two crons in the same hour must not collapse onto one."""
    a = _spread_cron("H 8 * * *", seed="user-a:callback-sweep:0")
    b = _spread_cron("H 8 * * *", seed="user-a:callback-sweep:1")
    assert a != b


def test_spread_leaves_every_cadence_that_is_not_an_H():
    """A named minute is a decision — a roster author who wrote 07:40 meant it,
    and so does an owner who asked for 10am. Only ``H`` defers the choice."""
    for cron in ["0 9 * * 1", "40 7 * * *", "*/15 * * * *", "* 9 * * 1", "not a cron"]:
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


def test_roster_routines_ask_before_they_run():
    """Every seeded routine is a proposal written for everybody, so each one has
    to name what it needs from this owner. A routine with no asks would schedule
    straight off the template, against guesses, unattended."""
    for entry in seed.ROSTER:
        for routine in entry["routines"]:
            assert routine["asks"], (entry["name"], routine["key"])


def test_roster_routines_ask_for_a_timezone():
    """Crons resolve in the owner's timezone and the suggested hour is a guess,
    so every routine has to settle when it actually runs."""
    for entry in seed.ROSTER:
        for routine in entry["routines"]:
            asks = " ".join(routine["asks"]).lower()
            assert "timezone" in asks, (entry["name"], routine["key"])


def test_every_session_mode_is_reachable_by_name():
    """The three modes are the tool's enum, so a name that stopped parsing here
    would silently collapse every routine onto THREAD."""
    for name in ["THREAD", "HERE", "FRESH"]:
        assert _session_mode(name).value == name


def test_session_mode_is_case_insensitive():
    assert _session_mode("thread").value == "THREAD"


def test_an_unknown_session_mode_falls_back_to_thread():
    """It arrives as a model argument. A typo should give the routine its own
    thread — the mode that keeps its memory — not fail a call the owner already
    agreed to."""
    assert _session_mode("pinned").value == "THREAD"
