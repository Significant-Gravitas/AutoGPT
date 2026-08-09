"""Usage-guard tests — the deterministic backstop that keeps the dream
pass from demoting facts the user demonstrably still uses.

Pure functions over ``FactRow`` / ``DreamDemotion``; no Graphiti, no
Redis, no LLM.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .fetch import FactRow
from .schemas import DreamDemotion
from .usage import (
    RECENT_RECALL_WINDOW,
    drop_recently_used_demotions,
    format_usage,
    protected_fact_uuids,
)


def _fact(
    uuid: str,
    *,
    recall_count: int | None = None,
    last_recalled_at: str | None = None,
    prev_recalled_at: str | None = None,
) -> FactRow:
    return FactRow(
        uuid=uuid,
        source="a",
        target="b",
        name="rel",
        fact="a relates to b",
        scope="real:global",
        confidence=0.8,
        status="active",
        created_at="2026-01-01T00:00:00+00:00",
        recall_count=recall_count,
        last_recalled_at=last_recalled_at,
        prev_recalled_at=prev_recalled_at,
    )


def _ago(**kwargs) -> str:
    return (datetime.now(timezone.utc) - timedelta(**kwargs)).isoformat()


def _demote(uuid: str) -> DreamDemotion:
    return DreamDemotion(edge_uuid=uuid, reason="stale_fact")


# ---------------------------------------------------------------------------
# protected_fact_uuids
# ---------------------------------------------------------------------------


def test_two_recalls_within_the_window_protect():
    """Both of the two latest (deduped) recalls inside the window —
    used on two separate days this week."""
    facts = [
        _fact(
            "hot",
            recall_count=2,
            last_recalled_at=_ago(days=1),
            prev_recalled_at=_ago(days=3),
        )
    ]
    assert protected_fact_uuids(facts) == {"hot"}


def test_never_recalled_fact_is_not_protected():
    """Absent props (the pre-hook default) must read as never-recalled —
    that's what makes the no-backfill migration story correct."""
    assert protected_fact_uuids([_fact("cold")]) == set()


def test_single_recall_is_not_enough_to_protect():
    """One hit is indistinguishable from an incidental similarity match."""
    facts = [_fact("brushed", recall_count=1, last_recalled_at=_ago(hours=1))]
    assert protected_fact_uuids(facts) == set()


def test_old_recalls_plus_one_incidental_hit_do_not_protect():
    """The lifetime counter must not stand in for windowed repetition:
    two recalls a year ago plus one incidental hit yesterday is exactly
    the single-hit case the policy refuses to protect."""
    facts = [
        _fact(
            "once-busy",
            recall_count=3,
            last_recalled_at=_ago(days=1),
            prev_recalled_at=_ago(days=365),
        )
    ]
    assert protected_fact_uuids(facts) == set()


def test_recalls_outside_the_window_do_not_protect():
    """A once-busy fact that has gone quiet becomes prunable again —
    without this the graph would ossify."""
    facts = [
        _fact(
            "gone-quiet",
            recall_count=50,
            last_recalled_at=_ago(days=RECENT_RECALL_WINDOW.days + 1),
            prev_recalled_at=_ago(days=RECENT_RECALL_WINDOW.days + 2),
        )
    ]
    assert protected_fact_uuids(facts) == set()


def test_second_recall_just_inside_the_window_still_protects():
    facts = [
        _fact(
            "edge-of-window",
            recall_count=2,
            last_recalled_at=_ago(days=1),
            prev_recalled_at=_ago(days=RECENT_RECALL_WINDOW.days, seconds=-60),
        )
    ]
    assert protected_fact_uuids(facts) == {"edge-of-window"}


def test_recall_count_without_timestamps_does_not_protect():
    """A count with no stamps can't be dated, so it can't be shown to be
    recent — fail towards allowing the demotion rather than pinning a
    fact forever on an unparseable signal."""
    facts = [_fact("undated", recall_count=99)]
    assert protected_fact_uuids(facts) == set()


def test_unparseable_timestamp_does_not_protect():
    facts = [
        _fact(
            "garbled",
            recall_count=99,
            last_recalled_at=_ago(hours=1),
            prev_recalled_at="not-a-date",
        )
    ]
    assert protected_fact_uuids(facts) == set()


def test_trailing_z_timestamp_is_parsed():
    """FalkorDB's toString(datetime) renders a trailing Z, which
    fromisoformat rejected before 3.11."""
    stamp = (datetime.now(timezone.utc) - timedelta(days=1)).replace(
        tzinfo=None
    ).isoformat() + "Z"
    facts = [
        _fact(
            "z-stamped",
            recall_count=3,
            last_recalled_at=_ago(hours=1),
            prev_recalled_at=stamp,
        )
    ]
    assert protected_fact_uuids(facts) == {"z-stamped"}


# ---------------------------------------------------------------------------
# drop_recently_used_demotions
# ---------------------------------------------------------------------------


def test_demotion_of_a_protected_fact_is_dropped():
    facts = [
        _fact(
            "hot",
            recall_count=5,
            last_recalled_at=_ago(days=1),
            prev_recalled_at=_ago(days=2),
        ),
        _fact("cold"),
    ]
    kept = drop_recently_used_demotions("p-1", [_demote("hot"), _demote("cold")], facts)
    assert [d.edge_uuid for d in kept] == ["cold"]


def test_guard_fails_open_without_usage_data():
    """No facts (bundle expired / caller had none) means no usage data —
    keep the demotions rather than zeroing the pass."""
    demotions = [_demote("anything")]
    assert drop_recently_used_demotions("p-2", demotions, None) == demotions
    assert drop_recently_used_demotions("p-2", demotions, []) == demotions


def test_guard_is_a_noop_when_nothing_is_protected():
    facts = [_fact("cold"), _fact("also-cold", recall_count=1)]
    demotions = [_demote("cold")]
    assert drop_recently_used_demotions("p-3", demotions, facts) == demotions


def test_empty_demotion_list_is_returned_untouched():
    facts = [
        _fact(
            "hot",
            recall_count=9,
            last_recalled_at=_ago(hours=2),
            prev_recalled_at=_ago(days=1),
        )
    ]
    assert drop_recently_used_demotions("p-4", [], facts) == []


def test_contradiction_and_retraction_reasons_override_protection():
    """Usage disproves staleness, not wrongness — direct contradictions
    and explicit user retractions demote even heavily-used facts."""
    facts = [
        _fact(
            "hot",
            recall_count=9,
            last_recalled_at=_ago(hours=2),
            prev_recalled_at=_ago(days=1),
        )
    ]
    demotions = [
        DreamDemotion(edge_uuid="hot", reason="contradicted_by:abc-123"),
        DreamDemotion(edge_uuid="hot", reason="web_contradicted:https://x.test"),
        DreamDemotion(edge_uuid="hot", reason="user_signal"),
    ]
    assert drop_recently_used_demotions("p-5", demotions, facts) == demotions


def test_unknown_and_staleness_reasons_stay_blocked_for_protected_facts():
    """Only the explicit contradiction/retraction vocabulary overrides —
    ``stale_fact``, ``entity_invalidated:*``, and free-text reasons the
    model invents are all treated as staleness claims."""
    facts = [
        _fact(
            "hot",
            recall_count=9,
            last_recalled_at=_ago(hours=2),
            prev_recalled_at=_ago(days=1),
        )
    ]
    demotions = [
        DreamDemotion(edge_uuid="hot", reason="stale_fact"),
        DreamDemotion(edge_uuid="hot", reason="entity_invalidated:abc-123"),
        DreamDemotion(edge_uuid="hot", reason="no longer seems relevant"),
    ]
    assert drop_recently_used_demotions("p-6", demotions, facts) == []


# ---------------------------------------------------------------------------
# format_usage — what the dream prompt shows the model
# ---------------------------------------------------------------------------


def test_never_recalled_facts_render_an_explicit_zero():
    """Absence is rendered, not omitted — the model should read 'never
    used' as a positive signal to prune, not have to infer it."""
    assert format_usage(_fact("cold")) == "recalls=0(never)"
    assert format_usage(_fact("cold", recall_count=0)) == "recalls=0(never)"


def test_recalled_facts_render_count_and_last_recall():
    rendered = format_usage(
        _fact("hot", recall_count=4, last_recalled_at="2026-08-01T00:00:00+00:00")
    )
    assert "recalls=4" in rendered
    assert "2026-08-01T00:00:00+00:00" in rendered


def test_count_without_a_timestamp_renders_a_placeholder():
    rendered = format_usage(_fact("undated", recall_count=4))
    assert (
        rendered
        == "recalls=4 last_recall=? prior_recall=none usage=demotable-on-staleness"
    )


def test_rendered_verdict_matches_the_guard_decision():
    """The prompt promises usage-protected demotions get dropped by the
    guard — so what the model reads must be the guard's own verdict, not
    a proxy it has to infer from the count."""
    protected = _fact(
        "hot",
        recall_count=2,
        last_recalled_at=_ago(days=1),
        prev_recalled_at=_ago(days=2),
    )
    # High lifetime count, recent last recall — but the SECOND-latest
    # recall is ancient, so the guard will not block a staleness demotion.
    looks_hot_but_isnt = _fact(
        "stale",
        recall_count=40,
        last_recalled_at=_ago(hours=2),
        prev_recalled_at=_ago(days=400),
    )

    assert "usage=protected" in format_usage(protected)
    assert protected_fact_uuids([protected]) == {"hot"}
    assert "usage=demotable-on-staleness" in format_usage(looks_hot_but_isnt)
    assert protected_fact_uuids([looks_hot_but_isnt]) == set()
