"""Tests for the LaunchDarkly -> PostHog flag mapping and the sync script.

Every export here is synthetic: invented keys, ids and addresses. Mapped flags
are evaluated with the PostHog SDK's own local evaluator, so a test asserts
what a user would be served, not just the payload's shape.
"""

import json
from typing import Any

import pytest
from posthog.feature_flags import InconclusiveMatchError, match_feature_flag_properties

from scripts import sync_feature_flags_to_posthog as cli
from scripts.feature_flag_sync import (
    MappedCohort,
    MappedFlag,
    describe_filters,
    map_flag,
    map_segment,
    plan_sync,
    resolve_cohort_refs,
)

ENV = "test"
ADA, BOB, CAROL = "user-ada", "user-bob", "user-carol"
STAFF = {
    "email": "ada@example.org",
    "email_domain": "example.org",
    "role": "authenticated",
}
ADMIN = {
    "email": "root@elsewhere.test",
    "email_domain": "elsewhere.test",
    "role": "admin",
}
OUTSIDER = {
    "email": "eve@elsewhere.test",
    "email_domain": "elsewhere.test",
    "role": "authenticated",
    "created_at": "2026-01-01T00:00:00+00:00",
}

SEGMENTS = {
    "staff": {
        "key": "staff",
        "name": "Staff",
        "included": [CAROL],
        "excluded": [BOB],
        "rules": [
            {
                "clauses": [
                    {
                        "attribute": "email",
                        "op": "endsWith",
                        "values": ["@example.org"],
                        "contextKind": "user",
                        "negate": False,
                    }
                ]
            }
        ],
    },
    "admins": {
        "key": "admins",
        "name": "Admins",
        "rules": [
            {
                "clauses": [
                    {
                        "attribute": "/custom/role",
                        "op": "in",
                        "values": ["admin"],
                        "negate": False,
                    }
                ]
            }
        ],
    },
    "empty": {"key": "empty", "name": "Empty", "rules": []},
}


def test_segment_becomes_dynamic_cohort_with_members_rules_and_exclusions():
    cohort = map_segment(SEGMENTS["staff"], ENV)

    assert cohort.payload is not None
    assert cohort.payload["name"] == "ld:staff"
    assert cohort.payload["is_static"] is False
    assert cohort.payload["filters"]["properties"] == {
        "type": "OR",
        "values": [
            {
                "type": "AND",
                "values": [
                    _prop("distinct_id", "exact", [CAROL]),
                    _prop("distinct_id", "is_not", [BOB]),
                ],
            },
            {
                "type": "AND",
                "values": [
                    _prop("email", "regex", r"@example\.org$"),
                    _prop("distinct_id", "is_not", [BOB]),
                ],
            },
        ],
    }


@pytest.mark.parametrize(
    "segment, reason",
    [
        ({"key": "big", "unbounded": True}, "unbounded"),
        (
            {"key": "w", "rules": [{"weight": 50000, "clauses": []}]},
            "percentage weight",
        ),
        (
            {
                "key": "n",
                "rules": [
                    {
                        "clauses": [
                            {
                                "attribute": "segmentMatch",
                                "op": "segmentMatch",
                                "values": ["x"],
                            }
                        ]
                    }
                ],
            },
            "another segment",
        ),
        (
            {
                "key": "d",
                "includedContexts": [{"contextKind": "device", "values": ["dev-1"]}],
            },
            "device",
        ),
    ],
)
def test_segment_without_posthog_equivalent_needs_a_decision(segment, reason):
    cohort = map_segment(segment, ENV)

    assert cohort.payload is None
    assert reason in cohort.decisions[0].reason


def test_boolean_flag_serves_what_launchdarkly_served():
    flag = _flag(
        "boolean",
        targets=[{"values": [ADA], "variation": 0}],
        rules=[
            _rule([_segment_clause("staff", "admins")], 0),
            _rule(
                [{"attribute": "created_at", "op": "after", "values": [1778112000000]}],
                0,
            ),
        ],
        fallthrough=1,
    )

    mapped = map_flag(flag, ENV, _cohorts())

    assert mapped.decisions == []
    assert (mapped.targets, mapped.segments, mapped.rules) == (
        1,
        ["admins", "staff"],
        2,
    )
    served = _served(mapped.payload)
    assert served(ADA, {"email": "ada@nowhere.test"}) is True
    assert served("user-x", STAFF) is True
    assert served("user-x", ADMIN) is True
    assert (
        served(
            CAROL,
            {"email": "carol@nowhere.test", "created_at": "2026-01-01T00:00:00+00:00"},
        )
        is True
    )
    assert served(BOB, {**STAFF, "created_at": "2026-01-01T00:00:00+00:00"}) is False
    assert (
        served("user-x", {**OUTSIDER, "created_at": "2026-06-01T00:00:00+00:00"})
        is True
    )
    assert served("user-x", OUTSIDER) is False


def test_false_serving_target_ahead_of_true_fallthrough_becomes_an_exclusion():
    flag = _flag("boolean", targets=[{"values": [BOB], "variation": 1}], fallthrough=0)

    served = _served(map_flag(flag, ENV, _cohorts()).payload)

    assert served(BOB, OUTSIDER) is False
    assert served(ADA, OUTSIDER) is True


def test_false_serving_rule_ahead_of_true_rule_needs_a_decision():
    flag = _flag(
        "boolean",
        rules=[
            _rule([_email_domain("elsewhere.test")], 1),
            _rule([_segment_clause("admins")], 0),
        ],
        fallthrough=1,
    )

    mapped = map_flag(flag, ENV, _cohorts())

    assert mapped.payload is None
    assert "serves false ahead" in mapped.decisions[0].reason


@pytest.mark.parametrize(
    "off_variation, active", [(1, False), (0, True), (None, False)]
)
def test_off_boolean_flag_serves_its_off_variation_to_everyone(off_variation, active):
    flag = _flag(
        "boolean",
        on=False,
        off=off_variation,
        rules=[_rule([_segment_clause("admins")], 0)],
    )

    payload = _payload(map_flag(flag, ENV, _cohorts()))

    assert payload["active"] is active
    assert _served(payload)("user-x", ADMIN) is active


@pytest.mark.parametrize(
    "clause",
    [
        {"attribute": "country", "op": "in", "values": ["IN"], "negate": True},
        {"attribute": "country", "op": "endsWith", "values": ["N"], "negate": True},
    ],
)
def test_negated_clause_never_serves_a_user_without_the_attribute(clause):
    """LaunchDarkly: a missing attribute matches no clause, negated or not.

    The trial offer's production rule is "country is not one of IN". Without
    the is_set guard, PostHog would be free to serve it to a visitor whose
    country is unknown -- the one case LaunchDarkly withholds it.
    """
    flag = _flag("boolean", rules=[_rule([clause], 0)], fallthrough=1)

    mapped = map_flag(flag, ENV, _cohorts())

    assert mapped.decisions == []
    served = _served(mapped.payload)
    assert served("user-x", {"country": "US"}) is True
    assert served("user-x", {"country": "IN"}) is False
    # The local evaluator defers to PostHog's servers here, and the group it
    # defers with can only match once country is set.
    group = _payload(mapped)["filters"]["groups"][0]["properties"]
    assert {
        "key": "country",
        "type": "person",
        "operator": "is_set",
        "value": "is_set",
    } in group


def test_country_rule_ports_instead_of_needing_a_decision():
    """The trial offer's country targeting must survive the migration."""
    flag = _flag(
        "boolean",
        rules=[
            _rule([{"attribute": "country", "op": "in", "values": ["US", "GB"]}], 0)
        ],
        fallthrough=1,
    )

    mapped = map_flag(flag, ENV, _cohorts())

    assert mapped.decisions == []
    served = _served(mapped.payload)
    assert served("user-x", {"country": "GB"}) is True
    assert served("user-x", {"country": "IN"}) is False


def test_multivariate_flag_serves_launchdarkly_values_as_payloads():
    flag = _flag(
        "multivariate",
        variations=[
            {"value": {"daily": 1}, "name": "Low"},
            {"value": {"daily": 9}, "name": "High"},
            {"value": "price_abc"},
        ],
        rules=[_rule([_email_domain("example.org")], 1)],
        fallthrough=0,
    )

    payload = _payload(map_flag(flag, ENV, _cohorts()))

    assert [v["key"] for v in payload["filters"]["multivariate"]["variants"]] == [
        "low",
        "high",
        "price_abc",
    ]
    served = _served(payload)
    assert served("user-x", STAFF) == {"daily": 9}
    assert served("system", {}) == {"daily": 1}


def test_variant_keys_stay_unique_when_a_fallback_collides_with_a_name():
    flag = _flag(
        "multivariate",
        variations=[
            {"value": 1, "name": "x"},
            {"value": 2, "name": "x-2"},
            {"value": 3, "name": "x"},
        ],
        fallthrough=2,
    )

    payload = _payload(map_flag(flag, ENV, _cohorts()))

    keys = [v["key"] for v in payload["filters"]["multivariate"]["variants"]]
    assert len(set(keys)) == 3
    assert _served(payload)("user-x", {}) == 3


def test_system_keyed_config_flag_resolves_without_person_properties():
    flag = _flag(
        "multivariate", variations=[{"value": {"PRO": 5}}, {"value": {}}], fallthrough=0
    )

    assert _served(map_flag(flag, ENV, _cohorts()).payload)("system", {}) == {"PRO": 5}


def test_rule_on_only_empty_segments_matches_nobody():
    flag = _flag("boolean", rules=[_rule([_segment_clause("empty")], 0)], fallthrough=1)

    mapped = map_flag(flag, ENV, _cohorts())

    assert _payload(mapped)["active"] is False


def test_target_on_a_context_kind_no_client_sends_is_dropped_with_a_note():
    flag = _flag(
        "boolean",
        context_targets=[
            {"contextKind": "email", "values": ["x@example.org"], "variation": 0}
        ],
        fallthrough=1,
    )

    mapped = map_flag(flag, ENV, _cohorts())

    assert _payload(mapped)["active"] is False
    assert "never matched" in mapped.notes[0]


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"prerequisites": [{"key": "other", "variation": 0}]}, "prerequisite"),
        ({"fallthrough": {"rollout": {"variations": []}}}, "percentage rollout"),
        (
            {"rules": [{"rollout": {"variations": []}, "clauses": []}]},
            "percentage rollout",
        ),
        (
            {
                "rules": [
                    {
                        "clauses": [
                            {"attribute": "plan", "op": "in", "values": ["pro"]}
                        ],
                        "variation": 0,
                    }
                ]
            },
            "never sends",
        ),
        (
            {
                "rules": [
                    {
                        "clauses": [
                            {
                                "attribute": "segmentMatch",
                                "op": "segmentMatch",
                                "values": ["unknown"],
                            }
                        ],
                        "variation": 0,
                    }
                ]
            },
            "export lacks",
        ),
        (
            {
                "rules": [
                    {
                        "clauses": [
                            {
                                "attribute": "segmentMatch",
                                "op": "segmentMatch",
                                "values": ["admins"],
                                "negate": True,
                            }
                        ],
                        "variation": 0,
                    }
                ]
            },
            "negated segment",
        ),
        (
            {
                "rules": [
                    {
                        "clauses": [
                            {
                                "attribute": "created_at",
                                "op": "after",
                                "values": [0],
                                "negate": True,
                            }
                        ],
                        "variation": 0,
                    }
                ]
            },
            "negated",
        ),
        (
            {
                "contextTargets": [
                    {"contextKind": "device", "values": ["d"], "variation": 0}
                ]
            },
            "device",
        ),
    ],
)
def test_unmappable_targeting_needs_a_decision(overrides, reason):
    flag = _flag("boolean", fallthrough=1)
    flag["environments"][ENV].update(overrides)

    mapped = map_flag(flag, ENV, _cohorts())

    assert mapped.payload is None
    assert reason in mapped.decisions[0].reason


def test_multivariate_off_without_off_variation_needs_a_decision():
    flag = _flag(
        "multivariate", variations=[{"value": {}}, {"value": []}], on=False, off=None
    )

    assert "caller's default" in map_flag(flag, ENV, _cohorts()).decisions[0].reason


def test_server_only_flag_keeps_off_the_browser():
    flag = _flag("boolean", fallthrough=0)
    flag["clientSideAvailability"] = {"usingEnvironmentId": False}

    assert _payload(map_flag(flag, ENV, _cohorts()))["evaluation_runtime"] == "server"


def test_plan_creates_missing_updates_changed_and_leaves_matching_alone():
    cohorts = [map_segment(SEGMENTS["admins"], ENV)]
    by_key = {c.segment_key: c for c in cohorts}
    admins_only = map_flag(
        _flag(
            "boolean",
            key="admins-only",
            rules=[_rule([_segment_clause("admins")], 0)],
            fallthrough=1,
        ),
        ENV,
        by_key,
    )
    everyone = map_flag(_flag("boolean", key="everyone", fallthrough=0), ENV, by_key)
    fresh = map_flag(_flag("boolean", key="fresh", fallthrough=0), ENV, by_key)
    existing_cohort = {**_payload(cohorts[0]), "id": 7}
    existing_flags = [
        {**_with_cohort_ids(_payload(admins_only), {"ld:admins": 7}), "id": 1},
        {**_payload(everyone), "id": 2, "active": False},
        {"key": "session-recording", "id": 3, "filters": {}},
        {**_payload(fresh), "id": 4, "deleted": True},
    ]

    changes = {
        c.key: c
        for c in plan_sync(
            cohorts, [admins_only, everyone, fresh], [existing_cohort], existing_flags
        )
    }

    assert changes["ld:admins"].action == "unchanged"
    assert changes["admins-only"].action == "unchanged"
    assert changes["everyone"].action == "update"
    assert changes["everyone"].diff == ["active: False -> True"]
    assert changes["everyone"].existing_id == 2
    assert changes["fresh"].action == "create"
    assert "session-recording" not in changes


def test_plan_names_a_cohort_it_has_yet_to_create_and_skips_undecided_flags():
    cohorts = [
        map_segment(SEGMENTS["admins"], ENV),
        map_segment({"key": "big", "unbounded": True}, ENV),
    ]
    by_key = {c.segment_key: c for c in cohorts}
    admins_only = map_flag(
        _flag(
            "boolean", key="admins-only", rules=[_rule([_segment_clause("admins")], 0)]
        ),
        ENV,
        by_key,
    )
    big_only = map_flag(
        _flag("boolean", key="big-only", rules=[_rule([_segment_clause("big")], 0)]),
        ENV,
        by_key,
    )

    changes = {c.key: c for c in plan_sync(cohorts, [admins_only, big_only], [], [])}

    assert changes["ld:admins"].action == "create"
    assert changes["admins-only"].diff == ["group 0: in cohort ld:admins @ 100%"]
    assert changes["big"].action == "needs-decision"
    assert changes["big-only"].action == "needs-decision"
    assert changes["big-only"].payload is None


def test_plan_updates_a_change_hidden_by_redaction():
    target = map_flag(
        _flag("boolean", key="t", targets=[{"values": [ADA], "variation": 0}]), ENV, {}
    )
    existing = _payload(target)
    existing = {
        **existing,
        "id": 5,
        "filters": {
            **existing["filters"],
            "groups": [
                {
                    **existing["filters"]["groups"][0],
                    "properties": [_prop("distinct_id", "exact", [BOB])],
                }
            ],
        },
    }

    [change] = plan_sync([], [target], [], [existing])

    assert change.action == "update"
    assert change.diff == ["filters: changed (only in redacted values)"]


def test_plan_updates_a_cohort_whose_redacted_members_changed():
    cohort = map_segment({"key": "vip", "included": [ADA]}, ENV)
    existing = {
        **_payload(cohort),
        "id": 8,
        "filters": {
            "properties": {
                "type": "OR",
                "values": [
                    {"type": "AND", "values": [_prop("distinct_id", "exact", [BOB])]}
                ],
            }
        },
    }

    [change] = plan_sync([cohort], [], [existing], [])

    assert change.action == "update"


def test_described_filters_never_show_addresses_or_user_ids():
    uid = "123e4567-e89b-12d3-a456-426614174000"
    flag = _flag(
        "multivariate",
        variations=[{"value": [uid, "ops@example.org"]}, {"value": []}],
        targets=[{"values": [uid], "variation": 0}],
        rules=[
            _rule(
                [{"attribute": "email", "op": "in", "values": ["ops@example.org"]}], 0
            ),
            _rule(
                [
                    {
                        "attribute": "email",
                        "op": "startsWith",
                        "values": ["ada@example.org"],
                    }
                ],
                0,
            ),
            _rule(
                [{"attribute": "key", "op": "matches", "values": ["^customer-123$"]}], 0
            ),
        ],
        fallthrough=1,
    )

    text = "\n".join(
        describe_filters(_payload(map_flag(flag, ENV, _cohorts()))["filters"])
    )

    assert uid not in text and "ops@example.org" not in text
    assert "ada@" not in text and "customer-123" not in text
    assert "[1 values]" in text


def test_apply_without_an_explicit_project_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(cli._Client, "send", _no_network)

    with pytest.raises(SystemExit) as exc:
        cli.main(["--ld-env", ENV, "--apply"])

    assert exc.value.code == 2
    assert "--project" in capsys.readouterr().err


def test_dry_run_reads_but_never_writes(monkeypatch):
    calls = _fake_apis(monkeypatch)

    assert cli.main(["--ld-env", ENV]) == 0

    assert {method for method, _ in calls} == {"GET"}
    assert any("/api/projects/@current/" in url for _, url in calls)


def test_apply_creates_cohorts_before_the_flags_that_reference_them(monkeypatch):
    calls = _fake_apis(monkeypatch)

    assert cli.main(["--ld-env", ENV, "--apply", "--project", "42"]) == 0

    writes = [(m, url, body) for m, url, body in calls.bodies if m != "GET"]
    assert [(m, url.rsplit("/api/projects/42/", 1)[1]) for m, url, _ in writes] == [
        ("POST", "cohorts/"),
        ("POST", "feature_flags/"),
    ]
    flag_body = writes[1][2]
    assert flag_body["filters"]["groups"][0]["properties"] == [
        {"key": "id", "type": "cohort", "value": 99}
    ]


def _with_cohort_ids(payload: dict[str, Any], ids: dict[str, int]) -> dict[str, Any]:
    return {**payload, "filters": resolve_cohort_refs(payload["filters"], ids)}


def _served(payload: dict[str, Any] | None):
    """What the PostHog SDK's local evaluator serves a user for this mapped flag."""
    assert payload is not None
    cohorts = [_payload(c) for c in _cohorts().values() if c.payload is not None]
    ids = {c["name"]: i for i, c in enumerate(cohorts, start=1)}
    cohort_properties = {
        str(ids[c["name"]]): c["filters"]["properties"] for c in cohorts
    }
    flag = _with_cohort_ids(payload, ids)

    def serve(distinct_id: str, properties: dict[str, Any]) -> Any:
        if not flag["active"]:
            return False
        try:
            value = match_feature_flag_properties(
                flag,
                distinct_id,
                {"distinct_id": distinct_id, **properties},
                cohort_properties=cohort_properties,
                bucketing_value=distinct_id,
            )
        except InconclusiveMatchError:
            return None
        if isinstance(value, bool):
            return value
        return json.loads(flag["filters"]["payloads"][value])

    return serve


def _payload(mapped: MappedFlag | MappedCohort) -> dict[str, Any]:
    assert mapped.payload is not None, mapped.decisions
    return mapped.payload


def _cohorts() -> dict[str, MappedCohort]:
    return {key: map_segment(segment, ENV) for key, segment in SEGMENTS.items()}


def _flag(
    kind: str,
    *,
    key: str = "flag",
    variations: list[dict[str, Any]] | None = None,
    on: bool = True,
    off: int | None = 1,
    targets: list[dict[str, Any]] | None = None,
    context_targets: list[dict[str, Any]] | None = None,
    rules: list[dict[str, Any]] | None = None,
    fallthrough: int = 1,
) -> dict[str, Any]:
    return {
        "key": key,
        "name": key.title(),
        "kind": kind,
        "clientSideAvailability": {"usingEnvironmentId": True},
        "variations": variations or [{"value": True}, {"value": False}],
        "environments": {
            ENV: {
                "on": on,
                "offVariation": off,
                "targets": targets or [],
                "contextTargets": context_targets or [],
                "rules": rules or [],
                "fallthrough": {"variation": fallthrough},
                "prerequisites": [],
            }
        },
    }


def _rule(clauses: list[dict[str, Any]], variation: int) -> dict[str, Any]:
    return {"clauses": clauses, "variation": variation}


def _segment_clause(*segments: str) -> dict[str, Any]:
    return {
        "attribute": "segmentMatch",
        "op": "segmentMatch",
        "values": list(segments),
        "negate": False,
    }


def _email_domain(domain: str) -> dict[str, Any]:
    return {
        "attribute": "email_domain",
        "op": "in",
        "values": [domain],
        "negate": False,
    }


def _prop(key: str, operator: str, value: Any) -> dict[str, Any]:
    return {"key": key, "type": "person", "operator": operator, "value": value}


def _no_network(*_args, **_kwargs):
    raise AssertionError("no request may be made")


class _Calls(list):
    bodies: list[tuple[str, str, Any]]


def _fake_apis(monkeypatch) -> _Calls:
    """Canned LaunchDarkly export and an empty PostHog project, recording every request."""
    monkeypatch.setenv("LAUNCHDARKLY_API_TOKEN", "ld-test-token")
    monkeypatch.setenv("POSTHOG_PERSONAL_API_KEY", "phx-test-key")
    flag = _flag(
        "boolean", key="admins-only", rules=[_rule([_segment_clause("admins")], 0)]
    )
    calls = _Calls()
    calls.bodies = []

    def send(self, method: str, path_or_url: str, body: Any = None) -> Any:
        url = path_or_url if path_or_url.startswith("http") else self.base + path_or_url
        calls.append((method, url))
        calls.bodies.append((method, url, body))
        if "/flags/default" in url:
            return {"items": [flag], "totalCount": 1}
        if "/segments/default/" in url:
            return SEGMENTS["admins"]
        if url.endswith(("/@current/", "/42/")):
            return {"id": 42, "name": "Test project"}
        if method == "POST" and url.endswith("/cohorts/"):
            return {"id": 99}
        if method == "GET":
            return {"results": [], "next": None}
        return {}

    monkeypatch.setattr(cli._Client, "send", send)
    return calls
