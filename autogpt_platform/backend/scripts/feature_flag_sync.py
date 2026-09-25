"""Map LaunchDarkly flags and segments onto PostHog feature flags and cohorts.

Pure functions over the two vendors' REST shapes, with no I/O, so the mapping
can be tested against a synthetic export. ``sync_feature_flags_to_posthog.py``
does the fetching, planning output and applying.

Anything PostHog cannot express exactly becomes a :class:`Decision` and the
flag or cohort is left out of the plan, never approximated.
"""

from __future__ import annotations

import difflib
import itertools
import json
import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel

# What `_person_properties` in `backend.util.feature_flag` sends for a user;
# the SDK adds `distinct_id` (the user id, LaunchDarkly's context key) itself.
# `country` is not stored on the person: it is the visitor's ISO country code,
# passed per evaluation by callers that know it (the trial offer, from the
# country token the proxy signs), so it is absent whenever a caller does not.
PERSON_PROPERTIES = frozenset(
    {"email", "email_domain", "role", "created_at", "country"}
)
ATTRIBUTE_ALIASES = {
    "/custom/role": "role",
    "custom.role": "role",
    "key": "distinct_id",
}
COHORT_PREFIX = "ld:"
# Context kinds our LaunchDarkly clients send: `user` everywhere, `device` for
# signed-out browsers. A target on any other kind has never matched anyone.
_SENT_CONTEXT_KINDS = frozenset({"user", "device"})

_REGEX_OPS = {"endsWith": "{}$", "startsWith": "^{}", "contains": "{}"}
_NUMERIC_OPS = {
    "lessThan": "lt",
    "lessThanOrEqual": "lte",
    "greaterThan": "gt",
    "greaterThanOrEqual": "gte",
}
_REDACTED_VALUE_KEYS = frozenset({"email", "distinct_id"})
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+")
# An address inside a regex has escaped dots the pattern above misses.
_LOCAL_PART = re.compile(r"[A-Za-z0-9._%+-]+@(?=[A-Za-z0-9-])")
_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)


class Decision(BaseModel):
    subject: str
    reason: str


class MappedCohort(BaseModel):
    segment_key: str
    payload: dict[str, Any] | None
    decisions: list[Decision] = []
    notes: list[str] = []


class MappedFlag(BaseModel):
    key: str
    kind: Literal["boolean", "multivariate"]
    targets: int
    segments: list[str]
    rules: int
    payload: dict[str, Any] | None
    decisions: list[Decision] = []
    notes: list[str] = []

    @property
    def targeted(self) -> bool:
        return bool(self.targets or self.segments or self.rules)


class Change(BaseModel):
    kind: Literal["cohort", "flag"]
    key: str
    action: Literal["create", "update", "unchanged", "needs-decision"]
    diff: list[str] = []
    payload: dict[str, Any] | None = None
    existing_id: int | None = None


def map_segment(segment: dict[str, Any], env: str) -> MappedCohort:
    """A LaunchDarkly segment as a dynamic PostHog cohort payload.

    ``payload`` is None for a segment with no members, which a flag then
    treats as matching nobody.
    """
    key = segment["key"]
    subject = f"segment `{key}`"
    notes: list[str] = []
    try:
        groups = _segment_groups(segment, notes)
    except Unmappable as e:
        return MappedCohort(
            segment_key=key,
            payload=None,
            decisions=[Decision(subject=subject, reason=str(e))],
        )
    if not groups:
        return MappedCohort(segment_key=key, payload=None, notes=notes)
    return MappedCohort(
        segment_key=key,
        notes=notes,
        payload={
            "name": cohort_name(key),
            "description": (
                f"{segment.get('name') or key}: synced from LaunchDarkly segment "
                f"`{key}` ({env})."
            ),
            "is_static": False,
            "filters": {
                "properties": {
                    "type": "OR",
                    "values": [{"type": "AND", "values": g} for g in groups],
                }
            },
        },
    )


def map_flag(
    flag: dict[str, Any], env: str, cohorts: dict[str, MappedCohort]
) -> MappedFlag:
    """A LaunchDarkly flag's configuration in *env* as a PostHog flag payload.

    LaunchDarkly serves the first matching target, rule or fallthrough;
    PostHog serves the first matching condition group, so each path becomes
    groups in the same order. Cohort conditions carry the cohort's NAME until
    :func:`resolve_cohort_refs` swaps in the target project's id.
    """
    key = flag["key"]
    cfg = flag["environments"][env]
    boolean = flag.get("kind") == "boolean"
    rules = cfg.get("rules") or []
    segments = sorted(
        {
            s
            for r in rules
            for c in r.get("clauses", [])
            if c.get("op") == "segmentMatch"
            for s in c.get("values", [])
        }
    )
    mapped = MappedFlag(
        key=key,
        kind="boolean" if boolean else "multivariate",
        targets=sum(
            len(t.get("values") or [])
            for t in (cfg.get("targets") or []) + (cfg.get("contextTargets") or [])
        ),
        segments=segments,
        rules=len(rules),
        payload=None,
    )
    try:
        paths = _serving_paths(cfg, cohorts, mapped.notes)
        filters, active = (
            _boolean_filters(flag, paths)
            if boolean
            else _multivariate_filters(flag, paths)
        )
    except Unmappable as e:
        mapped.decisions.append(Decision(subject=f"flag `{key}`", reason=str(e)))
        return mapped

    mapped.payload = {
        "key": key,
        "name": flag.get("name") or key,
        "active": active,
        "filters": filters,
        "ensure_experience_continuity": False,
        "evaluation_runtime": (
            "all"
            if (flag.get("clientSideAvailability") or {}).get("usingEnvironmentId")
            else "server"
        ),
    }
    return mapped


def cohort_name(segment_key: str) -> str:
    return f"{COHORT_PREFIX}{segment_key}"


def plan_sync(
    cohorts: list[MappedCohort],
    flags: list[MappedFlag],
    existing_cohorts: list[dict[str, Any]],
    existing_flags: list[dict[str, Any]],
) -> list[Change]:
    """Create / update / unchanged per cohort and flag; nothing is ever deleted.

    Cohorts match by name and flags by key. A flag's cohort references are
    resolved against the cohorts already in the project, so a flag waiting on
    a cohort this plan creates shows the cohort's name in its diff.
    """
    live_cohorts = {c["name"]: c for c in existing_cohorts if not c.get("deleted")}
    live_flags = {f["key"]: f for f in existing_flags if not f.get("deleted")}
    cohort_ids = {name: c["id"] for name, c in live_cohorts.items()}
    changes: list[Change] = []

    for cohort in cohorts:
        if cohort.decisions:
            changes.append(
                Change(
                    kind="cohort",
                    key=cohort.segment_key,
                    action="needs-decision",
                    diff=[d.reason for d in cohort.decisions],
                )
            )
        elif cohort.payload is not None:
            changes.append(
                _change(
                    "cohort",
                    cohort.payload["name"],
                    cohort.payload,
                    live_cohorts.get(cohort.payload["name"]),
                )
            )

    for flag in flags:
        if flag.payload is None:
            changes.append(
                Change(
                    kind="flag",
                    key=flag.key,
                    action="needs-decision",
                    diff=[d.reason for d in flag.decisions],
                )
            )
            continue
        payload = {
            **flag.payload,
            "filters": resolve_cohort_refs(flag.payload["filters"], cohort_ids),
        }
        changes.append(_change("flag", flag.key, payload, live_flags.get(flag.key)))
    return changes


def resolve_cohort_refs(
    filters: dict[str, Any], cohort_ids: dict[str, int]
) -> dict[str, Any]:
    """Replace cohort-name placeholders with ids, leaving unknown names as-is."""
    return {
        **filters,
        "groups": [
            {
                **g,
                "properties": [
                    (
                        {**p, "value": cohort_ids.get(p["value"], p["value"])}
                        if p.get("type") == "cohort"
                        else p
                    )
                    for p in g.get("properties") or []
                ],
            }
            for g in filters.get("groups", [])
        ],
    }


def describe_filters(filters: dict[str, Any]) -> list[str]:
    """Readable lines for a flag's filters, with email and user-id values counted, not shown."""
    variants = (filters.get("multivariate") or {}).get("variants") or []
    payloads = filters.get("payloads") or {}
    lines = [
        f"group {i}: "
        + (
            " AND ".join(describe_property(p) for p in g.get("properties") or [])
            or "everyone"
        )
        + f" @ {_rollout(g)}%"
        + (f" -> {g['variant']}" if g.get("variant") else "")
        for i, g in enumerate(filters.get("groups", []))
    ]
    lines += [
        f"variant {v['key']} ({v.get('rollout_percentage', 0)}%): "
        f"{_short(_canonical_payload(payloads.get(v['key'])))}"
        for v in variants
    ]
    return lines


def describe_cohort(filters: dict[str, Any]) -> list[str]:
    groups = (filters.get("properties") or {}).get("values") or []
    return [
        "OR " + " AND ".join(describe_property(p) for p in g.get("values") or [])
        for g in groups
    ]


def describe_property(prop: dict[str, Any]) -> str:
    if prop.get("type") == "cohort":
        return f"in cohort {prop.get('value')}"
    value = prop.get("value")
    if prop.get("key") in _REDACTED_VALUE_KEYS and isinstance(value, list):
        shown = f"[{len(value)} values]"
    elif prop.get("key") == "distinct_id":
        shown = "<user-id pattern>"
    else:
        shown = _redact(json.dumps(value))
    return f"{prop.get('key')} {prop.get('operator') or 'exact'} {shown}"


class Unmappable(Exception):
    """A LaunchDarkly construct with no exact PostHog equivalent."""


def _segment_groups(
    segment: dict[str, Any], notes: list[str]
) -> list[list[dict[str, Any]]]:
    if segment.get("unbounded"):
        raise Unmappable("big (unbounded) segment: membership lives outside the API")
    included = _user_keys(
        [{"values": segment.get("included")}, *(segment.get("includedContexts") or [])],
        notes,
    )
    excluded = _user_keys(
        [{"values": segment.get("excluded")}, *(segment.get("excludedContexts") or [])],
        notes,
    )
    groups: list[list[dict[str, Any]]] = []
    if included:
        groups.append([_person("distinct_id", "exact", included)])
    for rule in segment.get("rules") or []:
        if rule.get("weight") is not None:
            raise Unmappable("segment rule has a percentage weight")
        clauses = rule.get("clauses") or []
        if any(c.get("op") == "segmentMatch" for c in clauses):
            raise Unmappable("segment rule references another segment")
        if _reachable(clauses, notes):
            groups.append([p for c in clauses for p in _clause_properties(c)])
    if excluded:
        groups = [g + [_person("distinct_id", "is_not", excluded)] for g in groups]
    return groups


def _targets(cfg: dict[str, Any], notes: list[str]) -> dict[int, list[str]]:
    """Individual user targets by variation; ``contextTargets`` of kind user mirror ``targets``."""
    entries = (cfg.get("targets") or []) + (cfg.get("contextTargets") or [])
    by_variation = {
        variation: _user_keys(
            [t for t in entries if t["variation"] == variation], notes
        )
        for variation in sorted({t["variation"] for t in entries})
    }
    return {v: keys for v, keys in by_variation.items() if keys}


def _user_keys(entries: list[dict[str, Any]], notes: list[str]) -> list[str]:
    keys: set[str] = set()
    for entry in entries:
        values = entry.get("values") or []
        if values and _kind_is_sent(
            entry.get("contextKind"), f"{len(values)} targets", notes
        ):
            keys.update(values)
    return sorted(keys)


def _reachable(clauses: list[dict[str, Any]], notes: list[str]) -> bool:
    """False for a rule conditioned on a context kind no client sends: it never matched."""
    return all(_kind_is_sent(c.get("contextKind"), "a rule", notes) for c in clauses)


def _kind_is_sent(kind: str | None, what: str, notes: list[str]) -> bool:
    if kind in (None, "user"):
        return True
    if kind in _SENT_CONTEXT_KINDS:
        raise Unmappable(
            f"{what} on context kind `{kind}`, which has no PostHog person"
        )
    notes.append(
        f"dropped {what} on context kind `{kind}`: no client sends it, so it never matched"
    )
    return False


class _Path(BaseModel):
    """One LaunchDarkly serving path: any of ``groups`` matches -> ``variation``."""

    groups: list[list[dict[str, Any]]]
    variation: int | None
    is_target: bool = False


def _serving_paths(
    cfg: dict[str, Any], cohorts: dict[str, MappedCohort], notes: list[str]
) -> list[_Path]:
    if cfg.get("prerequisites"):
        raise Unmappable("has prerequisite flags")
    if not cfg.get("on"):
        return [_Path(groups=[[]], variation=cfg.get("offVariation"))]

    paths = [
        _Path(
            groups=[[_person("distinct_id", "exact", keys)]],
            variation=v,
            is_target=True,
        )
        for v, keys in _targets(cfg, notes).items()
    ]
    for i, rule in enumerate(cfg.get("rules") or []):
        if rule.get("variation") is None:
            raise Unmappable(f"rule {i} serves a percentage rollout")
        clauses = rule.get("clauses") or []
        groups = _rule_groups(clauses, cohorts) if _reachable(clauses, notes) else []
        if groups:
            paths.append(_Path(groups=groups, variation=rule["variation"]))
    fallthrough = cfg.get("fallthrough") or {}
    if fallthrough.get("variation") is None:
        raise Unmappable("fallthrough serves a percentage rollout")
    paths.append(_Path(groups=[[]], variation=fallthrough["variation"]))
    return paths


def _rule_groups(
    clauses: list[dict[str, Any]], cohorts: dict[str, MappedCohort]
) -> list[list[dict[str, Any]]]:
    """A rule's clauses as condition groups: AND within, a segment list fans out into OR."""
    alternatives: list[list[list[dict[str, Any]]]] = []
    for clause in clauses:
        if clause.get("op") != "segmentMatch":
            alternatives.append([_clause_properties(clause)])
            continue
        if clause.get("negate"):
            raise Unmappable("negated segment match")
        options = []
        for seg in clause.get("values") or []:
            cohort = cohorts.get(seg)
            if cohort is None:
                raise Unmappable(f"references segment `{seg}`, which the export lacks")
            if cohort.decisions:
                raise Unmappable(f"references segment `{seg}`, which needs a decision")
            if cohort.payload is not None:
                options.append(
                    [{"key": "id", "type": "cohort", "value": cohort_name(seg)}]
                )
        # A rule naming only empty segments matches nobody.
        if not options:
            return []
        alternatives.append(options)
    return [
        [prop for option in combo for prop in option]
        for combo in itertools.product(*alternatives)
    ]


def _clause_properties(clause: dict[str, Any]) -> list[dict[str, Any]]:
    """One LaunchDarkly clause as PostHog properties, all of which must hold.

    A negated LaunchDarkly clause never matches a context that lacks the
    attribute: "country is not one of IN" serves nothing to a visitor with no
    country. PostHog's negated operators promise nothing about a missing
    property, so the property is required to be set as well -- otherwise the
    port would widen who is served exactly where the attribute is unknown.
    """
    prop = _clause_property(clause)
    if clause.get("negate"):
        return [_person(prop["key"], "is_set", "is_set"), prop]
    return [prop]


def _clause_property(clause: dict[str, Any]) -> dict[str, Any]:
    attribute = ATTRIBUTE_ALIASES.get(clause["attribute"], clause["attribute"])
    if attribute not in PERSON_PROPERTIES | {"distinct_id"}:
        raise Unmappable(
            f"clause on `{clause['attribute']}`, which the backend never sends"
        )
    op, values, negate = (
        clause["op"],
        clause.get("values") or [],
        bool(clause.get("negate")),
    )

    if op == "in":
        return _person(attribute, "is_not" if negate else "exact", values)
    if op in _REGEX_OPS or op == "matches":
        parts = (
            values
            if op == "matches"
            else [_REGEX_OPS[op].format(re.escape(str(v))) for v in values]
        )
        pattern = parts[0] if len(parts) == 1 else "|".join(f"(?:{p})" for p in parts)
        return _person(attribute, "not_regex" if negate else "regex", pattern)
    if negate:
        raise Unmappable(f"negated `{op}` clause")
    if op in ("after", "before"):
        # LaunchDarkly matches ANY listed date: after the earliest, before the latest.
        dates = [_iso_date(v) for v in values]
        return _person(
            attribute,
            f"is_date_{op}",
            (min if op == "after" else max)(dates).isoformat(),
        )
    if op in _NUMERIC_OPS and len(values) == 1:
        return _person(attribute, _NUMERIC_OPS[op], values[0])
    raise Unmappable(f"`{op}` clause on `{clause['attribute']}`")


def _person(key: str, operator: str, value: Any) -> dict[str, Any]:
    return {"key": key, "type": "person", "operator": operator, "value": value}


def _iso_date(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _boolean_filters(
    flag: dict[str, Any], paths: list[_Path]
) -> tuple[dict[str, Any], bool]:
    """A boolean flag serves true on any matching group, so a path serving false
    can only be expressed ahead of a true path as a user-id exclusion."""
    values = [v["value"] for v in flag["variations"]]
    exclusions: list[str] = []
    groups: list[dict[str, Any]] = []
    for i, path in enumerate(paths):
        serves = values[path.variation] if path.variation is not None else None
        if serves is True:
            for g in path.groups:
                props = g + (
                    [_person("distinct_id", "is_not", exclusions)] if exclusions else []
                )
                groups.append(
                    {"properties": props, "rollout_percentage": 100, "variant": None}
                )
            continue
        later_true = any(
            p.variation is not None and values[p.variation] is True
            for p in paths[i + 1 :]
        )
        if not later_true:
            continue
        if not path.is_target:
            raise Unmappable("a rule serves false ahead of a rule serving true")
        exclusions += path.groups[0][0]["value"]
    filters = {"groups": groups, "multivariate": None, "payloads": {}}
    return filters, bool(groups)


def _multivariate_filters(
    flag: dict[str, Any], paths: list[_Path]
) -> tuple[dict[str, Any], bool]:
    """Every path becomes groups overriding to its variant; the payload is LaunchDarkly's value."""
    variations = flag["variations"]
    keys = _variant_keys(variations)
    if paths[0].groups == [[]] and paths[0].variation is None:
        raise Unmappable(
            "off with no off variation: LaunchDarkly returns the caller's default, "
            "PostHog would return false"
        )
    groups: list[dict[str, Any]] = []
    for path in paths:
        if path.variation is None:
            raise Unmappable("a path serves no variation")
        for g in path.groups:
            groups.append(
                {
                    "properties": g,
                    "rollout_percentage": 100,
                    "variant": keys[path.variation],
                }
            )
    filters = {
        "groups": groups,
        "multivariate": {
            "variants": [
                {
                    "key": k,
                    "name": v.get("name") or None,
                    "rollout_percentage": 100 if i == 0 else 0,
                }
                for i, (k, v) in enumerate(zip(keys, variations))
            ]
        },
        "payloads": {k: json.dumps(v["value"]) for k, v in zip(keys, variations)},
    }
    return filters, True


def _variant_keys(variations: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for i, v in enumerate(variations):
        source = v.get("name") or (v["value"] if isinstance(v["value"], str) else "")
        slug = re.sub(r"[^a-z0-9_-]+", "-", str(source).lower()).strip("-")[:40]
        key = candidate = slug or f"variation-{i}"
        suffix = i
        while candidate in keys:
            candidate = f"{key}-{suffix}"
            suffix += 1
        keys.append(candidate)
    return keys


def _change(
    kind: Literal["cohort", "flag"],
    key: str,
    payload: dict[str, Any],
    existing: dict[str, Any] | None,
) -> Change:
    if existing is None:
        return Change(
            kind=kind,
            key=key,
            action="create",
            payload=payload,
            diff=_render(kind, payload),
        )
    diff: list[str] = []
    for field, desired in payload.items():
        if field == "filters":
            if _canonical(kind, existing) == _canonical(kind, payload):
                continue
            before, after = _render(kind, existing), _render(kind, payload)
            lines = [
                line
                for line in difflib.unified_diff(before, after, lineterm="", n=0)
                if not line.startswith(("---", "+++", "@@"))
            ]
            diff += lines or ["filters: changed (only in redacted values)"]
        elif field != "key" and existing.get(field) != desired:
            diff.append(f"{field}: {existing.get(field)!r} -> {desired!r}")
    return Change(
        kind=kind,
        key=key,
        action="update" if diff else "unchanged",
        payload=payload,
        diff=diff,
        existing_id=existing.get("id"),
    )


def _canonical(kind: str, payload: dict[str, Any]) -> Any:
    filters = payload.get("filters") or {}
    if kind == "flag":
        return _canonical_filters(filters)
    return [
        [_canonical_property(p) for p in g.get("values") or []]
        for g in (filters.get("properties") or {}).get("values") or []
    ]


def _render(kind: str, payload: dict[str, Any]) -> list[str]:
    filters = (
        _canonical_filters(payload.get("filters") or {})
        if kind == "flag"
        else payload.get("filters") or {}
    )
    return describe_filters(filters) if kind == "flag" else describe_cohort(filters)


def _canonical_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """Drop the defaults PostHog fills in, so a round-tripped flag compares equal."""
    return {
        "groups": [
            {
                "properties": [
                    _canonical_property(p) for p in g.get("properties") or []
                ],
                "rollout_percentage": _rollout(g),
                "variant": g.get("variant"),
            }
            for g in filters.get("groups") or []
        ],
        "multivariate": filters.get("multivariate"),
        "payloads": filters.get("payloads") or {},
    }


def _canonical_property(prop: dict[str, Any]) -> dict[str, Any]:
    return {k: prop.get(k) for k in ("key", "type", "operator", "value")}


def _rollout(group: dict[str, Any]) -> int:
    rollout = group.get("rollout_percentage")
    return 100 if rollout is None else rollout


def _canonical_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload
    return payload


def _short(value: Any, limit: int = 80) -> str:
    text = _redact(json.dumps(value, sort_keys=True))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _redact(text: str) -> str:
    """Payloads and patterns can carry addresses and user ids; the output never does."""
    text = _LOCAL_PART.sub("<email>@", _EMAIL.sub("<email>", text))
    return _UUID.sub("<user-id>", text)
