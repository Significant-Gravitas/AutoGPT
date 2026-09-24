"""Re-create LaunchDarkly's feature flags and segments in a PostHog project.

Reads one LaunchDarkly environment's flags with their targeting, maps them
with ``scripts/feature_flag_sync.py`` (segments become dynamic cohorts,
individual targets ``distinct_id`` conditions, rules condition groups, JSON
variations payloads) and prints a plan against what the PostHog project
already holds. Dry run by default::

    LAUNCHDARKLY_API_TOKEN=… POSTHOG_PERSONAL_API_KEY=… \\
        poetry run python -m scripts.sync_feature_flags_to_posthog --ld-env test

``--apply --project <id>`` performs the plan: creates and updates by key,
never deletes. It is refused without an explicit project id. Flags or
cohorts listed under "needs a decision" are never written.

The output counts email and user-id values instead of printing them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from scripts.feature_flag_sync import (
    Change,
    MappedCohort,
    MappedFlag,
    describe_cohort,
    describe_filters,
    map_flag,
    map_segment,
    plan_sync,
    resolve_cohort_refs,
)

LD_API = "https://app.launchdarkly.com/api/v2"
DEFAULT_POSTHOG_HOST = "https://eu.posthog.com"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--ld-env",
        required=True,
        help="LaunchDarkly environment: test (Dev) or production",
    )
    parser.add_argument("--ld-project", default="default")
    parser.add_argument(
        "--project",
        type=int,
        help="PostHog project id (default for a dry run: the key's own)",
    )
    parser.add_argument("--posthog-host", default=DEFAULT_POSTHOG_HOST)
    parser.add_argument(
        "--flag", action="append", default=[], help="limit to these LaunchDarkly keys"
    )
    parser.add_argument(
        "--apply", action="store_true", help="perform the plan (requires --project)"
    )
    args = parser.parse_args(argv)
    if args.apply and args.project is None:
        parser.error("--apply writes to PostHog and needs an explicit --project <id>")

    ld = _Client(LD_API, {"Authorization": _env("LAUNCHDARKLY_API_TOKEN")})
    posthog = _Client(
        args.posthog_host,
        {"Authorization": f"Bearer {_env('POSTHOG_PERSONAL_API_KEY')}"},
    )

    flags = fetch_ld_flags(ld, args.ld_project, args.ld_env, set(args.flag))
    cohorts = [
        map_segment(s, args.ld_env)
        for s in fetch_ld_segments(ld, args.ld_project, args.ld_env, flags)
    ]
    by_segment = {c.segment_key: c for c in cohorts}
    mapped = [map_flag(f, args.ld_env, by_segment) for f in flags]

    project = posthog.get(f"/api/projects/{args.project or '@current'}/")
    project_path = f"/api/projects/{project['id']}"
    existing_flags = posthog.paginate(f"{project_path}/feature_flags/?limit=100")
    existing_cohorts = posthog.paginate(f"{project_path}/cohorts/?limit=100")
    changes = plan_sync(cohorts, mapped, existing_cohorts, existing_flags)

    print(
        f"LaunchDarkly `{args.ld_project}`/{args.ld_env} -> PostHog project {project['id']} ({project['name']})\n"
    )
    print_mapping(mapped, cohorts)
    print_plan(
        changes,
        {f["key"] for f in existing_flags if not f.get("deleted")}
        - {m.key for m in mapped},
    )
    if not args.apply:
        print(
            "\nDry run: nothing was written. Re-run with --apply --project <id> to perform it."
        )
        return 0
    apply_plan(posthog, project_path, changes)
    return 0


def fetch_ld_flags(
    ld: _Client, project: str, env: str, only: set[str]
) -> list[dict[str, Any]]:
    """Every non-archived flag with *env*'s full configuration (``summary=0``)."""
    flags: list[dict[str, Any]] = []
    while True:
        page = ld.get(
            f"/flags/{project}?env={env}&summary=0&limit=100&offset={len(flags)}"
        )
        flags += page["items"]
        if not page["items"] or len(flags) >= page["totalCount"]:
            break
    return [
        f for f in flags if not f.get("archived") and (not only or f["key"] in only)
    ]


def fetch_ld_segments(
    ld: _Client, project: str, env: str, flags: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Each segment a flag's rule references, fetched whole: a listing may truncate member lists."""
    keys = sorted(
        {
            s
            for f in flags
            for r in f["environments"][env].get("rules") or []
            for c in r.get("clauses") or []
            if c.get("op") == "segmentMatch"
            for s in c.get("values") or []
        }
    )
    segments = []
    for key in keys:
        try:
            segments.append(
                ld.get(f"/segments/{project}/{env}/{urllib.parse.quote(key)}")
            )
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    return segments


def print_mapping(flags: list[MappedFlag], cohorts: list[MappedCohort]) -> None:
    print("## Cohorts (one per referenced LaunchDarkly segment)")
    for c in cohorts:
        if c.payload is None:
            print(
                f"- {c.segment_key}: "
                + (
                    "needs a decision"
                    if c.decisions
                    else "no members; conditions on it match nobody"
                )
            )
            continue
        print(f"- {c.segment_key} -> cohort `{c.payload['name']}`")
        for line in describe_cohort(c.payload["filters"]) + c.notes:
            print(f"    {line}")

    targeted = [f for f in flags if f.targeted]
    print(f"\n## Flags with targeting ({len(targeted)} of {len(flags)})")
    for f in targeted:
        print(
            f"- {f.key} [{f.kind}] targets={f.targets} segments={','.join(f.segments) or '-'} rules={f.rules}"
        )
        for note in f.notes:
            print(f"    note: {note}")
        if f.payload is None:
            print("    needs a decision")
            continue
        print(
            f"    active={f.payload['active']} runtime={f.payload['evaluation_runtime']}"
        )
        for line in describe_filters(f.payload["filters"]):
            print(f"    {line}")

    decisions = [d for x in [*cohorts, *flags] for d in x.decisions]
    print(f"\n## Needs a decision ({len(decisions)}; never written)")
    for d in decisions:
        print(f"- {d.subject}: {d.reason}")


def print_plan(changes: list[Change], untouched: set[str]) -> None:
    counts: dict[str, int] = {}
    for c in changes:
        counts[f"{c.kind} {c.action}"] = counts.get(f"{c.kind} {c.action}", 0) + 1
    print("\n## Plan")
    print("  " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    for c in changes:
        if c.action == "unchanged":
            continue
        print(f"- {c.kind} {c.key}: {c.action}")
        for line in c.diff:
            print(f"    {line}")
    if untouched:
        print(f"\nIn PostHog only, left alone: {', '.join(sorted(untouched))}")


def apply_plan(posthog: _Client, project_path: str, changes: list[Change]) -> None:
    """Cohorts first, so flags can reference their ids; creates and updates only."""
    cohort_ids: dict[str, int] = {}
    for c in changes:
        if c.kind != "cohort" or c.payload is None:
            continue
        if c.action == "create":
            cohort_ids[c.key] = posthog.send(
                "POST", f"{project_path}/cohorts/", c.payload
            )["id"]
        elif c.existing_id is not None:
            if c.action == "update":
                posthog.send(
                    "PATCH", f"{project_path}/cohorts/{c.existing_id}/", c.payload
                )
            cohort_ids[c.key] = c.existing_id
        print(f"cohort {c.key}: {c.action}")
    for c in changes:
        if (
            c.kind != "flag"
            or c.action not in ("create", "update")
            or c.payload is None
        ):
            continue
        payload = {
            **c.payload,
            "filters": resolve_cohort_refs(c.payload["filters"], cohort_ids),
        }
        if c.action == "create":
            posthog.send("POST", f"{project_path}/feature_flags/", payload)
        else:
            posthog.send(
                "PATCH", f"{project_path}/feature_flags/{c.existing_id}/", payload
            )
        print(f"flag {c.key}: {c.action}")


class _Client:
    def __init__(self, base: str, headers: dict[str, str]):
        self.base = base.rstrip("/")
        self.headers = headers

    def get(self, path_or_url: str) -> Any:
        return self.send("GET", path_or_url)

    def paginate(self, path: str) -> list[dict[str, Any]]:
        """Every page of a PostHog listing, following ``next`` until it is absent."""
        results: list[dict[str, Any]] = []
        url: str | None = path
        while url:
            page = self.get(url)
            results += page["results"]
            url = page.get("next")
        return results

    def send(self, method: str, path_or_url: str, body: Any = None) -> Any:
        url = path_or_url if path_or_url.startswith("http") else self.base + path_or_url
        request = urllib.request.Request(
            url,
            method=method,
            headers={**self.headers, "Content-Type": "application/json"},
            data=json.dumps(body).encode() if body is not None else None,
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"{name} is not set")
    return value


if __name__ == "__main__":
    sys.exit(main())
