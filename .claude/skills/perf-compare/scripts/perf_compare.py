#!/usr/bin/env python3
"""Compare one numeric span field between a baseline and a candidate in Sentry.

This is the shared measurement recipe for AutoGPT performance work. It reads
transactions from Sentry's Events API (dataset ``spans``) for two selectors,
reports count, p50, p95, p99 and failure rate per transaction, then pulls a
bounded random sample of raw values per side and bootstraps a confidence
interval for the p50 and p95 difference. The interval decides the verdict.
``SKILL.md`` one directory up explains when to use it and how to read the
result.

Standard library only; Python 3.8 or newer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_ORG = "significant-gravitas"
DEFAULT_REGION_URL = "https://us.sentry.io"
DEFAULT_FIELD = "span.duration"
DEFAULT_PERIOD = "7d"
SCHEMA_VERSION = 1
USER_AGENT = "autogpt-perf-compare/1"
PAGE_SIZE = 100
MAX_PAGES = 200
MAX_SAMPLE_CAP = PAGE_SIZE * MAX_PAGES
OVERVIEW_ROWS = 15
RESAMPLE_CAP = 50000
TARGET_TAIL_DRAWS = 20
MIN_TAIL_DRAWS = 10
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

FASTER = "faster"
SLOWER = "slower"
NO_CHANGE = "no detectable change"
NOT_ENOUGH_DATA = "not enough data"
MORE_FAILURES = "more failures"
HIGHER = "higher"
LOWER = "lower"

EXIT_FASTER = 0
EXIT_SLOWER = 1
EXIT_USAGE = 2
EXIT_NO_CHANGE = 3
EXIT_NOT_ENOUGH_DATA = 4
EXIT_AUTH = 5
EXIT_API = 6

EXIT_CODES_HELP = """\
exit codes:
  0  faster: a compared transaction got faster and nothing regressed
  1  slower or more failures: at least one compared transaction regressed
  2  usage error: bad arguments, unknown project, or Sentry rejected the query
  3  no detectable change: nothing moved beyond the noise
  4  not enough data: no compared transaction met --min-count on both sides
  5  auth: SENTRY_AUTH_TOKEN missing, or Sentry rejected it (401/403)
  6  Sentry API or network error after retries, or any unexpected error
"""

TOKEN_SCOPES = "org:read  project:read  event:read"

MISSING_TOKEN_MESSAGE = f"""\
perf-compare: SENTRY_AUTH_TOKEN is not set.

This tool reads Sentry through its REST API and takes the token only from the
SENTRY_AUTH_TOKEN environment variable; it never looks for tokens on disk.
Use a read-only token with exactly these scopes:

    {TOKEN_SCOPES}

Either a user auth token (Sentry > Settings > Account > API > Auth Tokens) or
an internal integration with Read on Organization, Project and Issue & Event.
Organization auth tokens (sntrys_...) are locked to org:ci and cannot read
events. Then:

    export SENTRY_AUTH_TOKEN=<token>
"""

SIDE_KEYS = ("project", "env", "release", "query", "window", "start", "end", "period")
FIELD_PATTERN = re.compile(r"^[A-Za-z0-9_.\[\],:-]+$")
PERIOD_PATTERN = re.compile(r"^(\d+)([mhdw])$")
PERIOD_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
LINK_PATTERN = re.compile(r"<([^>]*)>((?:\s*;\s*[A-Za-z_]+=\"[^\"]*\")*)")
LINK_ATTR_PATTERN = re.compile(r"([A-Za-z_]+)=\"([^\"]*)\"")
UNIT_SUFFIX = {
    "nanosecond": "ns",
    "microsecond": "us",
    "millisecond": "ms",
    "second": "s",
    "byte": "B",
}

Transport = Callable[[str, Dict[str, str], float], Tuple[int, Dict[str, str], bytes]]


class PerfCompareError(Exception):
    exit_code = EXIT_API
    kind = "api"


class UsageError(PerfCompareError):
    exit_code = EXIT_USAGE
    kind = "usage"


class AuthError(PerfCompareError):
    exit_code = EXIT_AUTH
    kind = "auth"


class SentryApiError(PerfCompareError):
    exit_code = EXIT_API
    kind = "api"


class NotFoundError(SentryApiError):
    pass


# Time windows ---------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime

    @property
    def seconds(self) -> int:
        return int((self.end - self.start).total_seconds())

    def as_json(self) -> List[str]:
        return [format_filter_time(self.start), format_filter_time(self.end)]


def parse_period(text: str) -> timedelta:
    match = PERIOD_PATTERN.match(text.strip())
    if not match or int(match.group(1)) == 0:
        raise UsageError(
            f"bad period {text!r}: use a number and m, h, d or w, e.g. 24h or 7d"
        )
    return timedelta(**{PERIOD_UNITS[match.group(2)]: int(match.group(1))})


def parse_time(text: str) -> datetime:
    value = text.strip()
    if value[-1:] in ("Z", "z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise UsageError(
            f"bad time {text!r}: use ISO 8601 in UTC, e.g. 2026-10-11T14:00:00Z"
        ) from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def parse_window(text: str) -> Window:
    parts = text.split("/")
    if len(parts) != 2:
        raise UsageError(
            f"bad window {text!r}: use START/END, "
            "e.g. 2026-10-11T14:00:00Z/2026-10-11T14:30:00Z"
        )
    return Window(parse_time(parts[0]), parse_time(parts[1]))


def format_param_time(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def format_filter_time(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_windows(windows: Sequence[Window], label: str) -> List[Window]:
    ordered = sorted(windows, key=lambda window: window.start)
    for window in ordered:
        if window.end <= window.start:
            raise UsageError(
                f"{label}: window {format_filter_time(window.start)}/"
                f"{format_filter_time(window.end)} does not end after it starts"
            )
    for index in range(1, len(ordered)):
        if ordered[index].start < ordered[index - 1].end:
            raise UsageError(f"{label}: windows overlap; give each event one window")
    return ordered


def resolve_windows(
    label: str,
    windows: Sequence[str],
    start: Optional[str],
    end: Optional[str],
    period: Optional[str],
    now: datetime,
) -> Optional[List[Window]]:
    if not (windows or start or end or period):
        return None
    if (windows and (start or end or period)) or (period and (start or end)):
        raise UsageError(
            f"{label}: use one of window=, start=/end= or period=, not several"
        )
    if windows:
        resolved = [parse_window(window) for window in windows]
    elif period:
        resolved = [Window(now - parse_period(period), now)]
    else:
        if not start:
            raise UsageError(f"{label}: end needs a start")
        resolved = [Window(parse_time(start), parse_time(end) if end else now)]
    return check_windows(resolved, label)


# Selectors ------------------------------------------------------------------


@dataclass
class Shared:
    project: Optional[str]
    env: Optional[str]
    query: Optional[str]
    windows: List[Window]


@dataclass
class Side:
    label: str
    project: str
    environment: Optional[str]
    release: Optional[str]
    query: Optional[str]
    windows: List[Window]

    def selector(self) -> Dict[str, object]:
        return {
            "project": self.project,
            "environment": self.environment,
            "release": self.release,
            "query": self.query,
            "windows": [window.as_json() for window in self.windows],
        }


def parse_side(
    label: str, tokens: Sequence[str], shared: Shared, now: datetime
) -> Side:
    values: Dict[str, str] = {}
    windows: List[str] = []
    for token in tokens:
        key, separator, value = token.partition("=")
        key = key.strip().lower()
        key = "env" if key == "environment" else key
        if not separator or key not in SIDE_KEYS:
            raise UsageError(
                f"--{label} takes KEY=VALUE pairs with KEY one of "
                f"{', '.join(SIDE_KEYS)}; got {token!r}"
            )
        if key == "window":
            windows.append(value)
            continue
        if key in values:
            raise UsageError(f"--{label} sets {key}= twice")
        values[key] = value
    project = values.get("project") or shared.project
    if not project:
        raise UsageError(
            f"the {label} has no project: pass --project SLUG or "
            f"--{label} project=SLUG"
        )
    side_windows = resolve_windows(
        label,
        windows,
        values.get("start"),
        values.get("end"),
        values.get("period"),
        now,
    )
    return Side(
        label=label,
        project=project,
        environment=values.get("env", shared.env) or None,
        release=values.get("release") or None,
        query=values.get("query", shared.query) or None,
        windows=side_windows if side_windows is not None else list(shared.windows),
    )


# Query building -------------------------------------------------------------


def quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def window_clause(windows: Sequence[Window]) -> str:
    terms = [
        f"timestamp:>={format_filter_time(window.start)} "
        f"timestamp:<{format_filter_time(window.end)}"
        for window in windows
    ]
    return terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"


def transaction_clause(names: Sequence[str]) -> Optional[str]:
    terms = [f"transaction:{quote(name)}" for name in names]
    if not terms:
        return None
    return terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"


def build_query(
    side: Side,
    field_name: str,
    transactions: Sequence[str] = (),
    windows: Optional[Sequence[Window]] = None,
) -> str:
    """The Sentry search string for one side; valid for REST and the MCP."""
    parts = ["is_transaction:true"]
    if side.environment:
        parts.append(f"environment:{quote(side.environment)}")
    if side.release:
        parts.append(f"release:{quote(side.release)}")
    if field_name != DEFAULT_FIELD:
        parts.append(f"has:{field_name}")
    if side.query:
        parts.append(f"({side.query})")
    names = transaction_clause(transactions)
    if names:
        parts.append(names)
    parts.append(window_clause(side.windows if windows is None else windows))
    return " ".join(parts)


def aggregate_fields(field_name: str) -> List[str]:
    return [
        "transaction",
        "count()",
        f"p50({field_name})",
        f"p95({field_name})",
        f"p99({field_name})",
        "failure_rate()",
    ]


def sample_fields(field_name: str) -> List[str]:
    return [field_name, "trace", "id"]


SAMPLE_SORT = "id"


def events_params(
    project_id: str,
    query: str,
    fields: Sequence[str],
    sort: str,
    windows: Sequence[Window],
    per_page: int,
) -> List[Tuple[str, str]]:
    params = [("dataset", "spans"), ("project", str(project_id))]
    params.extend(("field", name) for name in fields)
    params.extend(
        [
            ("query", query),
            ("sort", sort),
            ("per_page", str(per_page)),
            ("start", format_param_time(min(window.start for window in windows))),
            ("end", format_param_time(max(window.end for window in windows))),
        ]
    )
    return params


def mcp_period(windows: Sequence[Window], now: datetime) -> str:
    earliest = min(window.start for window in windows)
    return f"{max(1, math.ceil((now - earliest).total_seconds() / 3600)) + 1}h"


def mcp_arguments(
    org: str,
    region_url: str,
    side: Side,
    query: str,
    fields: Sequence[str],
    sort: str,
    limit: int,
    now: datetime,
) -> Dict[str, object]:
    return {
        "organizationSlug": org,
        "regionUrl": region_url,
        "projectSlug": side.project,
        "dataset": "spans",
        "query": query,
        "fields": list(fields),
        "sort": sort,
        "period": mcp_period(side.windows, now),
        "limit": limit,
    }


# Sentry client --------------------------------------------------------------


def urllib_transport(
    url: str, headers: Dict[str, str], timeout: float
) -> Tuple[int, Dict[str, str], bytes]:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_headers = {k.lower(): v for k, v in response.headers.items()}
            return response.status, response_headers, response.read()
    except urllib.error.HTTPError as error:
        error_headers = {k.lower(): v for k, v in (error.headers or {}).items()}
        return error.code, error_headers, error.read()


def next_cursor(link_header: str) -> Optional[str]:
    for match in LINK_PATTERN.finditer(link_header or ""):
        attributes = dict(LINK_ATTR_PATTERN.findall(match.group(2)))
        if attributes.get("rel") == "next" and attributes.get("results") == "true":
            return attributes.get("cursor") or None
    return None


def error_detail(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    try:
        detail = json.loads(text).get("detail")
    except (ValueError, AttributeError):
        detail = None
    return str(detail if detail is not None else text).strip()[:400]


class SentryClient:
    def __init__(
        self,
        token: str,
        org: str = DEFAULT_ORG,
        region_url: str = DEFAULT_REGION_URL,
        transport: Optional[Transport] = None,
        timeout: float = 30.0,
        retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.token = token
        self.org = org
        self.region_url = region_url.rstrip("/")
        self.transport = transport or urllib_transport
        self.timeout = timeout
        self.retries = retries
        self.sleep = sleep
        self.requests = 0
        self._lock = threading.Lock()
        self._project_ids: Dict[str, str] = {}

    def get(
        self, path: str, params: Sequence[Tuple[str, str]] = ()
    ) -> Tuple[object, Dict[str, str]]:
        url = self.region_url + path
        if params:
            url += "?" + urllib.parse.urlencode(list(params))
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        last_problem = ""
        for attempt in range(self.retries + 1):
            with self._lock:
                self.requests += 1
            retry_after: Optional[str] = None
            try:
                status, response_headers, body = self.transport(
                    url, headers, self.timeout
                )
            except (OSError, http.client.HTTPException) as error:
                last_problem = f"network error: {error}"
            else:
                if status == 200:
                    try:
                        return json.loads(body.decode("utf-8")), response_headers
                    except ValueError:
                        raise SentryApiError(
                            f"Sentry returned invalid JSON for {path}"
                        ) from None
                detail = error_detail(body)
                if status in (401, 403):
                    raise AuthError(
                        f"Sentry rejected SENTRY_AUTH_TOKEN (HTTP {status}: {detail}). "
                        f"The token needs {TOKEN_SCOPES.replace('  ', ', ')}; "
                        "organization auth tokens (sntrys_...) only carry org:ci, so "
                        "use a user auth token or an internal integration token."
                    )
                if status == 400:
                    raise UsageError(
                        f"Sentry rejected the request (HTTP 400): {detail}"
                    )
                if status == 404:
                    raise NotFoundError(f"HTTP 404 from {path}: {detail}")
                if status != 429 and status < 500:
                    raise SentryApiError(f"HTTP {status} from {path}: {detail}")
                last_problem = f"HTTP {status}: {detail}"
                retry_after = response_headers.get("retry-after")
            if attempt < self.retries:
                self.sleep(backoff_seconds(attempt, retry_after))
        raise SentryApiError(
            f"giving up on {path} after {self.retries + 1} attempts: {last_problem}"
        )

    def project_id(self, project: str) -> str:
        if project.isdigit():
            return project
        if project not in self._project_ids:
            path = "/api/0/projects/{}/{}/".format(
                urllib.parse.quote(self.org, safe=""),
                urllib.parse.quote(project, safe=""),
            )
            try:
                data, _ = self.get(path)
            except NotFoundError:
                raise UsageError(
                    f"unknown Sentry project {project!r} in org {self.org!r}"
                ) from None
            if not isinstance(data, dict) or "id" not in data:
                raise SentryApiError(f"unexpected project response for {project!r}")
            self._project_ids[project] = str(data["id"])
        return self._project_ids[project]

    def events(
        self, params: Sequence[Tuple[str, str]]
    ) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, str]]:
        path = "/api/0/organizations/{}/events/".format(
            urllib.parse.quote(self.org, safe="")
        )
        data, headers = self.get(path, params)
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise SentryApiError("unexpected events response: no data list")
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        return data["data"], meta, headers

    def paginate(
        self, params: Sequence[Tuple[str, str]], limit: int
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        cursor: Optional[str] = None
        for _ in range(MAX_PAGES):
            page_params = list(params) + ([("cursor", cursor)] if cursor else [])
            page, _, headers = self.events(page_params)
            rows.extend(page)
            cursor = next_cursor(headers.get("link", ""))
            if len(rows) >= limit or not page or not cursor:
                break
        return rows[:limit]


def backoff_seconds(attempt: int, retry_after: Optional[str]) -> float:
    if retry_after:
        try:
            return min(60.0, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return float(min(30, 2**attempt))


# Statistics -----------------------------------------------------------------


def percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not sorted_values:
        raise ValueError("percentile of an empty sample")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    position = (len(sorted_values) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (
        position - low
    )


def _drawer(
    clusters: Sequence[Sequence[float]], rng: random.Random
) -> Callable[[], List[float]]:
    count = len(clusters)
    if all(len(cluster) == 1 for cluster in clusters):
        flat = [cluster[0] for cluster in clusters]
        return lambda: rng.choices(flat, k=count)
    return lambda: [
        value for cluster in rng.choices(clusters, k=count) for value in cluster
    ]


def bootstrap_quantile_diffs(
    baseline: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    quantiles: Sequence[float],
    resamples: int,
    rng: random.Random,
) -> Dict[float, List[float]]:
    """Candidate minus baseline quantile, per resample.

    Each side is a list of clusters (all values from one trace). Resampling
    whole clusters keeps correlated spans of one request together, so the
    interval does not pretend a page that fired ten calls is ten
    independent observations.
    """
    draw_baseline = _drawer(baseline, rng)
    draw_candidate = _drawer(candidate, rng)
    diffs: Dict[float, List[float]] = {quantile: [] for quantile in quantiles}
    for _ in range(resamples):
        base_values = sorted(draw_baseline())
        cand_values = sorted(draw_candidate())
        for quantile in quantiles:
            diffs[quantile].append(
                percentile(cand_values, quantile) - percentile(base_values, quantile)
            )
    return diffs


def confidence_interval(
    samples: Sequence[float], confidence: float
) -> Tuple[float, float]:
    ordered = sorted(samples)
    alpha = 1.0 - confidence
    return percentile(ordered, alpha / 2), percentile(ordered, 1 - alpha / 2)


def relative(diff: Optional[float], base: Optional[float]) -> Optional[float]:
    if diff is None or base is None or base == 0:
        return None
    return diff / base


# Verdicts -------------------------------------------------------------------


@dataclass
class Options:
    field: str = DEFAULT_FIELD
    transactions: List[str] = dataclass_field(default_factory=list)
    top: int = 5
    sample_cap: int = 1000
    min_count: int = 200
    min_traces: int = 30
    min_effect: float = 0.05
    min_failure_delta: float = 0.001
    confidence: float = 0.95
    resamples: int = 4000
    seed: int = 1234
    concurrency: int = 4


def adjusted_confidence(confidence: float, family_size: int) -> float:
    """Per-interval level that keeps ``family_size`` intervals at ``confidence``.

    Bonferroni: comparing ten percentiles at 95% each would call one of them
    a change by luck about 40% of the time; this holds the whole set at 95%.
    """
    return 1 - (1 - confidence) / max(1, family_size)


def tail_share(level: float) -> float:
    return (1 - level) / 2


def required_resamples(level: float) -> int:
    """Resamples that leave TARGET_TAIL_DRAWS draws in each tail at ``level``."""
    return min(RESAMPLE_CAP, math.ceil(TARGET_TAIL_DRAWS / tail_share(level) - 1e-9))


def effective_resamples(requested: int, level: float) -> int:
    return max(requested, required_resamples(level))


def sample_shortfall(
    baseline: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    options: Options,
) -> Optional[str]:
    """Why the samples cannot support a verdict, or None when they can.

    Spans from one trace move together, so the bootstrap learns nothing about
    variation between requests from a side made of a handful of traces, no
    matter how many spans those traces hold. Both counts have to clear.
    """
    spans = (sample_size(baseline), sample_size(candidate))
    traces = (len(baseline), len(candidate))
    if min(spans) < options.min_count:
        return (
            f"needs {options.min_count} samples per side, has "
            f"{spans[0]} / {spans[1]}"
        )
    if min(traces) < options.min_traces:
        return (
            f"needs {options.min_traces} traces per side, has "
            f"{traces[0]} / {traces[1]}"
        )
    return None


def quantile_verdict(
    ci_low: float,
    ci_high: float,
    effects: Sequence[Optional[float]],
    min_effect: float,
) -> str:
    """Faster or slower needs the interval to exclude zero and every known
    relative effect (sample and Sentry's all-events percentile) to clear
    ``min_effect`` in that direction."""
    known = [effect for effect in effects if effect is not None]
    if ci_high < 0 and all(effect <= -min_effect for effect in known):
        return FASTER
    if ci_low > 0 and all(effect >= min_effect for effect in known):
        return SLOWER
    return NO_CHANGE


def combine_verdicts(verdicts: Sequence[str]) -> str:
    if not verdicts or all(verdict == NOT_ENOUGH_DATA for verdict in verdicts):
        return NOT_ENOUGH_DATA
    if SLOWER in verdicts:
        return SLOWER
    if FASTER in verdicts:
        return FASTER
    return NO_CHANGE


QUANTILES = (("p50", 0.5), ("p95", 0.95))


def sample_size(clusters: Sequence[Sequence[float]]) -> int:
    return sum(len(cluster) for cluster in clusters)


def compare_quantiles(
    baseline: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    rng: random.Random,
    options: Options,
    family_size: int = 2,
    sentry: Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]] = (
        None,
        None,
    ),
) -> Dict[str, Dict[str, object]]:
    """p50 and p95 of each side, the bootstrap intervals, and a verdict each.

    ``sentry`` holds each side's Sentry aggregate (percentiles over every
    matching event); its relative change has to agree with the sample's.
    """
    base_values = sorted(value for cluster in baseline for value in cluster)
    cand_values = sorted(value for cluster in candidate for value in cluster)
    enough = sample_shortfall(baseline, candidate, options) is None
    diffs = (
        bootstrap_quantile_diffs(
            baseline,
            candidate,
            [quantile for _, quantile in QUANTILES],
            options.resamples,
            rng,
        )
        if enough
        else None
    )
    level = adjusted_confidence(options.confidence, family_size)
    result: Dict[str, Dict[str, object]] = {}
    for name, quantile in QUANTILES:
        base = percentile(base_values, quantile) if base_values else None
        cand = percentile(cand_values, quantile) if cand_values else None
        diff = cand - base if base is not None and cand is not None else None
        rel = relative(diff, base)
        sentry_base = sentry[0].get(name) if sentry[0] else None
        sentry_cand = sentry[1].get(name) if sentry[1] else None
        sentry_rel = (
            relative(float(sentry_cand) - float(sentry_base), float(sentry_base))
            if sentry_base is not None and sentry_cand is not None
            else None
        )
        interval: Optional[Tuple[float, float]] = None
        adjusted: Optional[Tuple[float, float]] = None
        verdict = NOT_ENOUGH_DATA
        if diffs is not None:
            interval = confidence_interval(diffs[quantile], options.confidence)
            adjusted = confidence_interval(diffs[quantile], level)
            verdict = quantile_verdict(
                adjusted[0], adjusted[1], (rel, sentry_rel), options.min_effect
            )
        result[name] = {
            "baseline": base,
            "candidate": cand,
            "diff": diff,
            "rel": rel,
            "sentry_rel": sentry_rel,
            "ci": list(interval) if interval else None,
            "ci_adjusted": list(adjusted) if adjusted else None,
            "verdict": verdict,
        }
    return result


def log_comb(total: int, chosen: int) -> float:
    return (
        math.lgamma(total + 1)
        - math.lgamma(chosen + 1)
        - math.lgamma(total - chosen + 1)
    )


def fisher_exact_p(
    base_failures: int, base_total: int, cand_failures: int, cand_total: int
) -> float:
    """Two-sided Fisher exact p-value for the 2x2 failure table.

    Sums the hypergeometric probability of every table with the same margins
    that is no more likely than the observed one. Probabilities are computed
    in log space; ties use the usual 1e-7 relative tolerance. The walk starts
    at the most likely table and, since the distribution is unimodal, stops in
    each direction once tables are e^-40 times less likely than the observed
    one. Everything further out changes the result by under 1e-9 of itself,
    so a table with millions of failures still takes milliseconds.
    """
    total = base_total + cand_total
    failures = base_failures + cand_failures
    low = max(0, cand_total - (total - failures))
    high = min(failures, cand_total)

    def log_probability(count: int) -> float:
        return (
            log_comb(failures, count)
            + log_comb(total - failures, cand_total - count)
            - log_comb(total, cand_total)
        )

    observed = log_probability(cand_failures) + math.log1p(1e-7)
    cutoff = observed - 40
    mode = (cand_total + 1) * (failures + 1) // (total + 2)
    mode = min(high, max(low, mode))
    p_value = 0.0
    for counts in (range(mode, low - 1, -1), range(mode + 1, high + 1)):
        for count in counts:
            probability = log_probability(count)
            if probability <= observed:
                p_value += math.exp(probability)
            if probability < cutoff:
                break
    return min(1.0, p_value)


def compare_failure_rates(
    baseline: Optional[Dict[str, object]],
    candidate: Optional[Dict[str, object]],
    options: Options,
    family_size: int = 1,
) -> Dict[str, object]:
    """Fisher's exact test on each side's Sentry ``failure_rate()``.

    Exact rather than a normal approximation, which is wrong with a handful
    of failures (0 of 1000 against 4 of 1000 would pass a z-test). The
    significance level is Bonferroni-adjusted over ``family_size`` tests.
    """

    def split(aggregate: Optional[Dict[str, object]]) -> Optional[Tuple[int, int]]:
        if not aggregate or aggregate.get("failure_rate") is None:
            return None
        total = int(aggregate.get("count") or 0)
        return total, int(round(float(aggregate["failure_rate"]) * total))

    base, cand = split(baseline), split(candidate)
    result: Dict[str, object] = {
        "baseline": baseline.get("failure_rate") if baseline else None,
        "candidate": candidate.get("failure_rate") if candidate else None,
        "diff": None,
        "p_value": None,
        "verdict": NOT_ENOUGH_DATA,
    }
    if base is None or cand is None or min(base[0], cand[0]) < options.min_count:
        return result
    base_rate, cand_rate = base[1] / base[0], cand[1] / cand[0]
    p_value = fisher_exact_p(base[1], base[0], cand[1], cand[0])
    significant = p_value < (1 - options.confidence) / max(1, family_size)
    if significant and cand_rate - base_rate >= options.min_failure_delta:
        verdict = HIGHER
    elif significant and base_rate - cand_rate >= options.min_failure_delta:
        verdict = LOWER
    else:
        verdict = NO_CHANGE
    result.update(
        {
            "baseline": base_rate,
            "candidate": cand_rate,
            "diff": cand_rate - base_rate,
            "p_value": p_value,
            "verdict": verdict,
        }
    )
    return result


def overall_outcome(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    reasons: List[str] = []
    for result in results:
        for name, _ in QUANTILES:
            stat = result[name]
            if stat["verdict"] in (FASTER, SLOWER):
                reasons.append(
                    f"{result['transaction']}: {name} {stat['verdict']} "
                    f"(sample {format_rel(stat['rel'])}, "
                    f"Sentry {format_rel(stat['sentry_rel'])})"
                )
        failure = result["failure_rate"]
        if failure["verdict"] in (HIGHER, LOWER):
            reasons.append(
                f"{result['transaction']}: failure rate {failure['verdict']} "
                f"({format_points(failure['diff'])})"
            )
    verdicts = [result["verdict"] for result in results]
    failures = [result["failure_rate"]["verdict"] for result in results]
    if SLOWER in verdicts:
        verdict, code = SLOWER, EXIT_SLOWER
    elif HIGHER in failures:
        verdict, code = MORE_FAILURES, EXIT_SLOWER
    elif FASTER in verdicts:
        verdict, code = FASTER, EXIT_FASTER
    elif NO_CHANGE in verdicts:
        verdict, code = NO_CHANGE, EXIT_NO_CHANGE
    else:
        verdict, code = NOT_ENOUGH_DATA, EXIT_NOT_ENOUGH_DATA
    return {"verdict": verdict, "exit_code": code, "reasons": reasons}


# Comparison -----------------------------------------------------------------


def summarize_aggregate(row: Dict[str, object], field_name: str) -> Dict[str, object]:
    def number(key: str) -> Optional[float]:
        value = row.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return {
        "count": int(row.get("count()") or 0),
        "p50": number(f"p50({field_name})"),
        "p95": number(f"p95({field_name})"),
        "p99": number(f"p99({field_name})"),
        "failure_rate": number("failure_rate()"),
    }


def choose_transactions(
    baseline: Dict[str, Dict[str, object]],
    candidate: Dict[str, Dict[str, object]],
    options: Options,
) -> List[str]:
    if options.transactions:
        return list(dict.fromkeys(options.transactions))
    shared = [name for name in baseline if name in candidate]
    shared.sort(
        key=lambda name: (
            -min(int(baseline[name]["count"]), int(candidate[name]["count"])),
            name,
        )
    )
    return shared[: options.top]


def to_clusters(
    rows: Sequence[Dict[str, object]], field_name: str
) -> List[List[float]]:
    clusters: Dict[str, List[float]] = {}
    for index, row in enumerate(rows):
        value = row.get(field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value):
            continue
        key = str(row.get("trace") or f"row-{index}")
        clusters.setdefault(key, []).append(float(value))
    return list(clusters.values())


def fetch_sample(
    client: SentryClient,
    side: Side,
    project_id: str,
    field_name: str,
    name: str,
    cap: int,
) -> List[Dict[str, object]]:
    """The first ``cap`` matching spans in span-id order.

    Span ids are random, so this is a uniform random sample of the side's
    windows: it follows the traffic profile Sentry's own percentiles use and
    is not one burst from one busy page, as "most recent N" would be.
    """
    params = events_params(
        project_id,
        build_query(side, field_name, [name]),
        sample_fields(field_name),
        SAMPLE_SORT,
        side.windows,
        min(PAGE_SIZE, cap),
    )
    return client.paginate(params, cap)


def collect_samples(
    client: SentryClient,
    side: Side,
    project_id: str,
    options: Options,
    names: Sequence[str],
    pool: concurrent.futures.Executor,
) -> Dict[str, List[List[float]]]:
    futures = {
        name: pool.submit(
            fetch_sample,
            client,
            side,
            project_id,
            options.field,
            name,
            options.sample_cap,
        )
        for name in names
    }
    return {
        name: to_clusters(future.result(), options.field)
        for name, future in futures.items()
    }


def side_summary(
    aggregate: Optional[Dict[str, object]], clusters: Sequence[Sequence[float]]
) -> Dict[str, object]:
    summary = dict(
        aggregate
        or {"count": 0, "p50": None, "p95": None, "p99": None, "failure_rate": None}
    )
    summary["sample_n"] = sum(len(cluster) for cluster in clusters)
    summary["sample_traces"] = len(clusters)
    return summary


def compare_transaction(
    name: str,
    baseline: Optional[Dict[str, object]],
    candidate: Optional[Dict[str, object]],
    base_clusters: Sequence[Sequence[float]],
    cand_clusters: Sequence[Sequence[float]],
    options: Options,
    quantile_family: int = 2,
    failure_family: int = 1,
) -> Dict[str, object]:
    rng = random.Random(f"{options.seed}/{name}")
    quantiles = compare_quantiles(
        base_clusters,
        cand_clusters,
        rng,
        options,
        quantile_family,
        (baseline, candidate),
    )
    failure = compare_failure_rates(baseline, candidate, options, failure_family)
    verdict = combine_verdicts([stat["verdict"] for stat in quantiles.values()])
    return {
        "transaction": name,
        "baseline": side_summary(baseline, base_clusters),
        "candidate": side_summary(candidate, cand_clusters),
        "p50": quantiles["p50"],
        "p95": quantiles["p95"],
        "failure_rate": failure,
        "verdict": verdict,
        "note": sample_shortfall(base_clusters, cand_clusters, options),
        "regression": verdict == SLOWER or failure["verdict"] == HIGHER,
    }


def fetch_aggregates(
    client: SentryClient, side: Side, project_id: str, options: Options
) -> Tuple[Dict[str, Dict[str, object]], Optional[str]]:
    params = events_params(
        project_id,
        build_query(side, options.field, options.transactions),
        aggregate_fields(options.field),
        "-count()",
        side.windows,
        PAGE_SIZE,
    )
    rows, meta, _ = client.events(params)
    units = meta.get("units") if isinstance(meta.get("units"), dict) else {}
    aggregates = {
        str(row.get("transaction")): summarize_aggregate(row, options.field)
        for row in rows
    }
    return aggregates, units.get(f"p50({options.field})")


def eligible_transactions(
    names: Sequence[str],
    base_aggregates: Dict[str, Dict[str, object]],
    cand_aggregates: Dict[str, Dict[str, object]],
    options: Options,
) -> List[str]:
    def count(aggregates: Dict[str, Dict[str, object]], name: str) -> int:
        return int(aggregates.get(name, {}).get("count") or 0)

    return [
        name
        for name in names
        if min(count(base_aggregates, name), count(cand_aggregates, name))
        >= options.min_count
    ]


def compare(
    client: SentryClient, baseline: Side, candidate: Side, options: Options
) -> Dict[str, object]:
    project_ids = {
        side.label: client.project_id(side.project) for side in (baseline, candidate)
    }
    base_aggregates, base_unit = fetch_aggregates(
        client, baseline, project_ids[baseline.label], options
    )
    cand_aggregates, cand_unit = fetch_aggregates(
        client, candidate, project_ids[candidate.label], options
    )
    names = choose_transactions(base_aggregates, cand_aggregates, options)
    eligible = eligible_transactions(names, base_aggregates, cand_aggregates, options)
    samples: Dict[str, Dict[str, List[List[float]]]] = {}
    with concurrent.futures.ThreadPoolExecutor(options.concurrency) as pool:
        for side in (baseline, candidate):
            samples[side.label] = collect_samples(
                client, side, project_ids[side.label], options, eligible, pool
            )
    return build_report(
        baseline,
        candidate,
        options,
        project_ids,
        (base_aggregates, cand_aggregates),
        (samples[baseline.label], samples[candidate.label]),
        base_unit or cand_unit,
        client.requests,
    )


def build_report(
    baseline: Side,
    candidate: Side,
    options: Options,
    project_ids: Dict[str, str],
    aggregates: Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]],
    samples: Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]]],
    unit: Optional[str],
    requests: int,
) -> Dict[str, object]:
    """Turn fetched aggregates and samples into the report; no I/O."""
    base_aggregates, cand_aggregates = aggregates
    base_samples, cand_samples = samples
    names = choose_transactions(base_aggregates, cand_aggregates, options)
    sampled = [
        name
        for name in names
        if sample_shortfall(
            base_samples.get(name, []), cand_samples.get(name, []), options
        )
        is None
    ]
    quantile_family = len(QUANTILES) * len(sampled)
    failure_family = len(
        eligible_transactions(names, base_aggregates, cand_aggregates, options)
    )
    level = adjusted_confidence(options.confidence, quantile_family)
    run_options = replace(
        options, resamples=effective_resamples(options.resamples, level)
    )
    results = [
        compare_transaction(
            name,
            base_aggregates.get(name),
            cand_aggregates.get(name),
            base_samples.get(name, []),
            cand_samples.get(name, []),
            run_options,
            quantile_family,
            failure_family,
        )
        for name in names
    ]

    def counts(name: str) -> Tuple[int, int]:
        return (
            int(base_aggregates.get(name, {}).get("count") or 0),
            int(cand_aggregates.get(name, {}).get("count") or 0),
        )

    overview_names = sorted(
        set(base_aggregates) | set(cand_aggregates),
        key=lambda name: (-min(counts(name)), -sum(counts(name)), name),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "field": options.field,
        "unit": unit,
        "baseline": dict(
            baseline.selector(),
            project_id=project_ids.get(baseline.label),
            aggregate_query=build_query(baseline, options.field, options.transactions),
        ),
        "candidate": dict(
            candidate.selector(),
            project_id=project_ids.get(candidate.label),
            aggregate_query=build_query(candidate, options.field, options.transactions),
        ),
        "settings": {
            "transactions": options.transactions or None,
            "top": options.top,
            "sample_cap": options.sample_cap,
            "sample_sort": SAMPLE_SORT,
            "min_count": options.min_count,
            "min_traces": options.min_traces,
            "min_effect": options.min_effect,
            "min_failure_delta": options.min_failure_delta,
            "confidence": options.confidence,
            "comparisons": quantile_family,
            "failure_comparisons": failure_family,
            "adjusted_confidence": level,
            "resamples": run_options.resamples,
            "seed": options.seed,
        },
        "overview": [
            {
                "transaction": name,
                "baseline": base_aggregates.get(name),
                "candidate": cand_aggregates.get(name),
            }
            for name in overview_names
        ],
        "transactions": results,
        "overall": overall_outcome(results),
        "requests": requests,
    }


# Rendering ------------------------------------------------------------------


def format_number(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    size = abs(value)
    if size >= 1000:
        return f"{value:,.0f}"
    if size >= 100:
        return f"{value:.0f}"
    if size >= 10:
        return f"{value:.1f}"
    if size >= 1:
        return f"{value:.2f}"
    return f"{value:.3g}"


def format_signed(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    text = format_number(value)
    return text if text.startswith("-") else "+" + text


def format_rel(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def format_points(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:+.2f} pp"


def format_percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def format_change(stat: Dict[str, object], suffix: str) -> str:
    if stat["diff"] is None:
        return "n/a"
    return (
        f"{format_signed(stat['diff'])}{suffix} ({format_rel(stat['rel'])}); "
        f"Sentry {format_rel(stat['sentry_rel'])}"
    )


def format_interval(stat: Dict[str, object], suffix: str) -> str:
    if not stat["ci_adjusted"]:
        return "n/a"
    low, high = stat["ci_adjusted"]
    return f"[{format_signed(low)}, {format_signed(high)}]{suffix}"


def format_level(level: float) -> str:
    return f"{level * 100:.3g}%"


def cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def describe_side(selector: Dict[str, object]) -> str:
    windows = ", ".join(f"{pair[0]} to {pair[1]}" for pair in selector["windows"])
    parts = [
        f"project `{selector['project']}`",
        f"env `{selector['environment'] or 'any'}`",
        f"release `{selector['release'] or 'any'}`",
    ]
    if selector.get("query"):
        parts.append(f"query `{selector['query']}`")
    return ", ".join(parts) + f"; windows {windows}"


def render_markdown(report: Dict[str, object]) -> str:
    unit = UNIT_SUFFIX.get(str(report.get("unit")), "")
    suffix = f" {unit}" if unit else ""
    settings = report["settings"]
    lines = [
        f"## perf-compare: `{report['field']}`" + (f" ({unit})" if unit else ""),
        "",
        f"- **Baseline:** {describe_side(report['baseline'])}",
        f"- **Candidate:** {describe_side(report['candidate'])}",
        "",
        "### Sentry aggregates (every matching event)",
        "",
        "| transaction | n base | n cand | p50 base | p50 cand | p95 base "
        "| p95 cand | p99 base | p99 cand | fail base | fail cand |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    empty: Dict[str, object] = {}
    for row in report["overview"][:OVERVIEW_ROWS]:
        base = row["baseline"] or empty
        cand = row["candidate"] or empty
        lines.append(
            "| "
            + " | ".join(
                cell(value)
                for value in (
                    row["transaction"],
                    base.get("count", 0),
                    cand.get("count", 0),
                    format_number(base.get("p50")),
                    format_number(cand.get("p50")),
                    format_number(base.get("p95")),
                    format_number(cand.get("p95")),
                    format_number(base.get("p99")),
                    format_number(cand.get("p99")),
                    format_percent(base.get("failure_rate")),
                    format_percent(cand.get("failure_rate")),
                )
            )
            + " |"
        )
    hidden = len(report["overview"]) - OVERVIEW_ROWS
    if hidden > 0:
        lines.append(f"\n{hidden} more transactions in `--json` output.")
    confidence = format_level(float(settings["confidence"]))
    adjusted = format_level(float(settings["adjusted_confidence"]))
    lines += [
        "",
        "### Verdicts (candidate minus baseline)",
        "",
        f"Intervals are {adjusted} bootstrap CIs: {confidence} held jointly over "
        f"the {settings['comparisons']} percentile comparisons below "
        "(Bonferroni). Changes are from the sample, then from Sentry's "
        "percentiles over every event.",
        "",
        "| transaction | samples base / cand | p50 change | p50 CI | p95 change "
        "| p95 CI | failure rate change | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in report["transactions"]:
        base, cand = result["baseline"], result["candidate"]
        failure = result["failure_rate"]
        failure_text = format_points(failure["diff"])
        if failure["verdict"] in (HIGHER, LOWER):
            failure_text += f" ({failure['verdict']})"
        verdict = f"**{result['verdict']}**"
        if failure["verdict"] == HIGHER:
            verdict = f"**{result['verdict']}, more failures**"
        if result.get("note"):
            verdict += f" ({result['note']})"
        lines.append(
            "| "
            + " | ".join(
                cell(value)
                for value in (
                    result["transaction"],
                    f"{base['sample_n']} / {cand['sample_n']} "
                    f"({base['sample_traces']} / {cand['sample_traces']} traces)",
                    format_change(result["p50"], suffix),
                    format_interval(result["p50"], suffix),
                    format_change(result["p95"], suffix),
                    format_interval(result["p95"], suffix),
                    failure_text,
                    verdict,
                )
            )
            + " |"
        )
    overall = report["overall"]
    lines += [
        "",
        f"**Overall: {overall['verdict']}** (exit {overall['exit_code']})",
    ]
    lines += [f"- {reason}" for reason in overall["reasons"]]
    lines += [
        "",
        f"Rules: a side needs {settings['min_count']} samples from "
        f"{settings['min_traces']} traces. Faster or slower needs the CI to "
        "exclude zero, and the sample change and Sentry's change to both be at "
        f"least {float(settings['min_effect']) * 100:g}% in that direction. "
        "Failure rates use Fisher's exact test over "
        f"{settings['failure_comparisons']} transactions (Bonferroni). Up to "
        f"{settings['sample_cap']} samples per side in span-id order, resampled "
        f"by trace {settings['resamples']} times with seed {settings['seed']}. "
        f"{report['requests']} Sentry requests.",
        "",
    ]
    return "\n".join(lines)


def render_queries(
    org: str,
    region_url: str,
    baseline: Side,
    candidate: Side,
    options: Options,
    now: datetime,
) -> str:
    events_url = "{}/api/0/organizations/{}/events/".format(region_url.rstrip("/"), org)
    names = options.transactions or ["<TRANSACTION>"]
    lines = [
        "# Queries perf_compare.py sends (Events API, dataset=spans)",
        "",
        "REST needs the numeric project id (GET /api/0/projects/{org}/{slug}/ "
        "returns it). The MCP takes the slug. The query string is identical for "
        "both, and its timestamp filter pins the exact windows, so the MCP "
        "`period` only has to cover them.",
        "",
        "Raw samples are sorted by span `id`, which is random, so the first N "
        "rows are a uniform random sample of the windows. REST pages through "
        "them with the `Link` cursor. The MCP stops at 100 rows per call: for a "
        "bigger sample, cut the window into sub-windows (narrow the timestamp "
        "filter), get count() for each, and take the first rows by `id` from "
        "each sub-window in proportion to its count.",
    ]
    for side in (baseline, candidate):
        plans = [
            (
                "aggregate per transaction",
                build_query(side, options.field, options.transactions),
                aggregate_fields(options.field),
                "-count()",
                PAGE_SIZE,
            )
        ]
        plans.extend(
            (
                f"raw sample for {name} (first {options.sample_cap} rows by id)",
                build_query(side, options.field, [name]),
                sample_fields(options.field),
                SAMPLE_SORT,
                PAGE_SIZE,
            )
            for name in names
        )
        for title, query, fields, sort, limit in plans:
            params = events_params(
                "PROJECT_ID", query, fields, sort, side.windows, limit
            )
            lines += [
                "",
                f"## {side.label}: {title}",
                "",
                "REST:",
                "",
                f"    GET {events_url}?{urllib.parse.urlencode(params)}",
                "",
                f"(PROJECT_ID = id of `{side.project}`)",
                "",
                "MCP `search_events` arguments (or `search_traces` without "
                "`dataset`):",
                "",
                json.dumps(
                    mcp_arguments(
                        org, region_url, side, query, fields, sort, limit, now
                    ),
                    indent=2,
                ),
            ]
    return "\n".join(lines) + "\n"


# Command line ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="perf_compare.py",
        description=(
            "Compare a numeric span field (duration by default) between a "
            "baseline and a candidate selector in Sentry, per transaction, and "
            "say whether the candidate is faster, slower, unchanged, or "
            "under-sampled."
        ),
        epilog=EXIT_CODES_HELP
        + "\nselector keys for --baseline/--candidate: "
        + ", ".join(SIDE_KEYS)
        + "\nexample:\n  perf_compare.py --project builder --env app:prod-behave:cloud "
        "\\\n    --baseline release=OLD_SHA --candidate release=NEW_SHA --period 7d",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    selectors = parser.add_argument_group("selectors")
    selectors.add_argument(
        "--baseline",
        nargs="+",
        required=True,
        metavar="KEY=VALUE",
        help="baseline selector, e.g. release=SHA env=pr-123 window=START/END",
    )
    selectors.add_argument(
        "--candidate",
        nargs="+",
        required=True,
        metavar="KEY=VALUE",
        help="candidate selector, same keys as --baseline",
    )
    selectors.add_argument("--project", help="project slug used by both sides")
    selectors.add_argument(
        "--env", help="environment used by both sides, e.g. app:prod-behave:cloud"
    )
    selectors.add_argument(
        "--query",
        help="extra Sentry search filter for both sides, e.g. span.op:pageload",
    )
    timing = parser.add_argument_group(
        "time (both sides; a side's own window=, start=/end= or period= wins)"
    )
    timing.add_argument(
        "--period", help=f"window ending now, e.g. 24h (default {DEFAULT_PERIOD})"
    )
    timing.add_argument("--start", help="window start, ISO 8601 UTC")
    timing.add_argument("--end", help="window end, ISO 8601 UTC (default now)")
    timing.add_argument(
        "--window",
        action="append",
        default=[],
        help="START/END in UTC; repeat for interleaved runs",
    )
    what = parser.add_argument_group("what to compare")
    what.add_argument(
        "--transaction",
        action="append",
        default=[],
        help="transaction name to compare; repeat for several "
        "(default: the --top busiest transactions on both sides)",
    )
    what.add_argument(
        "--top",
        type=int,
        default=5,
        help="without --transaction, compare this many of the busiest "
        "transactions present on both sides (default 5)",
    )
    what.add_argument(
        "--field",
        default=DEFAULT_FIELD,
        help="numeric field, e.g. span.duration, measurements.lcp, "
        "gen_ai.usage.input_tokens, tags[my.attribute,number]",
    )
    stats = parser.add_argument_group("statistics")
    stats.add_argument(
        "--sample-cap",
        type=int,
        default=1000,
        help="raw values fetched per side per transaction, a random sample in "
        f"span-id order (default 1000, at most {MAX_SAMPLE_CAP})",
    )
    stats.add_argument(
        "--min-count",
        type=int,
        default=200,
        help="samples each side needs before any verdict (default 200)",
    )
    stats.add_argument(
        "--min-traces",
        type=int,
        default=30,
        help="distinct traces each side's samples must come from before any "
        "verdict (default 30)",
    )
    stats.add_argument(
        "--min-effect",
        type=float,
        default=0.05,
        help="smallest relative change that can be called faster or slower "
        "(default 0.05 = 5%%)",
    )
    stats.add_argument(
        "--min-failure-delta",
        type=float,
        default=0.001,
        help="smallest failure-rate change, as a fraction, that counts "
        "(default 0.001 = 0.1 pp)",
    )
    stats.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="confidence held jointly over every percentile compared in the "
        "run; each interval is Bonferroni-adjusted (default 0.95)",
    )
    stats.add_argument(
        "--resamples", type=int, default=4000, help="bootstrap resamples (default 4000)"
    )
    stats.add_argument(
        "--seed", type=int, default=1234, help="bootstrap seed (default 1234)"
    )
    output = parser.add_argument_group("output and connection")
    output.add_argument("--json", action="store_true", help="machine-readable output")
    output.add_argument(
        "--print-queries",
        action="store_true",
        help="print the REST and MCP queries without calling Sentry",
    )
    output.add_argument("--org", default=DEFAULT_ORG, help=f"default {DEFAULT_ORG}")
    output.add_argument(
        "--region-url", default=DEFAULT_REGION_URL, help=f"default {DEFAULT_REGION_URL}"
    )
    output.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="parallel Sentry requests (default 4)",
    )
    output.add_argument(
        "--timeout", type=float, default=30.0, help="seconds per request (default 30)"
    )
    return parser


def check_region_url(url: str) -> None:
    """https with a host, or http only for a server on this machine."""
    parts = urllib.parse.urlsplit(url)
    secure = parts.scheme == "https" and bool(parts.hostname)
    local = parts.scheme == "http" and parts.hostname in LOCAL_HOSTS
    bare = parts.path in ("", "/") and not parts.query and not parts.fragment
    if not ((secure or local) and bare):
        raise UsageError(
            f"bad --region-url {url!r}: use the scheme and host only, such as "
            f"{DEFAULT_REGION_URL} (http is allowed only for localhost)"
        )


def build_options(args: argparse.Namespace) -> Options:
    checks = (
        (args.top >= 1, "--top must be at least 1"),
        (args.sample_cap >= 1, "--sample-cap must be at least 1"),
        (args.min_count >= 1, "--min-count must be at least 1"),
        (args.min_traces >= 1, "--min-traces must be at least 1"),
        (args.min_effect >= 0, "--min-effect cannot be negative"),
        (args.min_failure_delta >= 0, "--min-failure-delta cannot be negative"),
        (0 < args.confidence < 1, "--confidence must be between 0 and 1"),
        (args.resamples >= 100, "--resamples must be at least 100"),
        (args.concurrency >= 1, "--concurrency must be at least 1"),
        (args.timeout > 0, "--timeout must be more than 0 seconds"),
        (bool(FIELD_PATTERN.match(args.field)), f"bad --field {args.field!r}"),
    )
    for passed, message in checks:
        if not passed:
            raise UsageError(message)
    check_region_url(args.region_url)
    sample_cap = min(args.sample_cap, MAX_SAMPLE_CAP)
    if sample_cap < args.min_count:
        raise UsageError(
            f"--sample-cap {sample_cap} is below --min-count {args.min_count}, "
            "so no transaction could get a verdict"
        )
    most = len(QUANTILES) * (len(set(args.transaction)) or args.top)
    level = adjusted_confidence(args.confidence, most)
    if RESAMPLE_CAP * tail_share(level) < MIN_TAIL_DRAWS:
        raise UsageError(
            f"{most} percentile comparisons need more than {RESAMPLE_CAP} "
            "bootstrap resamples; compare fewer transactions"
        )
    return Options(
        field=args.field,
        transactions=list(args.transaction),
        top=args.top,
        sample_cap=sample_cap,
        min_count=args.min_count,
        min_traces=args.min_traces,
        min_effect=args.min_effect,
        min_failure_delta=args.min_failure_delta,
        confidence=args.confidence,
        resamples=args.resamples,
        seed=args.seed,
        concurrency=args.concurrency,
    )


def build_sides(args: argparse.Namespace, now: datetime) -> Tuple[Side, Side]:
    shared_windows = resolve_windows(
        "shared time", args.window, args.start, args.end, args.period, now
    ) or [Window(now - parse_period(DEFAULT_PERIOD), now)]
    shared = Shared(args.project, args.env, args.query, shared_windows)
    baseline = parse_side("baseline", args.baseline, shared, now)
    candidate = parse_side("candidate", args.candidate, shared, now)
    if baseline.selector() == candidate.selector():
        raise UsageError(
            "baseline and candidate select the same events; change a release, "
            "environment, project, query or window on one side"
        )
    return baseline, candidate


def main(
    argv: Optional[Sequence[str]] = None,
    environ: Optional[Dict[str, str]] = None,
    transport: Optional[Transport] = None,
    now: Optional[datetime] = None,
    stdout=None,
    stderr=None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    environ = os.environ if environ is None else environ
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    args = build_parser().parse_args(argv)

    def fail(kind: str, message: str, code: int) -> int:
        stderr.write(message if message.endswith("\n") else message + "\n")
        if args.json:
            error = {"type": kind, "message": message.strip()}
            stdout.write(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "error": error,
                        "exit_code": code,
                    }
                )
                + "\n"
            )
        return code

    try:
        moment = (now or datetime.now(timezone.utc)).replace(microsecond=0)
        options = build_options(args)
        if options.sample_cap < args.sample_cap:
            stderr.write(
                f"perf-compare: --sample-cap is limited to {MAX_SAMPLE_CAP}; "
                f"using {options.sample_cap}\n"
            )
        baseline, candidate = build_sides(args, moment)
        if args.print_queries:
            stdout.write(
                render_queries(
                    args.org, args.region_url, baseline, candidate, options, moment
                )
            )
            return 0
        token = (environ.get("SENTRY_AUTH_TOKEN") or "").strip()
        if not token:
            return fail("auth", MISSING_TOKEN_MESSAGE, EXIT_AUTH)
        client = SentryClient(
            token,
            args.org,
            args.region_url,
            transport=transport,
            timeout=args.timeout,
            sleep=sleep,
        )
        report = compare(client, baseline, candidate, options)
    except PerfCompareError as error:
        return fail(error.kind, f"perf-compare: {error}", error.exit_code)
    except Exception as error:
        return fail(
            "internal",
            f"perf-compare: unexpected {type(error).__name__}: {error}",
            EXIT_API,
        )
    report["generated_at"] = format_filter_time(moment)
    report["org"] = args.org
    if args.json:
        stdout.write(json.dumps(report, indent=2) + "\n")
    else:
        stdout.write(render_markdown(report))
    return int(report["overall"]["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
