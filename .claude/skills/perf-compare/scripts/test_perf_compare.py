"""Offline tests for perf_compare.py.

Run from this directory with ``python3 -m unittest``. Sentry is replaced by a
fake transport that serves the synthetic responses in ``fixtures/``.
"""

import contextlib
import io
import json
import random
import sys
import unittest
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import perf_compare as pc  # noqa: E402

FIXTURES = HERE / "fixtures"
NOW = datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc)
BASE_SHA = "a" * 40
CAND_SHA = "b" * 40
WINDOW_B = "2026-10-10T14:00:00Z/2026-10-11T14:00:00Z"
WINDOW_C = "2026-10-11T14:00:00Z/2026-10-12T14:00:00Z"
EVENTS_PATH = "/api/0/organizations/significant-gravitas/events/"
PROJECT_PATH = "/api/0/projects/significant-gravitas/builder/"
LIBRARY = "GET /api/library/agents"
STORE = "GET /api/store/agents"


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def respond(name, status=200):
    data = fixture(name)
    return status, data["headers"], json.dumps(data["body"]).encode()


class FakeTransport:
    """Answers through ``route`` and records every request."""

    def __init__(self, route):
        self.route = route
        self.calls = []

    def __call__(self, url, headers, timeout):
        parts = urllib.parse.urlsplit(url)
        params = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
        self.calls.append({"path": parts.path, "params": params, "headers": headers})
        return self.route(parts.path, params)


def sentry_route(candidate_data="candidate"):
    """Route like Sentry: project lookup, aggregates, then paged samples."""

    def route(path, params):
        if path == PROJECT_PATH:
            return respond("project_builder.json")
        if path != EVENTS_PATH:
            return 404, {}, b'{"detail": "not found"}'
        query = params["query"][0]
        side = "baseline" if f'release:"{BASE_SHA}"' in query else candidate_data
        if "count()" in params.get("field", []):
            return respond(f"aggregate_{side}.json")
        page = 2 if params.get("cursor") == ["0:100:0"] else 1
        return respond(f"samples_{side}_page{page}.json")

    return route


def side(**overrides):
    values = dict(
        label="baseline",
        project="builder",
        environment="app:prod-behave:cloud",
        release=BASE_SHA,
        query=None,
        windows=[pc.parse_window(WINDOW_B)],
    )
    values.update(overrides)
    return pc.Side(**values)


def run_main(argv, environ=None, transport=None):
    stdout, stderr = io.StringIO(), io.StringIO()
    code = pc.main(
        argv,
        environ={"SENTRY_AUTH_TOKEN": "test-token"} if environ is None else environ,
        transport=transport,
        now=NOW,
        stdout=stdout,
        stderr=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue()


COMPARE_ARGS = [
    "--project",
    "builder",
    "--env",
    "app:prod-behave:cloud",
    "--baseline",
    f"release={BASE_SHA}",
    f"window={WINDOW_B}",
    "--candidate",
    f"release={CAND_SHA}",
    f"window={WINDOW_C}",
    "--min-count",
    "150",
    "--resamples",
    "500",
]


class QueryBuildingTest(unittest.TestCase):
    def test_release_side_query_has_every_filter_and_the_window(self):
        self.assertEqual(
            pc.build_query(side(), "span.duration"),
            'is_transaction:true environment:"app:prod-behave:cloud" '
            f'release:"{BASE_SHA}" '
            "timestamp:>=2026-10-10T14:00:00Z timestamp:<2026-10-11T14:00:00Z",
        )

    def test_non_duration_field_requires_the_field_to_exist(self):
        query = pc.build_query(side(environment=None, release=None), "measurements.lcp")
        self.assertTrue(query.startswith("is_transaction:true has:measurements.lcp "))

    def test_extra_query_is_grouped_so_an_or_cannot_escape(self):
        query = pc.build_query(
            side(query="span.op:pageload OR span.op:navigation"), "span.duration"
        )
        self.assertIn("(span.op:pageload OR span.op:navigation)", query)

    def test_several_transactions_become_one_or_group_with_escaping(self):
        query = pc.build_query(side(), "span.duration", [LIBRARY, 'say "hi" \\ bye'])
        self.assertIn(
            f'(transaction:"{LIBRARY}" OR transaction:"say \\"hi\\" \\\\ bye")', query
        )

    def test_interleaved_windows_become_one_or_group(self):
        windows = [
            pc.parse_window("2026-10-11T10:00:00Z/2026-10-11T10:10:00Z"),
            pc.parse_window("2026-10-11T10:20:00Z/2026-10-11T10:30:00Z"),
        ]
        self.assertTrue(
            pc.build_query(side(windows=windows), "span.duration").endswith(
                "(timestamp:>=2026-10-11T10:00:00Z timestamp:<2026-10-11T10:10:00Z "
                "OR timestamp:>=2026-10-11T10:20:00Z timestamp:<2026-10-11T10:30:00Z)"
            )
        )

    def test_events_params_cover_the_union_of_windows(self):
        windows = [
            pc.parse_window("2026-10-11T10:00:00Z/2026-10-11T10:10:00Z"),
            pc.parse_window("2026-10-11T10:20:00Z/2026-10-11T10:30:00Z"),
        ]
        params = pc.events_params(
            "42", "q", pc.aggregate_fields("span.duration"), "-count()", windows, 100
        )
        self.assertEqual(
            params,
            [
                ("dataset", "spans"),
                ("project", "42"),
                ("field", "transaction"),
                ("field", "count()"),
                ("field", "p50(span.duration)"),
                ("field", "p95(span.duration)"),
                ("field", "p99(span.duration)"),
                ("field", "failure_rate()"),
                ("query", "q"),
                ("sort", "-count()"),
                ("per_page", "100"),
                ("start", "2026-10-11T10:00:00"),
                ("end", "2026-10-11T10:30:00"),
            ],
        )

    def test_samples_are_sorted_by_span_id(self):
        self.assertEqual(pc.SAMPLE_SORT, "id")
        self.assertEqual(
            pc.sample_fields("measurements.lcp"), ["measurements.lcp", "trace", "id"]
        )

    def test_mcp_arguments_use_a_period_that_covers_the_window(self):
        arguments = pc.mcp_arguments(
            "org",
            "https://us.sentry.io",
            side(),
            "q",
            ["count()"],
            "-count()",
            100,
            NOW,
        )
        self.assertEqual(arguments["period"], "50h")
        self.assertEqual(arguments["projectSlug"], "builder")
        self.assertEqual(arguments["dataset"], "spans")


class SelectorTest(unittest.TestCase):
    def shared(self, **overrides):
        values = dict(
            project="builder",
            env="app:prod-behave:cloud",
            query=None,
            windows=[pc.Window(NOW - timedelta(days=7), NOW)],
        )
        values.update(overrides)
        return pc.Shared(**values)

    def test_side_keys_override_shared_defaults(self):
        parsed = pc.parse_side(
            "candidate",
            ["environment=pr-123", "release=abc", "period=24h"],
            self.shared(),
            NOW,
        )
        self.assertEqual(parsed.environment, "pr-123")
        self.assertEqual(parsed.release, "abc")
        self.assertEqual(parsed.windows, [pc.Window(NOW - timedelta(hours=24), NOW)])

    def test_empty_env_clears_the_shared_environment(self):
        parsed = pc.parse_side("baseline", ["env="], self.shared(), NOW)
        self.assertIsNone(parsed.environment)

    def test_unknown_or_repeated_keys_are_usage_errors(self):
        for tokens in (["colour=red"], ["release"], ["release=a", "release=b"]):
            with self.subTest(tokens=tokens), self.assertRaises(pc.UsageError):
                pc.parse_side("baseline", tokens, self.shared(), NOW)

    def test_missing_project_is_a_usage_error(self):
        with self.assertRaises(pc.UsageError):
            pc.parse_side("baseline", ["release=a"], self.shared(project=None), NOW)

    def test_bad_windows_are_usage_errors(self):
        cases = (
            ["window=2026-10-11T10:00:00Z/2026-10-11T09:00:00Z"],
            [
                "window=2026-10-11T10:00:00Z/2026-10-11T11:00:00Z",
                "window=2026-10-11T10:30:00Z/2026-10-11T12:00:00Z",
            ],
            ["window=2026-10-11T10:00:00Z"],
            ["period=24h", "start=2026-10-11T10:00:00Z"],
            ["period=0h"],
        )
        for tokens in cases:
            with self.subTest(tokens=tokens), self.assertRaises(pc.UsageError):
                pc.parse_side("baseline", tokens, self.shared(), NOW)

    def test_identical_selectors_are_refused(self):
        code, _, err = run_main(
            [
                "--project",
                "builder",
                "--baseline",
                "release=a",
                "--candidate",
                "release=a",
            ]
        )
        self.assertEqual(code, pc.EXIT_USAGE)
        self.assertIn("select the same events", err)

    def test_bad_field_is_refused_before_any_request(self):
        transport = FakeTransport(sentry_route())
        code, _, _ = run_main(
            COMPARE_ARGS + ["--field", "p50(x) OR 1"], transport=transport
        )
        self.assertEqual(code, pc.EXIT_USAGE)
        self.assertEqual(transport.calls, [])

    def test_malformed_selector_exits_with_usage_code(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
            SystemExit
        ) as raised:
            pc.build_parser().parse_args(["--baseline"])
        self.assertEqual(raised.exception.code, pc.EXIT_USAGE)


class PaginationTest(unittest.TestCase):
    def client(self, route, sleep=None):
        transport = FakeTransport(route)
        client = pc.SentryClient(
            "test-token", transport=transport, sleep=sleep or (lambda seconds: None)
        )
        return client, transport

    def test_link_header_cursor_parsing(self):
        first = fixture("samples_baseline_page1.json")["headers"]["link"]
        last = fixture("samples_baseline_page2.json")["headers"]["link"]
        self.assertEqual(pc.next_cursor(first), "0:100:0")
        self.assertIsNone(pc.next_cursor(last))
        self.assertIsNone(pc.next_cursor(""))

    def test_paginate_follows_the_cursor_until_sentry_has_no_more(self):
        client, transport = self.client(sentry_route())
        rows = client.paginate([("query", f'release:"{BASE_SHA}"')], 1000)
        self.assertEqual(len(rows), 160)
        self.assertEqual(len(transport.calls), 2)
        self.assertNotIn("cursor", transport.calls[0]["params"])
        self.assertEqual(transport.calls[1]["params"]["cursor"], ["0:100:0"])

    def test_paginate_stops_at_the_cap(self):
        client, transport = self.client(sentry_route())
        self.assertEqual(len(client.paginate([("query", "x")], 50)), 50)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(len(client.paginate([("query", "x")], 130)), 130)
        self.assertEqual(len(transport.calls), 3)

    def test_rate_limits_are_retried_after_the_advertised_delay(self):
        replies = [(429, {"retry-after": "7"}, b"{}"), respond("project_builder.json")]
        slept = []
        client, _ = self.client(lambda path, params: replies.pop(0), slept.append)
        self.assertEqual(client.project_id("builder"), "1000001")
        self.assertEqual(slept, [7.0])

    def test_network_errors_are_retried(self):
        replies = [OSError("reset"), respond("project_builder.json")]

        def route(path, params):
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        client, _ = self.client(route)
        self.assertEqual(client.project_id("builder"), "1000001")

    def test_server_errors_give_up_after_the_retries(self):
        slept = []
        client, transport = self.client(
            lambda path, params: (503, {}, b"unavailable"), slept.append
        )
        with self.assertRaises(pc.SentryApiError):
            client.project_id("builder")
        self.assertEqual(len(transport.calls), 4)
        self.assertEqual(slept, [1.0, 2.0, 4.0])

    def test_rejected_token_is_an_auth_error(self):
        client, _ = self.client(lambda path, params: (403, {}, b'{"detail": "no"}'))
        with self.assertRaises(pc.AuthError) as raised:
            client.project_id("builder")
        self.assertIn("event:read", str(raised.exception))

    def test_rejected_query_is_a_usage_error_with_sentry_detail(self):
        client, _ = self.client(lambda path, params: respond("error_400.json", 400))
        with self.assertRaises(pc.UsageError) as raised:
            client.events([("query", "x")])
        self.assertIn("p50(nope)", str(raised.exception))

    def test_unknown_project_is_a_usage_error(self):
        client, _ = self.client(lambda path, params: (404, {}, b'{"detail": "x"}'))
        with self.assertRaises(pc.UsageError):
            client.project_id("nope")

    def test_project_ids_are_looked_up_once(self):
        client, transport = self.client(sentry_route())
        client.project_id("builder")
        client.project_id("builder")
        self.assertEqual(client.project_id("12345"), "12345")
        self.assertEqual(len(transport.calls), 1)


class StatisticsTest(unittest.TestCase):
    def test_percentile_interpolates_between_ranks(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(pc.percentile(values, 0.0), 1.0)
        self.assertEqual(pc.percentile(values, 0.5), 3.0)
        self.assertAlmostEqual(pc.percentile(values, 0.95), 4.8)
        self.assertEqual(pc.percentile(values, 1.0), 5.0)
        self.assertEqual(pc.percentile([7.0], 0.95), 7.0)
        with self.assertRaises(ValueError):
            pc.percentile([], 0.5)

    def test_confidence_interval_takes_the_matching_tails(self):
        low, high = pc.confidence_interval([float(i) for i in range(1, 101)], 0.9)
        self.assertAlmostEqual(low, 5.95)
        self.assertAlmostEqual(high, 95.05)

    def test_bonferroni_adjustment(self):
        self.assertAlmostEqual(pc.adjusted_confidence(0.95, 10), 0.995)
        self.assertEqual(pc.adjusted_confidence(0.95, 1), 0.95)
        self.assertEqual(pc.adjusted_confidence(0.95, 0), 0.95)

    def test_bootstrap_of_a_constant_shift_is_exact(self):
        diffs = pc.bootstrap_quantile_diffs(
            [[100.0]] * 50, [[110.0]] * 50, [0.5, 0.95], 200, random.Random(1)
        )
        self.assertEqual(set(diffs[0.5]), {10.0})
        self.assertEqual(set(diffs[0.95]), {10.0})

    def test_bootstrap_resamples_whole_traces(self):
        draw = pc._drawer([[5.0, 5.0, 5.0], [1.0]], random.Random(3))
        for _ in range(50):
            values = draw()
            self.assertEqual(values.count(5.0) % 3, 0)
            self.assertEqual(len(values), values.count(5.0) + values.count(1.0))

    def test_bootstrap_is_deterministic_for_a_seed(self):
        rng = random.Random(11)
        baseline = [[rng.lognormvariate(4.8, 0.4)] for _ in range(120)]
        candidate = [[rng.lognormvariate(4.9, 0.4)] for _ in range(120)]

        def run(seed):
            return pc.compare_quantiles(
                baseline,
                candidate,
                random.Random(seed),
                pc.Options(min_count=100, resamples=300),
            )

        self.assertEqual(run(5), run(5))
        self.assertNotEqual(run(5)["p95"]["ci"], run(6)["p95"]["ci"])

    def test_bootstrap_matches_pinned_values(self):
        rng = random.Random(2024)
        baseline = [[round(rng.uniform(90, 110), 3)] for _ in range(40)]
        candidate = [[round(rng.uniform(100, 125), 3)] for _ in range(40)]
        result = pc.compare_quantiles(
            baseline,
            candidate,
            random.Random(99),
            pc.Options(min_count=30, resamples=200),
            family_size=1,
        )
        self.assertAlmostEqual(result["p50"]["diff"], PINNED_P50_DIFF, places=9)
        self.assertEqual(
            [round(bound, 9) for bound in result["p50"]["ci"]], PINNED_P50_CI
        )
        self.assertEqual(
            [round(bound, 9) for bound in result["p95"]["ci"]], PINNED_P95_CI
        )

    def test_a_known_shift_is_found_and_no_shift_is_not(self):
        rng = random.Random(42)
        baseline = [[rng.lognormvariate(4.8, 0.3)] for _ in range(400)]
        slower = [[value * 1.3 for value in cluster] for cluster in baseline]
        same = [[rng.lognormvariate(4.8, 0.3)] for _ in range(400)]
        options = pc.Options(resamples=400)
        shifted = pc.compare_quantiles(baseline, slower, random.Random(1), options)
        self.assertEqual(shifted["p50"]["verdict"], pc.SLOWER)
        self.assertEqual(shifted["p95"]["verdict"], pc.SLOWER)
        steady = pc.compare_quantiles(baseline, same, random.Random(1), options)
        self.assertEqual(steady["p50"]["verdict"], pc.NO_CHANGE)
        self.assertEqual(steady["p95"]["verdict"], pc.NO_CHANGE)


class VerdictTest(unittest.TestCase):
    def test_quantile_verdict_rules(self):
        cases = [
            ((-10, -2, (-0.10, -0.08)), pc.FASTER),
            ((2, 10, (0.10, 0.12)), pc.SLOWER),
            ((2, 10, (0.10, None)), pc.SLOWER),
            ((2, 10, (None, None)), pc.SLOWER),
            ((-10, 2, (-0.10, -0.10)), pc.NO_CHANGE),
            ((0, 10, (0.50, 0.50)), pc.NO_CHANGE),
            ((-10, -2, (-0.10, -0.03)), pc.NO_CHANGE),
            ((-10, -2, (-0.03, -0.10)), pc.NO_CHANGE),
            ((-10, -2, (-0.10, 0.06)), pc.NO_CHANGE),
        ]
        for (low, high, effects), expected in cases:
            with self.subTest(low=low, high=high, effects=effects):
                self.assertEqual(
                    pc.quantile_verdict(low, high, effects, 0.05), expected
                )

    def test_too_few_samples_is_not_enough_data(self):
        baseline = [[100.0]] * 199
        candidate = [[200.0]] * 400
        result = pc.compare_quantiles(
            baseline, candidate, random.Random(1), pc.Options(resamples=100)
        )
        self.assertEqual(result["p50"]["verdict"], pc.NOT_ENOUGH_DATA)
        self.assertIsNone(result["p50"]["ci"])
        self.assertEqual(result["p50"]["diff"], 100.0)

    def test_transaction_verdict_takes_the_worst_percentile(self):
        self.assertEqual(pc.combine_verdicts([pc.FASTER, pc.SLOWER]), pc.SLOWER)
        self.assertEqual(pc.combine_verdicts([pc.FASTER, pc.NO_CHANGE]), pc.FASTER)
        self.assertEqual(
            pc.combine_verdicts([pc.NO_CHANGE, pc.NO_CHANGE]), pc.NO_CHANGE
        )
        self.assertEqual(
            pc.combine_verdicts([pc.NOT_ENOUGH_DATA, pc.NOT_ENOUGH_DATA]),
            pc.NOT_ENOUGH_DATA,
        )
        self.assertEqual(pc.combine_verdicts([]), pc.NOT_ENOUGH_DATA)

    def test_failure_rate_test(self):
        options = pc.Options(min_count=100)

        def check(base, cand):
            return pc.compare_failure_rates(base, cand, options)["verdict"]

        def agg(count, rate):
            return {"count": count, "failure_rate": rate}

        self.assertEqual(check(agg(1500, 0.002), agg(1450, 43 / 1450)), pc.HIGHER)
        self.assertEqual(check(agg(1450, 43 / 1450), agg(1500, 0.002)), pc.LOWER)
        self.assertEqual(check(agg(1500, 0.002), agg(1500, 0.0027)), pc.NO_CHANGE)
        self.assertEqual(check(agg(1500, 0.0), agg(1500, 0.0)), pc.NO_CHANGE)
        self.assertEqual(check(agg(50, 0.0), agg(1500, 0.5)), pc.NOT_ENOUGH_DATA)
        self.assertEqual(check(agg(1500, None), agg(1500, 0.5)), pc.NOT_ENOUGH_DATA)

    def test_overall_outcome_and_exit_codes(self):
        def result(verdict, failure=pc.NO_CHANGE):
            stat = {"verdict": verdict, "rel": 0.1, "sentry_rel": 0.1}
            return {
                "transaction": "t",
                "p50": stat,
                "p95": stat,
                "failure_rate": {"verdict": failure, "diff": 0.01},
                "verdict": verdict,
            }

        cases = [
            ([result(pc.FASTER), result(pc.SLOWER)], pc.SLOWER, pc.EXIT_SLOWER),
            (
                [result(pc.FASTER, failure=pc.HIGHER)],
                pc.MORE_FAILURES,
                pc.EXIT_SLOWER,
            ),
            ([result(pc.FASTER), result(pc.NO_CHANGE)], pc.FASTER, pc.EXIT_FASTER),
            (
                [result(pc.NO_CHANGE), result(pc.NOT_ENOUGH_DATA)],
                pc.NO_CHANGE,
                pc.EXIT_NO_CHANGE,
            ),
            ([result(pc.NOT_ENOUGH_DATA)], pc.NOT_ENOUGH_DATA, pc.EXIT_NOT_ENOUGH_DATA),
            ([], pc.NOT_ENOUGH_DATA, pc.EXIT_NOT_ENOUGH_DATA),
        ]
        for results, verdict, code in cases:
            with self.subTest(verdict=verdict):
                outcome = pc.overall_outcome(results)
                self.assertEqual(outcome["verdict"], verdict)
                self.assertEqual(outcome["exit_code"], code)


class MissingTokenTest(unittest.TestCase):
    def refuse(self, path, params):
        raise AssertionError("no request may be sent without a token")

    def test_missing_token_explains_the_scopes_and_sends_nothing(self):
        transport = FakeTransport(self.refuse)
        code, out, err = run_main(COMPARE_ARGS, environ={}, transport=transport)
        self.assertEqual(code, pc.EXIT_AUTH)
        self.assertIn("SENTRY_AUTH_TOKEN is not set", err)
        self.assertIn("org:read  project:read  event:read", err)
        self.assertIn("never looks for tokens on disk", err)
        self.assertEqual(out, "")
        self.assertEqual(transport.calls, [])

    def test_blank_token_counts_as_missing_and_json_mode_reports_it(self):
        code, out, _ = run_main(
            COMPARE_ARGS + ["--json"],
            environ={"SENTRY_AUTH_TOKEN": "  "},
            transport=FakeTransport(self.refuse),
        )
        self.assertEqual(code, pc.EXIT_AUTH)
        payload = json.loads(out)
        self.assertEqual(payload["exit_code"], pc.EXIT_AUTH)
        self.assertEqual(payload["error"]["type"], "auth")

    def test_printing_queries_needs_no_token(self):
        code, out, _ = run_main(
            COMPARE_ARGS + ["--transaction", LIBRARY, "--print-queries"],
            environ={},
            transport=FakeTransport(self.refuse),
        )
        self.assertEqual(code, 0)
        self.assertIn('"sort": "id"', out)
        self.assertIn('"period": "50h"', out)
        self.assertIn(f'transaction:\\"{LIBRARY}\\"', out)


class EndToEndTest(unittest.TestCase):
    def test_slower_candidate_with_more_failures_exits_one(self):
        transport = FakeTransport(sentry_route())
        code, out, _ = run_main(COMPARE_ARGS + ["--json"], transport=transport)
        self.assertEqual(code, pc.EXIT_SLOWER)
        report = json.loads(out)
        library, store = report["transactions"]
        self.assertEqual(library["transaction"], LIBRARY)
        self.assertEqual(library["verdict"], pc.SLOWER)
        self.assertEqual(library["failure_rate"]["verdict"], pc.HIGHER)
        self.assertTrue(library["regression"])
        self.assertEqual(library["baseline"]["sample_n"], 160)
        self.assertEqual(library["candidate"]["sample_n"], 160)
        self.assertGreater(library["p50"]["rel"], 0.15)
        self.assertAlmostEqual(library["p50"]["sentry_rel"], 150.6 / 120.4 - 1)
        self.assertEqual(store["verdict"], pc.NOT_ENOUGH_DATA)
        self.assertEqual(store["baseline"]["sample_n"], 0)
        self.assertEqual(report["settings"]["comparisons"], 2)
        self.assertEqual(report["overall"]["verdict"], pc.SLOWER)
        self.assertEqual(report["unit"], "millisecond")
        self.assertEqual(report["requests"], 7)

    def test_requests_match_the_documented_queries(self):
        transport = FakeTransport(sentry_route())
        run_main(COMPARE_ARGS + ["--json"], transport=transport)
        paths = [call["path"] for call in transport.calls]
        self.assertEqual(paths.count(PROJECT_PATH), 1)
        self.assertEqual(paths.count(EVENTS_PATH), 6)
        for call in transport.calls:
            self.assertEqual(call["headers"]["Authorization"], "Bearer test-token")
        aggregate = transport.calls[1]["params"]
        self.assertEqual(aggregate["dataset"], ["spans"])
        self.assertEqual(aggregate["project"], ["1000001"])
        self.assertEqual(aggregate["sort"], ["-count()"])
        self.assertEqual(aggregate["field"], pc.aggregate_fields("span.duration"))
        self.assertEqual(aggregate["start"], ["2026-10-10T14:00:00"])
        self.assertEqual(aggregate["end"], ["2026-10-11T14:00:00"])
        samples = [
            call["params"]
            for call in transport.calls
            if "trace" in call["params"].get("field", [])
        ]
        self.assertEqual(len(samples), 4)
        for params in samples:
            self.assertEqual(params["sort"], ["id"])
            self.assertEqual(params["per_page"], ["100"])
            self.assertIn(f'transaction:"{LIBRARY}"', params["query"][0])

    def test_markdown_report_names_the_verdict(self):
        code, out, _ = run_main(COMPARE_ARGS, transport=FakeTransport(sentry_route()))
        self.assertEqual(code, pc.EXIT_SLOWER)
        self.assertIn("**Overall: slower** (exit 1)", out)
        self.assertIn(f"| {LIBRARY} | 160 / 160", out)
        self.assertIn("**slower, more failures**", out)

    def test_identical_data_is_no_detectable_change(self):
        transport = FakeTransport(sentry_route(candidate_data="baseline"))
        code, out, _ = run_main(COMPARE_ARGS + ["--json"], transport=transport)
        self.assertEqual(code, pc.EXIT_NO_CHANGE)
        library = json.loads(out)["transactions"][0]
        self.assertEqual(library["p50"]["diff"], 0.0)
        self.assertEqual(library["verdict"], pc.NO_CHANGE)

    def test_too_little_traffic_is_not_enough_data(self):
        transport = FakeTransport(sentry_route())
        code, out, _ = run_main(
            COMPARE_ARGS + ["--min-count", "2000", "--json"], transport=transport
        )
        self.assertEqual(code, pc.EXIT_NOT_ENOUGH_DATA)
        self.assertEqual(json.loads(out)["overall"]["verdict"], pc.NOT_ENOUGH_DATA)
        self.assertEqual(len(transport.calls), 3)

    def test_rejected_token_exits_with_the_auth_code(self):
        code, _, err = run_main(
            COMPARE_ARGS, transport=FakeTransport(lambda p, q: (401, {}, b"{}"))
        )
        self.assertEqual(code, pc.EXIT_AUTH)
        self.assertIn("HTTP 401", err)


PINNED_P50_DIFF = 11.6705
PINNED_P50_CI = [6.881525, 18.8584125]
PINNED_P95_CI = [12.03550875, 18.80014375]


if __name__ == "__main__":
    unittest.main()
