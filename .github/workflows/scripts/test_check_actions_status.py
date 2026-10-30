import unittest
from unittest.mock import MagicMock, call, patch

import requests

from check_actions_status import (
    get_paginated_items,
    make_api_request,
    process_check_runs,
)


PASSING_RUN = {
    "id": 102,
    "workflow_id": 7,
    "head_sha": "abc123",
    "event": "pull_request",
    "run_number": 12,
    "run_attempt": 1,
    "status": "completed",
    "conclusion": "success",
}

CANCELED_RUN = {
    "id": 101,
    "workflow_id": 7,
    "head_sha": "abc123",
    "event": "pull_request",
    "run_number": 11,
    "run_attempt": 1,
    "status": "completed",
    "conclusion": "cancelled",
}

CANCELED_CHECK = {
    "id": 1001,
    "name": "Build",
    "status": "completed",
    "conclusion": "cancelled",
    "details_url": "https://github.com/example/repo/actions/runs/101/job/1001",
}


class ProcessCheckRunsTests(unittest.TestCase):
    def test_accepts_skipped_check(self):
        skipped_check = {
            **CANCELED_CHECK,
            "conclusion": "skipped",
        }

        self.assertEqual(process_check_runs([skipped_check]), (False, True))

    def test_accepts_cancellation_superseded_by_passing_workflow_run(self):
        result = process_check_runs([CANCELED_CHECK], [CANCELED_RUN, PASSING_RUN])

        self.assertEqual(result, (False, True))

    def test_accepts_passing_concurrent_run_despite_reversed_run_order(self):
        source = {**CANCELED_RUN, "created_at": "2026-09-09T08:51:20Z"}
        candidate = {
            **PASSING_RUN,
            "run_number": 10,
            "created_at": source["created_at"],
        }
        self.assertEqual(
            process_check_runs([CANCELED_CHECK], [source, candidate]), (False, True)
        )

    def test_rejects_older_or_unrelated_passing_run(self):
        source = {**CANCELED_RUN, "created_at": "2026-09-09T08:51:20Z"}
        candidate = {
            **PASSING_RUN,
            "run_number": 10,
            "created_at": source["created_at"],
        }
        for change in (
            {"created_at": "2026-09-09T08:51:19Z"},
            {"created_at": None},
            {"head_sha": "different"},
            {"event": "merge_group"},
            {"conclusion": "failure"},
        ):
            with self.subTest(change=change):
                self.assertEqual(
                    process_check_runs(
                        [CANCELED_CHECK], [source, {**candidate, **change}]
                    ),
                    (False, False),
                )

    def test_rejects_cancellation_without_replacement(self):
        result = process_check_runs([CANCELED_CHECK], [CANCELED_RUN])

        self.assertEqual(result, (False, False))

    def test_rejects_cancellation_replaced_by_failed_workflow_run(self):
        failed_run = {**PASSING_RUN, "conclusion": "failure"}

        result = process_check_runs([CANCELED_CHECK], [CANCELED_RUN, failed_run])

        self.assertEqual(result, (False, False))

    def test_rejects_cancellation_replaced_by_different_workflow(self):
        unrelated_run = {**PASSING_RUN, "workflow_id": 8}

        result = process_check_runs([CANCELED_CHECK], [CANCELED_RUN, unrelated_run])

        self.assertEqual(result, (False, False))

    def test_rejects_external_cancellation(self):
        external_check = {
            **CANCELED_CHECK,
            "details_url": "https://checks.example.com/result/1001",
        }

        result = process_check_runs([external_check], [CANCELED_RUN, PASSING_RUN])

        self.assertEqual(result, (False, False))

    def test_ignores_an_advisory_reviewer_that_could_not_run(self):
        """Copilot code review concludes "failure" when Copilot itself is out
        of quota (HTTP 402); that says nothing about the code and must not
        block the merge."""
        copilot = {
            "id": 7,
            "name": "copilot-pull-request-reviewer",
            "status": "completed",
            "conclusion": "failure",
        }
        self.assertEqual(process_check_runs([copilot]), (False, True))

    def test_an_advisory_reviewer_still_running_does_not_hold_the_check(self):
        copilot = {
            "id": 7,
            "name": "copilot-pull-request-reviewer",
            "status": "in_progress",
            "conclusion": None,
        }
        self.assertEqual(process_check_runs([copilot]), (False, True))

    def test_a_real_failure_beside_an_advisory_one_still_fails(self):
        copilot = {
            "id": 7,
            "name": "copilot-pull-request-reviewer",
            "status": "completed",
            "conclusion": "failure",
        }
        failed = {
            "id": 8,
            "name": "test",
            "status": "completed",
            "conclusion": "failure",
        }
        self.assertEqual(process_check_runs([copilot, failed]), (False, False))

    def test_a_failing_journeys_leg_is_advisory_but_smoke_still_blocks(self):
        journeys = {
            "id": 9,
            "name": "end-to-end tests (journeys)",
            "status": "completed",
            "conclusion": "failure",
        }
        smoke = {**journeys, "id": 10, "name": "end-to-end tests (smoke)"}
        self.assertEqual(process_check_runs([journeys]), (False, True))
        self.assertEqual(process_check_runs([journeys, smoke]), (False, False))

    def test_rejects_real_failure(self):
        failed_check = {**CANCELED_CHECK, "conclusion": "failure"}

        self.assertEqual(process_check_runs([failed_check]), (False, False))


class PaginationTests(unittest.TestCase):
    @patch("check_actions_status.requests.get")
    def test_gets_every_page(self, get: MagicMock):
        first_response = MagicMock()
        first_response.json.return_value = {"check_runs": [{"id": 1}]}
        first_response.links = {"next": {"url": "page-2"}}
        second_response = MagicMock()
        second_response.json.return_value = {"check_runs": [{"id": 2}]}
        second_response.links = {}
        get.side_effect = [first_response, second_response]

        result = get_paginated_items("page-1", {}, "check_runs")

        self.assertEqual(result, [{"id": 1}, {"id": 2}])
        self.assertEqual(get.call_count, 2)


class ApiRequestTests(unittest.TestCase):
    @staticmethod
    def response(status: int = 200) -> requests.Response:
        response = requests.Response()
        response.status_code = status
        response.url = "https://api.github.com/repos/example/repo/check-runs"
        response._content = b'{"check_runs": [{"id": 1}]}'
        return response

    @patch("check_actions_status.time.sleep")
    @patch("check_actions_status.requests.get")
    def test_recovers_from_transient_server_errors(self, get, sleep):
        for status in (500, 502, 503, 504):
            with self.subTest(status=status):
                get.reset_mock()
                sleep.reset_mock()
                get.side_effect = [self.response(status), self.response()]

                self.assertEqual(
                    make_api_request("check-runs", {"Accept": "application/json"}),
                    ({"check_runs": [{"id": 1}]}, None),
                )
                self.assertEqual(get.call_count, 2)
                sleep.assert_called_once_with(2)

    @patch("check_actions_status.time.sleep")
    @patch("check_actions_status.requests.get")
    def test_recovers_from_connection_errors_and_timeouts(self, get, sleep):
        get.side_effect = [
            requests.ConnectionError("connection reset"),
            requests.Timeout("read timeout"),
            self.response(),
        ]

        self.assertEqual(
            make_api_request("check-runs", {}),
            ({"check_runs": [{"id": 1}]}, None),
        )
        self.assertEqual(sleep.call_args_list, [call(2), call(4)])

    @patch("check_actions_status.time.sleep")
    @patch("check_actions_status.requests.get")
    def test_persistent_server_errors_fail_closed_after_bounded_retries(
        self, get, sleep
    ):
        get.return_value = self.response(500)

        with self.assertRaises(SystemExit) as raised:
            make_api_request("check-runs", {})

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(get.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(2), call(4)])

    @patch("check_actions_status.time.sleep")
    @patch("check_actions_status.requests.get")
    def test_client_errors_fail_closed_without_retrying(self, get, sleep):
        for status in (401, 403, 404, 422, 429):
            with self.subTest(status=status):
                get.reset_mock()
                get.return_value = self.response(status)

                with self.assertRaises(SystemExit) as raised:
                    make_api_request("check-runs", {})

                self.assertEqual(raised.exception.code, 1)
                get.assert_called_once_with("check-runs", headers={}, timeout=10)
                sleep.assert_not_called()

    @patch("check_actions_status.time.sleep")
    @patch("check_actions_status.requests.get")
    def test_pagination_keeps_results_when_a_later_page_needs_retry(self, get, sleep):
        first_page = self.response()
        first_page.headers["Link"] = '<page-2>; rel="next"'
        second_page = self.response()
        second_page._content = b'{"check_runs": [{"id": 2}]}'
        get.side_effect = [first_page, self.response(503), second_page]

        self.assertEqual(
            get_paginated_items("page-1", {}, "check_runs"),
            [{"id": 1}, {"id": 2}],
        )
        self.assertEqual(
            get.call_args_list,
            [
                call("page-1", headers={}, timeout=10),
                call("page-2", headers={}, timeout=10),
                call("page-2", headers={}, timeout=10),
            ],
        )


if __name__ == "__main__":
    unittest.main()
