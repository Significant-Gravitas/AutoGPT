import unittest
from unittest.mock import MagicMock, patch

from check_actions_status import get_paginated_items, process_check_runs


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
        result = process_check_runs(
            [CANCELED_CHECK], [CANCELED_RUN, PASSING_RUN]
        )

        self.assertEqual(result, (False, True))

    def test_rejects_cancellation_without_replacement(self):
        result = process_check_runs([CANCELED_CHECK], [CANCELED_RUN])

        self.assertEqual(result, (False, False))

    def test_rejects_cancellation_replaced_by_failed_workflow_run(self):
        failed_run = {**PASSING_RUN, "conclusion": "failure"}

        result = process_check_runs(
            [CANCELED_CHECK], [CANCELED_RUN, failed_run]
        )

        self.assertEqual(result, (False, False))

    def test_rejects_cancellation_replaced_by_different_workflow(self):
        unrelated_run = {**PASSING_RUN, "workflow_id": 8}

        result = process_check_runs(
            [CANCELED_CHECK], [CANCELED_RUN, unrelated_run]
        )

        self.assertEqual(result, (False, False))

    def test_rejects_external_cancellation(self):
        external_check = {
            **CANCELED_CHECK,
            "details_url": "https://checks.example.com/result/1001",
        }

        result = process_check_runs(
            [external_check], [CANCELED_RUN, PASSING_RUN]
        )

        self.assertEqual(result, (False, False))

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


if __name__ == "__main__":
    unittest.main()
