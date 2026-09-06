import json
import subprocess
import unittest
from unittest.mock import patch

import detect_overlaps


def completed(*, stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def pull_request_page(
    nodes: list[dict], *, total_count: int, has_next_page: bool, cursor: str | None
):
    return completed(
        stdout=json.dumps(
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "totalCount": total_count,
                            "edges": [{"node": node} for node in nodes],
                            "pageInfo": {
                                "endCursor": cursor,
                                "hasNextPage": has_next_page,
                            },
                        }
                    }
                }
            }
        )
    )


def pull_request_node(number: int):
    return {
        "number": number,
        "title": f"PR {number}",
        "url": f"https://example.invalid/{number}",
        "updatedAt": "2026-09-02T00:00:00Z",
        "author": {"login": "tester"},
        "headRefName": f"feature-{number}",
        "headRefOid": str(number) * 40,
        "baseRefName": "dev",
        "changedFiles": 1,
        "files": {
            "totalCount": 1,
            "nodes": [{"path": "file.py", "changeType": "MODIFIED"}],
            "pageInfo": {"hasNextPage": False},
        },
    }


class FindOverlappingPullRequestsTests(unittest.TestCase):
    def make_current(self, number: int, head_sha: str):
        return detect_overlaps.PullRequest(
            number=number,
            title=head_sha,
            author="tester",
            url=f"https://example.invalid/{number}",
            head_ref="feature",
            base_ref="dev",
            files=["changed.py"],
            changed_ranges={},
        )

    def make_open_pr(self, number: int, head_sha: str):
        return {"number": number, "head_sha": head_sha}

    def test_ref_mode_excludes_pull_request_with_matching_head_sha(self):
        head_sha = "a" * 40
        current = self.make_current(0, head_sha)
        matching = self.make_open_pr(41, head_sha)
        different = self.make_open_pr(42, "b" * 40)

        with (
            patch.object(
                detect_overlaps,
                "query_open_prs",
                return_value=[matching, different],
            ),
            patch.object(
                detect_overlaps,
                "find_file_overlap_candidates",
                return_value=[],
            ) as find_candidates,
        ):
            detect_overlaps.find_overlapping_prs(
                "owner", "repo", "dev", current, 0, skip_merge_test=True
            )

        self.assertEqual(find_candidates.call_args.args[1], [different])

    def test_pull_request_mode_filters_by_number_only(self):
        head_sha = "a" * 40
        current = self.make_current(42, head_sha)
        same_number = self.make_open_pr(42, "b" * 40)
        same_head = self.make_open_pr(43, head_sha)

        with (
            patch.object(
                detect_overlaps,
                "query_open_prs",
                return_value=[same_number, same_head],
            ),
            patch.object(
                detect_overlaps,
                "find_file_overlap_candidates",
                return_value=[],
            ) as find_candidates,
        ):
            detect_overlaps.find_overlapping_prs(
                "owner", "repo", "dev", current, 42, skip_merge_test=True
            )

        self.assertEqual(find_candidates.call_args.args[1], [same_head])

    def test_binary_overlap_is_sent_to_authoritative_merge_test(self):
        current = self.make_current(42, "a" * 40)
        other = self.make_current(43, "b" * 40)
        overlap = detect_overlaps.Overlap(
            current,
            other,
            ["changed.py"],
            {},
            needs_merge_test=True,
        )
        pr_data = self.make_open_pr(43, "b" * 40)

        with (
            patch.object(detect_overlaps, "query_open_prs", return_value=[pr_data]),
            patch.object(
                detect_overlaps,
                "find_file_overlap_candidates",
                return_value=[(pr_data, ["changed.py"])],
            ),
            patch.object(
                detect_overlaps,
                "analyze_pr_overlap",
                return_value=(overlap, {}),
            ),
            patch.object(detect_overlaps, "run_batch_merge_tests") as merge_tests,
        ):
            detect_overlaps.find_overlapping_prs(
                "owner", "repo", "dev", current, 42, skip_merge_test=False
            )

        merge_tests.assert_called_once_with("owner", "repo", "dev", current, [overlap])

    def test_delete_modify_overlap_is_sent_to_authoritative_merge_test(self):
        current = self.make_current(42, "a" * 40)
        current.changed_ranges = {
            "changed.py": detect_overlaps.ChangedFile(
                path="changed.py",
                additions=[],
                deletions=[(1, 100)],
                is_deleted=True,
            )
        }
        other = self.make_current(43, "b" * 40)
        other.changed_ranges = {
            "changed.py": detect_overlaps.ChangedFile(
                path="changed.py",
                additions=[(200, 200)],
                deletions=[(200, 200)],
            )
        }
        needs_merge_test = detect_overlaps.requires_authoritative_merge_test(
            current.changed_ranges,
            other.changed_ranges,
            ["changed.py"],
        )
        overlap = detect_overlaps.Overlap(
            current,
            other,
            ["changed.py"],
            {},
            needs_merge_test=needs_merge_test,
        )
        pr_data = self.make_open_pr(43, "b" * 40)

        with (
            patch.object(detect_overlaps, "query_open_prs", return_value=[pr_data]),
            patch.object(
                detect_overlaps,
                "find_file_overlap_candidates",
                return_value=[(pr_data, ["changed.py"])],
            ),
            patch.object(
                detect_overlaps,
                "analyze_pr_overlap",
                return_value=(overlap, other.changed_ranges),
            ),
            patch.object(detect_overlaps, "run_batch_merge_tests") as merge_tests,
        ):
            detect_overlaps.find_overlapping_prs(
                "owner", "repo", "dev", current, 42, skip_merge_test=False
            )

        self.assertTrue(needs_merge_test)
        merge_tests.assert_called_once_with("owner", "repo", "dev", current, [overlap])


class PullRequestInventoryTests(unittest.TestCase):
    def test_rename_previous_filename_matches_old_path_edit(self):
        other_pr = {
            "files": ["new.py"],
            "file_aliases": ["old.py"],
            "updated_at": None,
        }

        other_renamed = detect_overlaps.find_file_overlap_candidates(
            ["old.py"], [other_pr]
        )
        current_renamed = detect_overlaps.find_file_overlap_candidates(
            ["new.py", "old.py"],
            [{"files": ["old.py"], "file_aliases": [], "updated_at": None}],
        )

        self.assertEqual(other_renamed[0][1], ["old.py"])
        self.assertEqual(current_renamed[0][1], ["old.py"])

    def test_get_pr_files_paginates_and_matches_expected_count(self):
        first_page = [{"filename": f"file-{index}.py"} for index in range(100)]
        second_page = [{"filename": "file-100.py"}]

        with patch.object(
            detect_overlaps,
            "run_gh",
            side_effect=[
                completed(stdout=json.dumps(first_page)),
                completed(stdout=json.dumps(second_page)),
            ],
        ) as run_gh:
            files = detect_overlaps.get_pr_files("owner", "repo", 17, 101)

        self.assertEqual(len(files.paths), 101)
        self.assertEqual(files.aliases, [])
        self.assertIn("page=1", run_gh.call_args_list[0].args[0][-1])
        self.assertIn("page=2", run_gh.call_args_list[1].args[0][-1])

    def test_get_pr_files_rejects_duplicate_filename(self):
        page = [{"filename": "same.py"}, {"filename": "same.py"}]

        with patch.object(
            detect_overlaps,
            "run_gh",
            return_value=completed(stdout=json.dumps(page)),
        ):
            with self.assertRaisesRegex(
                detect_overlaps.OverlapInfrastructureError,
                "duplicate file.*same.py",
            ):
                detect_overlaps.get_pr_files("owner", "repo", 17, 2)

    def test_get_pr_files_rejects_changed_file_count_mismatch(self):
        with patch.object(
            detect_overlaps,
            "run_gh",
            return_value=completed(stdout=json.dumps([{"filename": "only.py"}])),
        ):
            with self.assertRaisesRegex(
                detect_overlaps.OverlapInfrastructureError,
                "reported 2 changed files.*returned 1",
            ):
                detect_overlaps.get_pr_files("owner", "repo", 17, 2)

    def test_query_open_prs_rejects_duplicate_prs_across_pages(self):
        node = pull_request_node(17)

        with (
            patch.object(
                detect_overlaps,
                "run_gh",
                side_effect=[
                    pull_request_page(
                        [node], total_count=2, has_next_page=True, cursor="next"
                    ),
                    pull_request_page(
                        [node], total_count=2, has_next_page=False, cursor=None
                    ),
                ],
            ),
            patch.object(
                detect_overlaps,
                "get_pr_files",
                return_value=detect_overlaps.FileInventory(["file.py"], []),
            ),
        ):
            with self.assertRaisesRegex(
                detect_overlaps.OverlapInfrastructureError, "duplicate PR #17"
            ):
                detect_overlaps.query_open_prs_once("owner", "repo", "dev")

    def test_query_open_prs_rejects_total_count_mismatch(self):
        with (
            patch.object(
                detect_overlaps,
                "run_gh",
                return_value=pull_request_page(
                    [pull_request_node(17)],
                    total_count=2,
                    has_next_page=False,
                    cursor=None,
                ),
            ),
            patch.object(
                detect_overlaps,
                "get_pr_files",
                return_value=detect_overlaps.FileInventory(["file.py"], []),
            ),
        ):
            with self.assertRaisesRegex(
                detect_overlaps.OverlapInfrastructureError,
                "reported 2 open PRs.*returned 1",
            ):
                detect_overlaps.query_open_prs_once("owner", "repo", "dev")

    def test_query_open_prs_retries_one_inconsistent_snapshot(self):
        transient = detect_overlaps.OverlapInfrastructureError("snapshot moved")
        with patch.object(
            detect_overlaps,
            "query_open_prs_once",
            side_effect=[transient, [{"number": 17}]],
        ) as query_once:
            prs = detect_overlaps.query_open_prs("owner", "repo", "dev")

        self.assertEqual(prs, [{"number": 17}])
        self.assertEqual(query_once.call_count, 2)

    def test_query_open_prs_uses_complete_embedded_file_list(self):
        with (
            patch.object(
                detect_overlaps,
                "run_gh",
                return_value=pull_request_page(
                    [pull_request_node(17)],
                    total_count=1,
                    has_next_page=False,
                    cursor=None,
                ),
            ),
            patch.object(detect_overlaps, "get_pr_files") as get_pr_files,
        ):
            prs = detect_overlaps.query_open_prs("owner", "repo", "dev")

        self.assertEqual(prs[0]["files"], ["file.py"])
        get_pr_files.assert_not_called()

    def test_query_open_prs_rest_fetches_truncated_file_list(self):
        node = pull_request_node(17)
        node["changedFiles"] = 101
        node["files"] = {
            "totalCount": 101,
            "nodes": [
                {"path": f"file-{index}.py", "changeType": "MODIFIED"}
                for index in range(100)
            ],
            "pageInfo": {"hasNextPage": True},
        }
        complete_files = [f"file-{index}.py" for index in range(101)]

        with (
            patch.object(
                detect_overlaps,
                "run_gh",
                return_value=pull_request_page(
                    [node], total_count=1, has_next_page=False, cursor=None
                ),
            ),
            patch.object(
                detect_overlaps,
                "get_pr_files",
                return_value=detect_overlaps.FileInventory(complete_files, []),
            ) as get_pr_files,
        ):
            prs = detect_overlaps.query_open_prs("owner", "repo", "dev")

        self.assertEqual(prs[0]["files"], complete_files)
        get_pr_files.assert_called_once_with(
            "owner",
            "repo",
            17,
            101,
            expected_head_sha=node["headRefOid"],
        )

    def test_query_open_prs_rest_fetches_duplicate_embedded_files(self):
        node = pull_request_node(17)
        node["changedFiles"] = 2
        node["files"] = {
            "totalCount": 2,
            "nodes": [
                {"path": "same.py", "changeType": "MODIFIED"},
                {"path": "same.py", "changeType": "MODIFIED"},
            ],
            "pageInfo": {"hasNextPage": False},
        }

        with (
            patch.object(
                detect_overlaps,
                "run_gh",
                return_value=pull_request_page(
                    [node], total_count=1, has_next_page=False, cursor=None
                ),
            ),
            patch.object(
                detect_overlaps,
                "get_pr_files",
                return_value=detect_overlaps.FileInventory(
                    ["first.py", "second.py"], []
                ),
            ) as get_pr_files,
        ):
            prs = detect_overlaps.query_open_prs("owner", "repo", "dev")

        self.assertEqual(prs[0]["files"], ["first.py", "second.py"])
        get_pr_files.assert_called_once_with(
            "owner",
            "repo",
            17,
            2,
            expected_head_sha=node["headRefOid"],
        )

    def test_query_open_prs_rest_fetches_rename_aliases(self):
        node = pull_request_node(17)
        node["files"]["nodes"][0]["path"] = "new.py"
        node["files"]["nodes"][0]["changeType"] = "RENAMED"

        with (
            patch.object(
                detect_overlaps,
                "run_gh",
                return_value=pull_request_page(
                    [node], total_count=1, has_next_page=False, cursor=None
                ),
            ),
            patch.object(
                detect_overlaps,
                "get_pr_files",
                return_value=detect_overlaps.FileInventory(["new.py"], ["old.py"]),
            ),
        ):
            prs = detect_overlaps.query_open_prs("owner", "repo", "dev")

        self.assertEqual(prs[0]["files"], ["new.py"])
        self.assertEqual(prs[0]["file_aliases"], ["old.py"])

    def test_ref_details_include_parsed_rename_alias(self):
        rename_diff = (
            "diff --git a/old.py b/new.py\n"
            "similarity index 100%\n"
            "rename from old.py\n"
            "rename to new.py\n"
        )
        with (
            patch.object(detect_overlaps, "resolve_git_commit", return_value="head"),
            patch.object(detect_overlaps, "resolve_base_commit", return_value="base"),
            patch.object(
                detect_overlaps,
                "run_git",
                side_effect=[
                    completed(stdout="new.py\n"),
                    completed(stdout=rename_diff),
                ],
            ),
        ):
            pr = detect_overlaps.fetch_ref_details("HEAD", "dev", "owner", "repo")

        self.assertEqual(pr.files, ["new.py"])
        self.assertEqual(pr.file_aliases, ["old.py"])

    def test_get_pr_files_includes_previous_filename_alias(self):
        page = [
            {
                "filename": "new.py",
                "status": "renamed",
                "previous_filename": "old.py",
            }
        ]

        with patch.object(
            detect_overlaps,
            "run_gh",
            return_value=completed(stdout=json.dumps(page)),
        ):
            inventory = detect_overlaps.get_pr_files("owner", "repo", 17, 1)

        self.assertEqual(inventory.paths, ["new.py"])
        self.assertEqual(inventory.aliases, ["old.py"])


class PullRequestDiffTests(unittest.TestCase):
    def test_cli_diff_is_checked_against_inventoried_head(self):
        with patch.object(
            detect_overlaps,
            "run_gh",
            side_effect=[
                completed(stdout="diff body\n"),
                completed(stdout=json.dumps({"headRefOid": "expected-head"})),
            ],
        ):
            diff = detect_overlaps.get_pr_diff(
                17, "dev", expected_head_sha="expected-head"
            )

        self.assertEqual(diff, "diff body\n")

    def test_fallback_diff_rejects_changed_head(self):
        with (
            patch.object(
                detect_overlaps,
                "run_gh",
                return_value=completed(returncode=1, stderr="HTTP 406"),
            ),
            patch.object(
                detect_overlaps,
                "run_git",
                side_effect=[
                    completed(),
                    completed(),
                    completed(stdout="base-sha\n"),
                    completed(stdout="different-head\n"),
                ],
            ),
        ):
            with self.assertRaisesRegex(
                detect_overlaps.OverlapInfrastructureError,
                "head changed before its diff",
            ):
                detect_overlaps.get_pr_diff(
                    17, "dev", expected_head_sha="expected-head"
                )

    def test_falls_back_to_checked_local_git_diff(self):
        with (
            patch.object(
                detect_overlaps,
                "run_gh",
                return_value=completed(returncode=1, stderr="HTTP 406"),
            ),
            patch.object(
                detect_overlaps,
                "run_git",
                side_effect=[
                    completed(),
                    completed(),
                    completed(stdout="base-sha\n"),
                    completed(stdout="head-sha\n"),
                    completed(stdout="merge-base\n"),
                    completed(stdout="diff body\n"),
                ],
            ) as run_git,
        ):
            diff = detect_overlaps.get_pr_diff(17, "dev")

        self.assertEqual(diff, "diff body\n")
        self.assertEqual(
            run_git.call_args_list[0].args[0],
            [
                "fetch",
                "--no-tags",
                "origin",
                "+refs/heads/dev:refs/remotes/origin/dev",
            ],
        )
        self.assertEqual(
            run_git.call_args_list[1].args[0],
            ["fetch", "--no-tags", "origin", "pull/17/head"],
        )
        self.assertEqual(
            run_git.call_args_list[-1].args[0],
            [
                "diff",
                "--no-ext-diff",
                "--unified=0",
                "--find-renames",
                "merge-base",
                "head-sha",
                "--",
            ],
        )

    def test_each_fallback_git_stage_is_fatal(self):
        stages = [
            "refresh origin/dev for PR #17 diff failed",
            "fetch PR #17 for diff failed",
            "resolve origin/dev for PR #17 diff failed",
            "resolve fetched PR #17 head failed",
            "find merge base for PR #17 failed",
            "generate fallback diff for PR #17 failed",
        ]
        successful_results = [
            completed(),
            completed(),
            completed(stdout="base-sha\n"),
            completed(stdout="head-sha\n"),
            completed(stdout="merge-base\n"),
            completed(stdout="diff body\n"),
        ]

        for failed_index, expected_error in enumerate(stages):
            with self.subTest(stage=expected_error):
                results = list(successful_results)
                results[failed_index] = completed(returncode=1, stderr="stage failed")
                with (
                    patch.object(
                        detect_overlaps,
                        "run_gh",
                        return_value=completed(returncode=1, stderr="HTTP 406"),
                    ),
                    patch.object(
                        detect_overlaps,
                        "run_git",
                        side_effect=results,
                    ),
                ):
                    with self.assertRaisesRegex(
                        detect_overlaps.OverlapInfrastructureError,
                        expected_error,
                    ):
                        detect_overlaps.get_pr_diff(17, "dev")


class DiffParsingTests(unittest.TestCase):
    def test_deleted_file_hunk_is_attached_to_deleted_file(self):
        changes = detect_overlaps.parse_diff_ranges(
            "diff --git a/kept.py b/kept.py\n"
            "--- a/kept.py\n"
            "+++ b/kept.py\n"
            "@@ -1 +1 @@\n"
            "diff --git a/deleted.py b/deleted.py\n"
            "deleted file mode 100644\n"
            "--- a/deleted.py\n"
            "+++ /dev/null\n"
            "@@ -3,2 +0,0 @@\n"
        )

        self.assertEqual(changes["kept.py"].deletions, [(1, 1)])
        self.assertEqual(changes["deleted.py"].deletions, [(3, 4)])
        self.assertTrue(changes["deleted.py"].is_deleted)

    def test_binary_file_is_represented_without_text_ranges(self):
        changes = detect_overlaps.parse_diff_ranges(
            "diff --git a/image.png b/image.png\n"
            "index 123..456 100644\n"
            "Binary files a/image.png and b/image.png differ\n"
        )

        self.assertEqual(changes["image.png"].additions, [])
        self.assertEqual(changes["image.png"].deletions, [])

    def test_pure_rename_is_represented_as_merge_candidate(self):
        changes = detect_overlaps.parse_diff_ranges(
            "diff --git a/old.py b/new.py\n"
            "similarity index 100%\n"
            "rename from old.py\n"
            "rename to new.py\n"
        )

        renamed = changes["new.py"]
        self.assertTrue(renamed.is_rename)
        self.assertEqual(renamed.old_path, "old.py")
        self.assertTrue(
            detect_overlaps.requires_authoritative_merge_test(
                changes, changes, ["new.py"]
            )
        )

    def test_old_rename_alias_is_classified_and_formatted_as_medium_risk(self):
        renamed = detect_overlaps.ChangedFile(
            path="new.py",
            additions=[],
            deletions=[],
            is_rename=True,
            old_path="old.py",
        )
        edited = detect_overlaps.ChangedFile(
            path="old.py",
            additions=[(1, 1)],
            deletions=[(1, 1)],
        )
        current = detect_overlaps.PullRequest(
            42, "current", "tester", "url", "current", "dev", ["new.py"], {}
        )
        other = detect_overlaps.PullRequest(
            43, "other", "tester", "url", "other", "dev", ["old.py"], {}
        )
        overlap = detect_overlaps.Overlap(current, other, ["old.py"], {})

        risk = detect_overlaps.classify_overlap_risk(
            overlap,
            {"new.py": renamed},
            {"old.py": edited},
        )
        lines = []
        detect_overlaps.format_medium_risk_section(
            [(overlap, risk)],
            42,
            {"new.py": renamed},
            {43: {"old.py": edited}},
            lines,
        )

        self.assertEqual(risk, "medium")
        self.assertTrue(any("renamed/moved" in line for line in lines))


class MergePlumbingTests(unittest.TestCase):
    def test_merge_fetch_rejects_changed_head(self):
        with patch.object(
            detect_overlaps,
            "run_git",
            return_value=completed(stdout="different-head\n"),
        ):
            with self.assertRaisesRegex(
                detect_overlaps.OverlapInfrastructureError,
                "head changed before merge testing",
            ):
                detect_overlaps.verify_fetched_ref(
                    "repo", "pr-17", "expected-head", "PR #17"
                )

    def test_non_conflict_merge_failure_is_fatal(self):
        with patch.object(
            detect_overlaps,
            "run_git",
            side_effect=[
                completed(returncode=128, stderr="fatal: broken repository"),
                completed(stdout=""),
            ],
        ):
            with self.assertRaisesRegex(
                detect_overlaps.OverlapInfrastructureError,
                "failed without unmerged files",
            ):
                detect_overlaps.try_merge_pr("repo", 17)

    def test_unmerged_index_marks_a_genuine_conflict(self):
        conflict_info = detect_overlaps.ConflictInfo(path="conflict.py")
        with (
            patch.object(
                detect_overlaps,
                "run_git",
                side_effect=[
                    completed(returncode=1, stderr="CONFLICT"),
                    completed(stdout="conflict.py\n"),
                    completed(stdout="UU conflict.py\n"),
                    completed(),
                ],
            ),
            patch.object(
                detect_overlaps,
                "analyze_conflict_markers",
                return_value=conflict_info,
            ),
        ):
            conflict = detect_overlaps.try_merge_pr("repo", 17)

        self.assertEqual(conflict[0], ["conflict.py"])

    def test_clone_failure_is_fatal(self):
        with patch.object(
            detect_overlaps,
            "run_git",
            return_value=completed(returncode=128, stderr="clone failed"),
        ):
            with self.assertRaisesRegex(
                detect_overlaps.OverlapInfrastructureError, "clone repository failed"
            ):
                detect_overlaps.clone_repo("owner", "repo", "dev", "temp")

    def test_clone_fetches_complete_objects_for_merge_testing(self):
        with patch.object(
            detect_overlaps,
            "run_git",
            return_value=completed(),
        ) as run_git:
            detect_overlaps.clone_repo("owner", "repo", "dev", "temp")

        clone_args = run_git.call_args.args[0]
        self.assertNotIn("--filter=blob:none", clone_args)
        self.assertIn("--single-branch", clone_args)


if __name__ == "__main__":
    unittest.main()
