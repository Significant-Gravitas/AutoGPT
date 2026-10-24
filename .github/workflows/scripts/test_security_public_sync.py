import contextlib
import copy
import io
import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import security_public_api as api
import security_public_publish as publish
import security_public_sync as sync
from security_release_notes import BASE_URL, END, START, fetch_advisories
from test_security_release_notes import TAG, advisory

CANARY = "PRIVATE-CANARY-DO-NOT-PUBLISH"
EMPTY = {"schema_version": 1, "advisories": []}
POLICY = f"# Policy\n\n{sync.CREDIT_START}\nold\n{sync.CREDIT_END}\n\nHistorical credit stays.\n"


class AnonymousApiTests(unittest.TestCase):
    def test_ambient_tokens_proxies_and_credentials_are_not_used(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = b"[]"
        response.headers = {}
        with patch.dict(
            os.environ,
            {
                "GH_TOKEN": CANARY,
                "GITHUB_TOKEN": CANARY,
                "HTTPS_PROXY": "http://example.invalid",
                "GH_HOST": "example.invalid",
            },
        ), patch.object(api, "build_opener") as build:
            build.return_value.open.return_value = response
            self.assertEqual(
                api.public_json("security-advisories?state=published")[0], []
            )
        request = build.return_value.open.call_args.args[0]
        self.assertEqual(request.host, "api.github.com")
        self.assertNotIn("authorization", {k.lower() for k in request.headers})
        self.assertNotIn(CANARY, repr(request.headers))
        self.assertEqual(build.call_args.args[0].proxies, {})
        self.assertIsInstance(build.call_args.args[1], api.NoRedirects)
        self.assertEqual(build.return_value.open.call_args.kwargs["timeout"], 20)

    def test_failures_never_log_response_bodies_or_fall_back_to_auth(self):
        for error in (
            HTTPError(
                "https://api.github.com", 403, CANARY, {}, io.BytesIO(CANARY.encode())
            ),
            TimeoutError(CANARY),
            ValueError(CANARY),
        ):
            with self.subTest(error=type(error).__name__), patch.object(
                api, "build_opener"
            ) as build:
                build.return_value.open.side_effect = error
                with self.assertRaises(api.PublicDataError) as raised:
                    fetch_advisories()
                self.assertNotIn(CANARY, str(raised.exception))
                self.assertEqual(build.return_value.open.call_count, 1)

    def test_redirects_and_oversized_responses_fail_closed(self):
        with self.assertRaises(api.PublicDataError):
            api.NoRedirects().redirect_request(
                None, None, 302, "", {}, "https://example.invalid"
            )
        with patch.object(api, "build_opener") as build:
            response = build.return_value.open.return_value.__enter__.return_value
            response.status = 200
            response.read.return_value = b"x" * (api.MAX_BYTES + 1)
            with self.assertRaises(api.PublicDataError):
                api.public_json("releases")

    def test_cursor_pagination_is_bounded_and_stays_on_the_published_endpoint(self):
        endpoint = "security-advisories"
        next_url = (
            api.API_ROOT + endpoint + "?per_page=100&state=published&after=cursor"
        )
        with patch.object(
            api,
            "public_json",
            side_effect=[
                ([{"id": 1}], {"Link": f'<{next_url}>; rel="next"'}),
                ([{"id": 2}], {}),
            ],
        ) as read:
            self.assertEqual(
                api.public_pages(endpoint, "state=published"), [{"id": 1}, {"id": 2}]
            )
            self.assertIn("after=cursor", read.call_args.args[0])
        canonical = f"https://api.github.com/repositories/{api.REPOSITORY_ID}/releases?per_page=100&page=2"
        with patch.object(
            api,
            "public_json",
            side_effect=[
                ([{"id": 1}], {"Link": f'<{canonical}>; rel="next"'}),
                ([{"id": 2}], {}),
            ],
        ) as read:
            self.assertEqual(len(api.public_pages("releases")), 2)
            self.assertEqual(read.call_args.args[0], "releases?per_page=100&page=2")
        for bad_url in (
            next_url.replace("api.github.com", "example.invalid"),
            next_url.replace("published", "draft"),
            next_url.replace(endpoint, "issues"),
            next_url.replace("https:", "http:"),
            next_url + "&state=draft",
        ):
            with self.subTest(url=bad_url), patch.object(
                api,
                "public_json",
                return_value=(
                    [],
                    {"Link": f'<{bad_url}>; rel="next"'},
                ),
            ) as read, self.assertRaises(api.PublicDataError):
                api.public_pages(endpoint, "state=published")
            self.assertEqual(read.call_count, 1)
        with patch.object(
            api,
            "public_json",
            return_value=(
                [],
                {"Link": f'<{next_url}>; rel="next"'},
            ),
        ), self.assertRaisesRegex(api.PublicDataError, "repeated"):
            api.public_pages(endpoint, "state=published")


class PublicProjectionTests(unittest.TestCase):
    def test_committed_snapshot_and_generated_credits_obey_the_public_schema(self):
        root = Path(__file__).resolve().parents[3]
        snapshot = sync.validate_snapshot(
            json.loads((root / sync.SNAPSHOT_PATH).read_text(encoding="utf-8"))
        )
        security = (root / "SECURITY.md").read_text(encoding="utf-8")
        self.assertEqual(sync.credit_notes(snapshot, security), security)

    def test_end_to_end_canaries_do_not_reach_files_logs_or_release_sections(self):
        public = advisory()
        public.update(
            summary=CANARY,
            private_fork={"name": CANARY},
            discussion=CANARY,
            arbitrary_new_field=CANARY,
        )
        public[
            "description"
        ] += f"\n{CANARY}\nattachment: https://example.invalid/{CANARY}"
        public["credits_detailed"].append(
            {
                "user": {"login": CANARY},
                "state": "pending",
                "type": "reporter",
            }
        )
        private = [dict(public, state=state) for state in ("draft", "triage", "closed")]
        withdrawn = dict(public, withdrawn_at="2026-01-01")
        snapshot = sync.public_snapshot([public, *private, withdrawn])
        with TemporaryDirectory() as directory, patch.object(
            sync, "fetch_advisories", return_value=[public, *private, withdrawn]
        ):
            root = Path(directory)
            (root / sync.SNAPSHOT_PATH).parent.mkdir()
            (root / sync.SNAPSHOT_PATH).write_text(
                sync.snapshot_text(EMPTY), encoding="utf-8"
            )
            (root / "SECURITY.md").write_text(POLICY, encoding="utf-8")
            output = io.StringIO()
            argv = [
                "sync",
                "credits",
                "--root",
                str(root),
                "--expected-digest",
                sync.snapshot_digest(snapshot),
            ]
            with patch("sys.argv", argv), contextlib.redirect_stdout(
                output
            ), contextlib.redirect_stderr(output):
                sync.main()
            files = {
                str(p.relative_to(root)): p.read_text(encoding="utf-8")
                for p in root.rglob("*")
                if p.is_file()
            }
            self.assertEqual(len(files), 2)
            notes = sync.public_release_notes(
                snapshot, TAG, "# Features\nKeep these.\n"
            )
            for text in [*files.values(), notes, output.getvalue()]:
                self.assertNotIn(CANARY, text)
                self.assertNotIn("example.invalid", text)
            self.assertIn("@researcher", notes)
            self.assertIn("Historical credit stays.", files["SECURITY.md"])
            self.assertEqual(
                json.loads(output.getvalue()), {"added": 1, "removed": 0, "updated": 0}
            )
            self.assertEqual(
                sync.prepare_credits(root, sync.snapshot_digest(snapshot)),
                {"added": 0, "removed": 0, "updated": 0},
            )

    def test_snapshot_rejects_unknown_fields_and_free_text(self):
        snapshot = sync.public_snapshot([advisory()])
        for key, value in (
            ("description", CANARY),
            ("ghsa_id", CANARY),
            ("cve_id", CANARY),
            ("researchers", ["@all"]),
            ("release_tags", [TAG + "/" + CANARY]),
        ):
            changed = copy.deepcopy(snapshot)
            changed["advisories"][0][key] = value
            with self.subTest(field=key), self.assertRaises(api.PublicDataError):
                sync.snapshot_text(changed)

    def test_reconciliation_handles_later_publication_and_revoked_credit(self):
        old = sync.public_snapshot([advisory()])
        changed = advisory()
        changed["cve_id"] = "CVE-2026-12345"
        changed["credits_detailed"] = []
        new = sync.public_snapshot([changed, advisory("GHSA-dddd-eeee-ffff")])
        self.assertEqual(
            sync.changes(old, new), {"added": 1, "removed": 0, "updated": 1}
        )
        self.assertEqual(
            sync.changes(new, EMPTY), {"added": 0, "removed": 2, "updated": 0}
        )
        first = sync.public_release_notes(old, TAG, "Before\n") + "\nAfter\n"
        removed = sync.public_release_notes(EMPTY, TAG, first)
        self.assertNotIn("GHSA-", removed)
        self.assertNotIn("researcher", removed)
        self.assertTrue(removed.startswith("Before\n"))
        self.assertTrue(removed.endswith("After\n"))
        self.assertEqual(sync.public_release_notes(EMPTY, TAG, removed), removed)

    def test_later_validation_release_does_not_replace_first_fixed_release(self):
        item = advisory()
        item[
            "description"
        ] += f" Validated on {BASE_URL}/releases/tag/autogpt-platform-beta-v0.10.0"
        self.assertEqual(
            sync.public_snapshot([item])["advisories"][0]["release_tags"], [TAG]
        )

    def test_classic_and_old_platform_tags_and_extra_package_keys(self):
        for tag in ("autogpt-v0.9.0", "agpt-platform-beta-v0.9.0", "v0.9.0"):
            item = advisory()
            item["description"] = item["description"].replace(TAG, tag)
            self.assertEqual(
                sync.public_snapshot([item])["advisories"][0]["release_tags"], [tag]
            )
        item = advisory()
        package = {"ecosystem": "other", "name": "example"}
        item["vulnerabilities"] = [
            {
                "package": dict(package, future_field=CANARY),
                "patched_versions": "0.9.0",
            },
            {"package": {"ecosystem": "other", "name": "other"}},
        ]
        item["description"] += (
            "\n<!-- security-release: "
            + json.dumps({"tag": TAG, "package": package})
            + " -->"
        )
        self.assertEqual(
            sync.public_snapshot([item])["advisories"][0]["release_tags"], [TAG]
        )

    def test_changed_public_data_or_malformed_markers_do_not_overwrite_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / sync.SNAPSHOT_PATH).parent.mkdir()
            previous = sync.snapshot_text(EMPTY)
            (root / sync.SNAPSHOT_PATH).write_text(previous, encoding="utf-8")
            for policy, digest in (
                (POLICY, "0" * 64),
                ("No markers", sync.snapshot_digest(EMPTY)),
            ):
                (root / "SECURITY.md").write_text(policy, encoding="utf-8")
                with patch.object(
                    sync, "fetch_advisories", return_value=[]
                ), self.assertRaises(api.PublicDataError):
                    sync.prepare_credits(root, digest)
                self.assertEqual(
                    (root / "SECURITY.md").read_text(encoding="utf-8"), policy
                )
                self.assertEqual(
                    (root / sync.SNAPSHOT_PATH).read_text(encoding="utf-8"), previous
                )

    def test_error_logs_do_not_include_sensitive_exception_text(self):
        with patch("sys.argv", ["sync", "digest"]), patch.object(
            sync, "fetch_advisories", side_effect=ValueError(CANARY)
        ), contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(
            SystemExit
        ):
            sync.main()
        self.assertNotIn(CANARY, stderr.getvalue())

    def test_repository_data_cannot_redirect_generated_files_through_symlinks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "must-not-change.txt"
            outside.write_text(CANARY, encoding="utf-8")
            try:
                (root / "SECURITY.md").symlink_to(outside)
            except OSError:
                self.skipTest("Symlink creation is not available on this host")
            with patch.object(sync, "fetch_advisories") as fetch, self.assertRaises(
                api.PublicDataError
            ):
                sync.prepare_credits(root, "0" * 64)
            fetch.assert_not_called()
            self.assertEqual(outside.read_text(encoding="utf-8"), CANARY)


class ReleaseWriterTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = sync.public_snapshot([advisory()])
        self.release = {
            "id": 123,
            "draft": False,
            "published_at": "2026-01-01",
            "tag_name": TAG,
            "body": "# Features\nOriginal\n",
        }
        self.digest = sync.snapshot_digest(self.snapshot)
        sleep = patch.object(publish.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    def test_only_managed_section_changes_and_second_run_is_noop(self):
        newer = dict(self.release, body="# Features\nMaintainer's recent edit\n")
        with patch.object(
            publish, "current_snapshot", return_value=self.snapshot
        ) as fetch, patch.object(
            publish, "public_pages", return_value=[self.release]
        ), patch.object(
            publish, "public_json", return_value=(newer, {"ETag": '"etag"'})
        ), patch.object(
            publish, "patch_release"
        ) as write:
            self.assertEqual(
                publish.update_releases(self.digest, apply=True)["changed_releases"], 1
            )
            self.assertEqual(fetch.call_count, 2)
            release_id, body = write.call_args.args
            self.assertEqual(release_id, 123)
            self.assertTrue(body.startswith(newer["body"]))
            self.assertEqual(body.count(START), 1)
            self.assertEqual(body.count(END), 1)
        with patch.object(
            publish, "current_snapshot", return_value=self.snapshot
        ), patch.object(
            publish, "public_pages", return_value=[dict(self.release, body=body)]
        ), patch.object(
            publish, "patch_release"
        ) as write:
            self.assertEqual(
                publish.update_releases(self.digest, apply=True)["changed_releases"], 0
            )
            write.assert_not_called()

    def test_drafts_immutable_releases_and_dry_run_are_never_written(self):
        with patch.object(
            publish, "current_snapshot", return_value=self.snapshot
        ), patch.object(
            publish,
            "public_pages",
            return_value=[
                dict(self.release, draft=True, body=CANARY),
                dict(self.release, immutable=True),
                self.release,
            ],
        ), patch.object(
            publish, "public_json", return_value=(self.release, {})
        ), patch.object(
            publish, "patch_release"
        ) as write:
            self.assertEqual(
                publish.update_releases(self.digest),
                {"changed_releases": 1, "immutable_releases_skipped": 1},
            )
            write.assert_not_called()

    def test_changed_projection_and_invalid_markers_prevent_all_writes(self):
        with patch.object(
            publish,
            "current_snapshot",
            side_effect=[self.snapshot, api.PublicDataError(CANARY)],
        ), patch.object(
            publish, "public_pages", return_value=[self.release]
        ), patch.object(
            publish, "patch_release"
        ) as write, self.assertRaises(
            api.PublicDataError
        ):
            publish.update_releases(self.digest, apply=True)
        write.assert_not_called()
        with patch.object(
            publish, "current_snapshot", return_value=self.snapshot
        ), patch.object(
            publish,
            "public_pages",
            return_value=[self.release, dict(self.release, id=124, body=START)],
        ), patch.object(
            publish, "patch_release"
        ) as write, self.assertRaises(
            api.PublicDataError
        ):
            publish.update_releases(self.digest, apply=True)
        write.assert_not_called()

    def test_release_becoming_private_stops_before_patch(self):
        with patch.object(
            publish, "current_snapshot", return_value=self.snapshot
        ), patch.object(
            publish, "public_pages", return_value=[self.release]
        ), patch.object(
            publish,
            "public_json",
            return_value=(dict(self.release, draft=True, body=CANARY), {}),
        ), patch.object(
            publish, "patch_release"
        ) as write, self.assertRaises(
            api.PublicDataError
        ):
            publish.update_releases(self.digest, apply=True)
        write.assert_not_called()

    def test_authenticated_transport_only_patches_body_and_suppresses_responses(self):
        response = subprocess.CompletedProcess([], 0, CANARY, CANARY)
        with patch.object(publish.subprocess, "run", return_value=response) as run:
            publish.patch_release(123, "Safe public notes")
        self.assertEqual(
            json.loads(run.call_args.kwargs["input"]), {"body": "Safe public notes"}
        )
        self.assertIn(
            "repos/Significant-Gravitas/AutoGPT/releases/123", run.call_args.args[0]
        )
        self.assertNotIn("security-advisories", " ".join(run.call_args.args[0]))
        self.assertTrue(run.call_args.kwargs["capture_output"])
        self.assertIn("--silent", run.call_args.args[0])
        with patch.object(
            publish.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 1, CANARY, CANARY),
        ), self.assertRaises(api.PublicDataError) as raised:
            publish.patch_release(123, "Safe public notes")
        self.assertNotIn(CANARY, str(raised.exception))

    def test_cli_refuses_writes_outside_trusted_workflow(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}), patch(
            "sys.argv", ["publish", "--expected-digest", self.digest, "--apply"]
        ), patch.object(
            publish, "update_releases"
        ) as update, contextlib.redirect_stderr(
            io.StringIO()
        ) as stderr, self.assertRaises(
            SystemExit
        ):
            publish.main()
        update.assert_not_called()
        self.assertNotIn(CANARY, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
