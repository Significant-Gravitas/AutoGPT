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
from security_release_notes import (
    BASE_URL,
    END,
    LOGIN_PATTERN,
    START,
    TEAM_LOGINS,
    fetch_advisories,
)
from test_security_release_notes import TAG, advisory

CANARY = "PRIVATE-CANARY-DO-NOT-PUBLISH"
EMPTY = {"schema_version": 1, "advisories": []}
POLICY = f"# Policy\n\n{sync.CREDIT_START}\nold\n{sync.CREDIT_END}\n\nHistorical credit stays.\n"


class SafeDiagnosticsTests(unittest.TestCase):
    def test_both_clis_log_only_allowlisted_reasons_not_exception_text(self):
        errors = [api.PublicDataError(code) for code in api.PublicErrorCode]
        errors += [api.PublicDataError(CANARY), ValueError(CANARY)]
        for module, argv, dependency in (
            (sync, ["sync", "digest"], "fetch_advisories"),
            (publish, ["publish", "--expected-digest", "0" * 64], "update_releases"),
        ):
            for error in errors:
                error.args = (CANARY,)
                error.__cause__ = RuntimeError(CANARY)
                output = io.StringIO()
                with self.subTest(
                    module=module.__name__, code=getattr(error, "code", None)
                ), patch("sys.argv", argv), patch.object(
                    module, dependency, side_effect=error
                ), contextlib.redirect_stdout(
                    output
                ), contextlib.redirect_stderr(
                    output
                ), self.assertRaises(
                    SystemExit
                ) as raised:
                    module.main()
                self.assertEqual(raised.exception.code, 1)
                self.assertNotIn(CANARY, output.getvalue())
                self.assertNotIn("Traceback", output.getvalue())
                if isinstance(error, api.PublicDataError):
                    self.assertIn(error.code.value, output.getvalue())

    def test_ambiguous_mapping_reports_a_reason_without_report_content(self):
        item = advisory()
        item["summary"] = CANARY
        item["description"] += "\n" + CANARY
        item["vulnerabilities"] = [
            {"patched_versions": "0.9.0"},
            {"patched_versions": "0.8.0"},
        ]
        output = io.StringIO()
        with patch("sys.argv", ["sync", "digest"]), patch.object(
            sync, "fetch_advisories", return_value=[item]
        ), contextlib.redirect_stdout(output), contextlib.redirect_stderr(
            output
        ), self.assertRaises(
            SystemExit
        ):
            sync.main()
        self.assertIn(api.PublicErrorCode.RELEASE_MAPPING.value, output.getvalue())
        self.assertNotIn(CANARY, output.getvalue())
        self.assertNotIn(item["ghsa_id"], output.getvalue())


class AnonymousApiTests(unittest.TestCase):
    def test_collection_members_must_be_objects_before_results_are_returned(self):
        for endpoint in ("security-advisories", "releases"):
            for member in (None, CANARY, 1, True, []):
                with self.subTest(endpoint=endpoint, member=member):
                    with patch.object(
                        api, "public_json", return_value=([{"id": 1}, member], {})
                    ), self.assertRaises(api.PublicDataError) as raised:
                        api.public_pages(endpoint)
                    self.assertIs(
                        raised.exception.code, api.PublicErrorCode.API_COLLECTION
                    )
                    self.assertNotIn(CANARY, str(raised.exception))

    def test_digest_cli_rejects_invalid_collection_without_printing_a_digest(self):
        captured = io.StringIO()
        with patch("sys.argv", ["sync", "digest"]), patch.object(
            api, "public_json", return_value=([advisory(), CANARY], {})
        ), contextlib.redirect_stdout(captured), contextlib.redirect_stderr(
            captured
        ), self.assertRaises(
            SystemExit
        ) as raised:
            sync.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn(api.PublicErrorCode.API_COLLECTION.value, captured.getvalue())
        self.assertNotRegex(captured.getvalue(), r"[0-9a-f]{64}")
        self.assertNotIn(CANARY, captured.getvalue())
        self.assertNotIn("Traceback", captured.getvalue())

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

    def test_pagination_stops_at_the_page_limit(self):
        endpoint = "security-advisories"
        numbers = iter(range(1, api.MAX_PAGES + 2))

        def read(path):
            number = next(numbers)
            link = f"{api.API_ROOT}{endpoint}?per_page=100&state=published&page={number + 1}"
            return [{"id": number}], {"Link": f'<{link}>; rel="next"'}

        with patch.object(api, "public_json", side_effect=read) as reader:
            with self.assertRaises(api.PublicDataError) as raised:
                api.public_pages(endpoint, "state=published")
        self.assertIs(raised.exception.code, api.PublicErrorCode.PAGE_LIMIT)
        self.assertEqual(reader.call_count, api.MAX_PAGES)


class PublicProjectionTests(unittest.TestCase):
    def test_committed_snapshot_and_generated_credits_obey_the_public_schema(self):
        root = Path(__file__).resolve().parents[3]
        snapshot = sync.validate_snapshot(
            json.loads((root / sync.SNAPSHOT_PATH).read_text(encoding="utf-8"))
        )
        security = (root / "SECURITY.md").read_text(encoding="utf-8")
        self.assertEqual(sync.acknowledgments(snapshot, security), security)

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

    def test_rows_that_agree_on_the_fixed_version_need_no_mapping(self):
        item = advisory()
        item["vulnerabilities"] = [
            {
                "package": {"ecosystem": "other", "name": name},
                "vulnerable_version_range": "< 0.9.0",
                "patched_versions": fixed,
            }
            for name, fixed in (
                ("autogpt-platform-beta", "0.9.0"),
                ("significantgravitas/autogpt", "v0.9.0"),
                ("autogpt-platform-backend", None),
            )
        ]
        self.assertEqual(
            sync.public_snapshot([item])["advisories"][0]["release_tags"], [TAG]
        )
        item["vulnerabilities"][2]["patched_versions"] = "0.8.0"
        with self.assertRaises(api.PublicDataError) as raised:
            sync.public_snapshot([item])
        self.assertIs(raised.exception.code, api.PublicErrorCode.RELEASE_MAPPING)

    def test_team_credits_are_not_outside_acknowledgments(self):
        item = advisory()
        item["credits_detailed"] += [
            {"user": {"login": "kcze"}, "state": "accepted", "type": "analyst"},
            {"user": {"login": "PWUTS"}, "state": "accepted", "type": "finder"},
        ]
        snapshot = sync.public_snapshot([item])
        self.assertEqual(snapshot["advisories"][0]["researchers"], ["researcher"])
        for text in (
            sync.acknowledgments(snapshot, POLICY),
            sync.public_release_notes(snapshot, TAG, ""),
        ):
            self.assertIn("@researcher", text)
            self.assertNotIn("kcze", text.casefold())
            self.assertNotIn("pwuts", text.casefold())

    def test_chosen_names_appear_beside_handles(self):
        item = advisory()
        item["credits_detailed"].append(
            {"user": {"login": "MegaManSec"}, "state": "accepted", "type": "reporter"}
        )
        snapshot = sync.public_snapshot([item])
        named = "Joshua Rogers ([@MegaManSec](https://github.com/MegaManSec))"
        credits = sync.acknowledgments(snapshot, POLICY)
        self.assertIn(f"- {named}\n", credits)
        self.assertIn("- [@researcher](https://github.com/researcher)\n", credits)
        self.assertIn(named, sync.public_release_notes(snapshot, TAG, ""))
        for login, name in sync.RESEARCHER_NAMES.items():
            with self.subTest(login=login):
                self.assertRegex(login, f"^{LOGIN_PATTERN}$")
                self.assertRegex(name, r"^[\w .'-]+$")
                self.assertNotIn(login.casefold(), TEAM_LOGINS)

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
        item = advisory()
        digest = sync.snapshot_digest(sync.public_snapshot([item]))
        original = {
            Path("SECURITY.md"): POLICY,
            sync.SNAPSHOT_PATH: sync.snapshot_text(EMPTY),
        }
        for relative in (Path("."), Path(".github"), *original):
            with self.subTest(path=str(relative)), TemporaryDirectory() as directory:
                root = Path(directory) / "checkout"
                outside = Path(directory) / "outside"
                (outside / ".github").mkdir(parents=True)
                for path, content in original.items():
                    (outside / path).write_text(content, encoding="utf-8")
                link = root / relative
                link.parent.mkdir(parents=True, exist_ok=True)
                try:
                    link.symlink_to(
                        outside / relative,
                        target_is_directory=relative in (Path("."), Path(".github")),
                    )
                except OSError:
                    self.skipTest("Symlink creation is not available on this host")
                (root / ".github").mkdir(exist_ok=True)
                for path, content in original.items():
                    if not (root / path).exists():
                        (root / path).write_text(content, encoding="utf-8")
                with patch.object(
                    sync, "fetch_advisories", return_value=[item]
                ) as fetch, self.assertRaises(api.PublicDataError) as raised:
                    sync.prepare_credits(root, digest)
                self.assertIs(raised.exception.code, api.PublicErrorCode.FILE_SYMLINK)
                fetch.assert_not_called()
                for location in (root, outside):
                    for path, content in original.items():
                        self.assertEqual(
                            (location / path).read_text(encoding="utf-8"), content
                        )


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
        self.write_environment = {
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": api.REPOSITORY,
            "GITHUB_REF": "refs/heads/master",
            "GH_TOKEN": CANARY,
        }
        sleep = patch.object(publish.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)
        self.latest = dict(
            self.release, id=999, tag_name="autogpt-platform-beta-v1.0.0"
        )

    def reads(self, *releases, latest=None):
        """Answer public_json by path: each release by id, and the latest release."""
        by_path = {f"releases/{release['id']}": release for release in releases}
        by_path["releases/latest"] = latest or self.latest
        return lambda path: (copy.deepcopy(by_path[path]), {})

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

    def test_drafts_and_dry_run_are_never_written(self):
        with patch.object(
            publish, "current_snapshot", return_value=self.snapshot
        ), patch.object(
            publish,
            "public_pages",
            return_value=[
                dict(self.release, draft=True, body=CANARY),
                self.release,
            ],
        ), patch.object(
            publish, "public_json", return_value=(self.release, {})
        ), patch.object(
            publish, "patch_release"
        ) as write:
            self.assertEqual(
                publish.update_releases(self.digest),
                {"changed_releases": 1},
            )
            write.assert_not_called()

    def test_immutable_release_updates_only_notes_even_if_immutability_changed(self):
        for was_immutable in (False, True):
            with self.subTest(was_immutable=was_immutable):
                latest = dict(
                    self.release,
                    immutable=True,
                    assets=[{"name": CANARY}],
                    body="Maintainer's latest notes\n",
                )
                with patch.object(
                    publish, "current_snapshot", return_value=self.snapshot
                ), patch.object(
                    publish,
                    "public_pages",
                    return_value=[dict(self.release, immutable=was_immutable)],
                ), patch.object(
                    publish, "public_json", side_effect=self.reads(latest)
                ), patch.object(
                    publish.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0, CANARY, CANARY),
                ) as run:
                    self.assertEqual(
                        publish.update_releases(self.digest, apply=True),
                        {"changed_releases": 1},
                    )
                run.assert_called_once()
                payload = json.loads(run.call_args.kwargs["input"])
                self.assertEqual(payload["make_latest"], "false")
                self.assertEqual(set(payload), {"body", "make_latest"})
                self.assertTrue(payload["body"].startswith(latest["body"]))
                self.assertIn("GHSA-aaaa-bbbb-cccc", payload["body"])
                self.assertNotIn(CANARY, run.call_args.kwargs["input"])

    def test_cli_preview_shows_only_generated_public_sections_and_never_writes(self):
        item = advisory()
        item["summary"] = CANARY
        item["description"] += CANARY
        item["cve_id"] = "CVE-2026-12345"
        item["credits_detailed"].append(
            {
                "user": {"login": "pending-person"},
                "state": "pending",
                "type": "reporter",
            }
        )
        private = dict(item, state="draft", ghsa_id="GHSA-dead-beef-cafe")
        digest = sync.snapshot_digest(sync.public_snapshot([item]))
        release = dict(
            self.release,
            immutable=True,
            name=CANARY,
            body=f"{CANARY}\n{START}\n{CANARY}\n{END}\n{CANARY}\n",
        )
        with patch.dict(os.environ, self.write_environment, clear=True), patch(
            "sys.argv", ["publish", "--expected-digest", digest, "--preview"]
        ), patch.object(
            sync, "fetch_advisories", return_value=[private, item]
        ), patch.object(
            publish, "public_pages", return_value=[dict(release, draft=True), release]
        ), patch.object(
            publish, "public_json", return_value=(release, {})
        ), patch.object(
            publish, "patch_release"
        ) as write, contextlib.redirect_stdout(
            io.StringIO()
        ) as output:
            publish.main()
        preview = output.getvalue()
        self.assertIn("Releases to update: 1", preview)
        self.assertIn(f"{BASE_URL}/releases/tag/{TAG}", preview)
        self.assertIn(
            "```markdown\n"
            f"{START}\n## Security fixes\n\n"
            f"- [GHSA-aaaa-bbbb-cccc]({BASE_URL}/security/advisories/GHSA-aaaa-bbbb-cccc)"
            " (CVE-2026-12345). Thanks to [@researcher](https://github.com/researcher).\n"
            f"\n{END}\n```",
            preview,
        )
        for forbidden in (CANARY, "pending-person", private["ghsa_id"]):
            self.assertNotIn(forbidden, preview)
        write.assert_not_called()

    def test_preview_includes_withdrawal_replacement_without_old_report_text(self):
        release = dict(self.release, body=f"{START}\n{CANARY}\n{END}\n")
        with patch.object(
            publish, "current_snapshot", return_value=EMPTY
        ), patch.object(publish, "public_pages", return_value=[release]), patch.object(
            publish, "public_json", return_value=(release, {})
        ), patch.object(
            publish, "patch_release"
        ) as write:
            result = publish.update_releases(sync.snapshot_digest(EMPTY), preview=True)
        self.assertEqual(result["changed_releases"], 1)
        self.assertIn(
            "No currently published advisories are linked to this release.",
            result["preview"],
        )
        self.assertNotIn(CANARY, result["preview"])
        write.assert_not_called()

    def test_cli_preview_failure_emits_no_partial_sections(self):
        with patch(
            "sys.argv", ["publish", "--expected-digest", self.digest, "--preview"]
        ), patch.object(
            publish, "current_snapshot", return_value=self.snapshot
        ), patch.object(
            publish,
            "public_pages",
            return_value=[self.release, dict(self.release, id=124)],
        ), patch.object(
            publish,
            "public_json",
            side_effect=[(self.release, {}), ValueError(CANARY)],
        ), patch.object(
            publish, "patch_release"
        ) as write, contextlib.redirect_stdout(
            io.StringIO()
        ) as stdout, contextlib.redirect_stderr(
            io.StringIO()
        ) as stderr, self.assertRaises(
            SystemExit
        ):
            publish.main()
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn(CANARY, stderr.getvalue())
        write.assert_not_called()

    def test_cli_rejects_preview_combined_with_apply_before_fetching(self):
        with patch(
            "sys.argv",
            ["publish", "--expected-digest", self.digest, "--preview", "--apply"],
        ), patch.object(
            publish, "update_releases"
        ) as update, contextlib.redirect_stderr(
            io.StringIO()
        ), self.assertRaises(
            SystemExit
        ) as raised:
            publish.main()
        self.assertEqual(raised.exception.code, 2)
        update.assert_not_called()

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
            side_effect=self.reads(dict(self.release, draft=True, body=CANARY)),
        ), patch.object(
            publish, "patch_release"
        ) as write, self.assertRaises(
            api.PublicDataError
        ) as raised:
            publish.update_releases(self.digest, apply=True)
        self.assertIs(raised.exception.code, api.PublicErrorCode.RELEASE_CHANGED)
        write.assert_not_called()

    def test_release_body_must_be_text_or_null(self):
        for body in (None, "", "Notes"):
            with self.subTest(body=body):
                self.assertTrue(
                    publish.published_release(dict(self.release, body=body))
                )
        for body in (0, False, [], {}, ["Notes"]):
            with self.subTest(body=body):
                self.assertFalse(
                    publish.published_release(dict(self.release, body=body))
                )

    def test_writes_keep_the_current_latest_release(self):
        newer_tag = "autogpt-platform-beta-v1.0.0"
        current = dict(
            self.latest, tag_name=newer_tag, body=f"Notes\n{START}\nstale\n{END}\n"
        )
        releases = [current, self.release]
        with patch.object(
            publish, "current_snapshot", return_value=self.snapshot
        ), patch.object(publish, "public_pages", return_value=releases), patch.object(
            publish, "public_json", side_effect=self.reads(*releases, latest=current)
        ), patch.object(
            publish, "patch_release"
        ) as write:
            self.assertEqual(
                publish.update_releases(self.digest, apply=True),
                {"changed_releases": 2},
            )
        self.assertEqual(
            [(c.args[0], c.kwargs["latest"]) for c in write.call_args_list],
            [(current["id"], True), (self.release["id"], False)],
        )

    def test_latest_release_must_be_known_and_unchanged(self):
        unreadable = dict(self.latest, draft=True)
        with patch.object(
            publish, "current_snapshot", return_value=self.snapshot
        ), patch.object(
            publish, "public_pages", return_value=[self.release]
        ), patch.object(
            publish,
            "public_json",
            side_effect=self.reads(self.release, latest=unreadable),
        ), patch.object(
            publish, "patch_release"
        ) as write, self.assertRaises(
            api.PublicDataError
        ) as raised:
            publish.update_releases(self.digest, apply=True)
        self.assertIs(raised.exception.code, api.PublicErrorCode.LATEST_RELEASE)
        write.assert_not_called()

        moved = iter([self.latest, self.release])
        before_and_after = self.reads(self.release)

        def read(path):
            if path == "releases/latest":
                return copy.deepcopy(next(moved)), {}
            return before_and_after(path)

        with patch.object(
            publish, "current_snapshot", return_value=self.snapshot
        ), patch.object(
            publish, "public_pages", return_value=[self.release]
        ), patch.object(
            publish, "public_json", side_effect=read
        ), patch.object(
            publish, "patch_release"
        ) as write, self.assertRaises(
            api.PublicDataError
        ) as raised:
            publish.update_releases(self.digest, apply=True)
        self.assertIs(raised.exception.code, api.PublicErrorCode.LATEST_CHANGED)
        write.assert_called_once()

        with patch.object(
            publish, "current_snapshot", return_value=self.snapshot
        ), patch.object(
            publish, "public_pages", return_value=[self.release]
        ), patch.object(
            publish, "public_json", side_effect=self.reads(self.release)
        ) as reader:
            publish.update_releases(self.digest, preview=True)
        self.assertNotIn("releases/latest", [c.args[0] for c in reader.call_args_list])

    def test_authenticated_transport_only_patches_body_and_suppresses_responses(self):
        response = subprocess.CompletedProcess([], 0, CANARY, CANARY)
        for latest, flag in ((False, "false"), (True, "true")):
            with patch.object(publish.subprocess, "run", return_value=response) as run:
                publish.patch_release(123, "Safe public notes", latest=latest)
            self.assertEqual(
                json.loads(run.call_args.kwargs["input"]),
                {"body": "Safe public notes", "make_latest": flag},
            )
        for invalid in ("false", None, 1):
            with self.subTest(latest=invalid), patch.object(
                publish.subprocess, "run"
            ) as never, self.assertRaises(api.PublicDataError):
                publish.patch_release(123, "Safe public notes", latest=invalid)
            never.assert_not_called()
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
            publish.patch_release(123, "Safe public notes", latest=False)
        self.assertNotIn(CANARY, str(raised.exception))

    def test_cli_refuses_writes_outside_trusted_workflow(self):
        for key, value in (
            ("GITHUB_ACTIONS", "false"),
            ("GITHUB_REPOSITORY", "different/repository"),
            ("GITHUB_REF", "refs/heads/dev"),
            ("GH_TOKEN", None),
            ("GH_TOKEN", ""),
        ):
            environment = self.write_environment.copy()
            if value is None:
                del environment[key]
            else:
                environment[key] = value
            with self.subTest(key=key, value=value):
                output = io.StringIO()
                with patch.dict(os.environ, environment, clear=True), patch(
                    "sys.argv", ["publish", "--expected-digest", self.digest, "--apply"]
                ), patch.object(
                    publish, "update_releases"
                ) as update, contextlib.redirect_stdout(
                    output
                ), contextlib.redirect_stderr(
                    output
                ), self.assertRaises(
                    SystemExit
                ) as raised:
                    publish.main()
                update.assert_not_called()
                self.assertEqual(raised.exception.code, 1)
                self.assertIn(
                    api.PublicErrorCode.WRITE_CONTEXT.value, output.getvalue()
                )
                self.assertNotIn(CANARY, output.getvalue())

    def test_cli_allows_writes_with_complete_trusted_context(self):
        result = {"changed_releases": 0}
        with patch.dict(os.environ, self.write_environment, clear=True), patch(
            "sys.argv", ["publish", "--expected-digest", self.digest, "--apply"]
        ), patch.object(
            publish, "update_releases", return_value=result
        ) as update, contextlib.redirect_stdout(
            io.StringIO()
        ) as output:
            publish.main()
        update.assert_called_once_with(self.digest, apply=True, preview=False)
        self.assertEqual(json.loads(output.getvalue()), result)
        self.assertNotIn(CANARY, output.getvalue())


if __name__ == "__main__":
    unittest.main()
