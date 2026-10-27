"""Component contracts; all native operations are mocked or audit-denied."""

import importlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

BUILD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(BUILD.parent / "runtime"))


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.probe = importlib.import_module("postgres_windows_check")
        self.temp = tempfile.TemporaryDirectory(
            dir=os.environ.get("PG_PROBE_TEST_ROOT")
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for target in (
            "subprocess.Popen",
            "subprocess.run",
            "socket.socket",
            "socket.create_connection",
            "urllib.request.urlopen",
            "ctypes.WinDLL",
        ):
            guard = patch(
                target,
                side_effect=AssertionError("native operation denied"),
                create=True,
            )
            guard.start()
            self.addCleanup(guard.stop)

    def test_host_rejected_before_work_or_download(self):
        work = self.root / "new"
        with patch.object(self.probe.smoke, "require_clean_host", side_effect=OSError):
            self.assertEqual(self.probe.main(["--work-dir", str(work)]), 1)
        self.assertFalse(work.exists())

    def test_existing_work_is_never_changed(self):
        sentinel = self.root / "sentinel"
        sentinel.write_bytes(b"preserve")
        with patch.object(self.probe.smoke, "require_clean_host"):
            self.assertEqual(self.probe.main(["--work-dir", str(self.root)]), 1)
        self.assertEqual(sentinel.read_bytes(), b"preserve")
        self.assertEqual(list(self.root.iterdir()), [sentinel])

    def test_strict_versions_do_not_accept_added_text(self):
        for tool in ("postgres", "initdb"):
            expected = f"{tool} (PostgreSQL) 18.6"
            self.assertEqual(
                self.probe.version_literal(tool, expected.encode() + b"\r\n", b""),
                expected,
            )
            for raw, err in (
                (expected.encode() + b"\nsecret", b""),
                (expected.encode(), b"secret"),
                (b"18.6", b""),
            ):
                with self.assertRaises(ValueError):
                    self.probe.version_literal(tool, raw, err)

    def test_classification_is_bounded_and_never_exports_text(self):
        secret = b"C:/Users/private-name/secret-token"
        raw = b"initdb: error: could not re-execute with restricted token: error code 5\r\n"
        result = self.probe.classify_output(b"", raw + secret + b"\n")
        self.assertEqual(result["classification"], "restricted_reexec")
        self.assertEqual(result["windows_error"], 5)
        self.assertNotIn("private-name", json.dumps(result))
        for bad in (
            raw + b"\x00",
            raw + b"\xff",
            raw + b"x" * 5000,
            b"x" * (self.probe.MAX_OUTPUT + 1),
        ):
            result = self.probe.classify_output(b"", bad)
            self.assertEqual(result["classification"], "unknown")
            self.assertTrue(result["output_rejected"])

    def test_unknown_exit_is_not_loader_classification(self):
        result = self.probe.native_result(-1073741515, b"", b"unmatched private text\n")
        self.assertEqual(result["child_exit_code"], -1073741515)
        self.assertEqual(result["windows_status"], 3221225781)
        self.assertEqual(result["classification"], "unknown")
        self.assertEqual(result["status"], "failed")

    def test_bad_archive_pin_never_extracts(self):
        response = io.BytesIO(b"wrong archive")
        response.status = 200
        response.geturl = lambda: "https://release-assets.githubusercontent.com/test"
        with patch.object(
            self.probe.urllib.request,
            "build_opener",
            return_value=SimpleNamespace(open=Mock(return_value=response)),
        ), self.assertRaises(ValueError):
            self.probe.fetch_archive(self.root)
        self.assertFalse((self.root / "runtime").exists())

    def archive(self, names):
        archive = self.root / f"fixture-{len(list(self.root.iterdir()))}.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for name, kind in names:
                item = tarfile.TarInfo(name)
                item.type = kind
                item.size = 2 if kind == tarfile.REGTYPE else 0
                item.linkname = "outside"
                stream.addfile(item, io.BytesIO(b"MZ") if item.size else None)
        return archive

    def test_extraction_rejects_unsafe_names_links_and_collisions(self):
        for names in (
            [("../escape", tarfile.REGTYPE)],
            [("payload/file:stream", tarfile.REGTYPE)],
            [("payload/CON.txt", tarfile.REGTYPE)],
            [("payload/link", tarfile.SYMTYPE)],
            [("payload/a", tarfile.REGTYPE), ("payload/A", tarfile.REGTYPE)],
        ):
            with self.subTest(names=names):
                archive = self.archive(names)
                with self.assertRaises(ValueError):
                    self.probe.extract_archive(archive, self.root / "runtime")
                self.assertFalse((self.root / "runtime").exists())

    def test_archive_extracts_only_checked_files_and_strips_single_root(self):
        archive = self.archive(
            [
                ("payload", tarfile.DIRTYPE),
                ("payload/bin", tarfile.DIRTYPE),
                ("payload/bin/postgres.exe", tarfile.REGTYPE),
                ("payload/bin/initdb.exe", tarfile.REGTYPE),
            ]
        )
        destination = self.root / "runtime"
        self.probe.extract_archive(archive, destination)
        self.assertEqual((destination / "bin/initdb.exe").read_bytes(), b"MZ")
        self.assertFalse((destination / "payload").exists())
        with self.assertRaises(ValueError):
            self.probe.extract_archive(archive, destination)

    def test_expansion_limit_fails_before_destination_creation(self):
        archive = self.archive([("payload/file", tarfile.REGTYPE)])
        with patch.object(self.probe, "MAX_EXPANDED", 1), self.assertRaises(ValueError):
            self.probe.extract_archive(archive, self.root / "runtime")
        self.assertFalse((self.root / "runtime").exists())

    def test_artifact_change_fails_before_any_request(self):
        with patch.object(self.probe, "PIN", "0" * 64), patch.object(
            self.probe.urllib.request, "build_opener"
        ) as opener, self.assertRaises(ValueError):
            self.probe.fetch_archive(self.root)
        opener.assert_not_called()

    def test_redirect_cannot_change_scheme_or_leave_release_hosts(self):
        handler = self.probe.ReleaseRedirect()
        for url in (
            "http://github.com/file",
            "https://evil.invalid/file",
            "https://user:password@github.com/file",
        ):
            with self.assertRaises(ValueError):
                handler.redirect_request(None, None, 302, None, None, url)

    def test_cleanup_never_blesses_forced_cleanup(self):
        clean = self.clean_record()
        self.assertEqual(self.probe.cleanup_summary(clean)["status"], "passed")
        for key, value in (
            ("leftovers_before_cleanup", 1),
            ("surviving_owned_pids", [1]),
            ("remaining_listeners", [{}]),
            ("observed_ports", [123]),
            ("inspection_failed", True),
            ("deadline_exceeded", True),
        ):
            with self.subTest(key=key):
                result = self.probe.cleanup_summary(dict(clean, **{key: value}))
                self.assertEqual(result["status"], "failed")

    @staticmethod
    def clean_record():
        return {
            "leftovers_before_cleanup": 0,
            "live_owned_during_cleanup": [],
            "surviving_owned_pids": [],
            "remaining_listeners": [],
            "ports_not_checked": [],
            "observed_ports": [],
            "inspection_failed": False,
            "deadline_exceeded": False,
        }

    def test_actual_initialize_exception_survives_diagnostic_failure(self):
        from autogpt_desktop import postgres
        from autogpt_desktop.layout import Bundle, DataDir

        data = DataDir(self.root / "data")
        data.prepare()
        failure = subprocess.CalledProcessError(
            7, ["private argv"], b"private stdout", b"private stderr"
        )
        with patch.object(postgres, "run_tool", side_effect=failure), patch.object(
            self.probe, "write_capture", side_effect=OSError
        ), self.assertRaises(RuntimeError) as raised:
            self.probe.initialize_once(postgres, Bundle(self.root), data, self.root, {})
        self.assertIs(raised.exception.__cause__, failure)
        self.assertTrue((data.config / "first-run.incomplete").exists())
        self.assertFalse(list(data.run.glob("postgres-password.*")))

    def test_existing_database_cannot_take_runtime_shortcut(self):
        from autogpt_desktop import postgres
        from autogpt_desktop.layout import Bundle, DataDir

        data = DataDir(self.root / "data")
        data.prepare()
        data.postgres.mkdir()
        (data.postgres / "PG_VERSION").write_text("18\n")
        with patch.object(postgres, "initialize") as initialize, self.assertRaises(
            ValueError
        ):
            self.probe.initialize_once(postgres, Bundle(self.root), data, self.root, {})
        initialize.assert_not_called()

    def test_verification_requires_real_config_and_control(self):
        from autogpt_desktop.layout import DataDir

        data = DataDir(self.root / "data")
        data.prepare()
        data.postgres.mkdir()
        (data.postgres / "PG_VERSION").write_text("18\n")
        with self.assertRaises((OSError, ValueError)):
            self.probe.verify_cluster(data)

    def test_diagnostic_parser_error_preserves_original_failure(self):
        from autogpt_desktop import postgres
        from autogpt_desktop.layout import Bundle, DataDir

        data = DataDir(self.root / "data")
        data.prepare()
        failure = subprocess.CalledProcessError(9, ["private argv"], b"", b"secret")
        observation = {}
        with patch.object(postgres, "run_tool", side_effect=failure), patch.object(
            self.probe, "classify_output", side_effect=ValueError
        ), self.assertRaises(RuntimeError) as raised:
            self.probe.initialize_once(
                postgres, Bundle(self.root), data, self.root, observation
            )
        self.assertIs(raised.exception.__cause__, failure)
        self.assertTrue(observation["capture_failed"])
        self.assertEqual(observation["child_exit_code"], 9)
        self.assertEqual(observation["status"], "failed")

    def test_initialization_timeout_stays_timeout(self):
        from autogpt_desktop import postgres
        from autogpt_desktop.layout import Bundle, DataDir

        data = DataDir(self.root / "data")
        data.prepare()
        (self.root / "logs").mkdir()
        failure = subprocess.TimeoutExpired(["private argv"], 60, b"private")
        observation = {}
        with patch.object(postgres, "run_tool", side_effect=failure), self.assertRaises(
            subprocess.TimeoutExpired
        ) as raised:
            self.probe.initialize_once(
                postgres, Bundle(self.root), data, self.root, observation
            )
        self.assertIs(raised.exception, failure)
        self.assertEqual(observation["status"], "timeout")
        self.assertNotIn("private", json.dumps(observation))

    def initialized_fixture(self, data):
        data.postgres.mkdir()
        (data.postgres / "base/1").mkdir(parents=True)
        (data.postgres / "global").mkdir()
        (data.postgres / "PG_VERSION").write_text("18\n")
        (data.postgres / "base/1/PG_VERSION").write_text("18\n")
        (data.postgres / "postgresql.conf").write_text("# fixture\n")
        (data.postgres / "pg_hba.conf").write_text(
            "host all all 127.0.0.1/32 scram-sha-256\n"
        )
        (data.postgres / "global/pg_control").write_bytes(b"fixture control")

    def test_real_initializer_argv_environment_and_cluster_verification(self):
        from autogpt_desktop import postgres
        from autogpt_desktop.layout import Bundle, DataDir

        data = DataDir(self.root / "data")
        data.prepare()
        (self.root / "logs").mkdir()
        observation = {}

        def fake_tool(argv, **kwargs):
            self.assertEqual(kwargs["timeout"], 60)
            self.assertTrue(kwargs["check"] and kwargs["capture_output"])
            self.assertNotIn("PG_RESTRICT_EXEC", kwargs["env"])
            self.assertNotIn("LC_ALL", kwargs["env"])
            self.assertEqual(kwargs["env"]["LANG"], "C")
            self.assertEqual(argv[1], f"--pgdata={data.postgres}")
            self.assertEqual(argv[2], "--username=postgres")
            self.assertEqual(
                argv[4:], ["--auth=scram-sha-256", "--encoding=UTF8", "--locale=C"]
            )
            self.assertTrue(Path(argv[3].split("=", 1)[1]).read_text().endswith("\n"))
            self.initialized_fixture(data)
            return subprocess.CompletedProcess(argv, 0, b"success\n", b"")

        with patch.dict(
            os.environ, {"PG_RESTRICT_EXEC": "1", "LC_ALL": "bogus"}
        ), patch.object(postgres, "run_tool", side_effect=fake_tool):
            self.probe.initialize_once(
                postgres, Bundle(self.root), data, self.root, observation
            )
        self.probe.verify_cluster(data)
        self.assertEqual(observation["status"], "passed")
        self.assertTrue((data.config / "first-run.incomplete").exists())
        self.assertFalse(list(data.run.glob("postgres-password.*")))

    def test_reexec_error_with_zero_exit_is_still_failure(self):
        from autogpt_desktop import postgres
        from autogpt_desktop.layout import Bundle, DataDir

        data = DataDir(self.root / "data")
        data.prepare()
        (self.root / "logs").mkdir()
        observation = {}

        def fake_tool(argv, **kwargs):
            self.initialized_fixture(data)
            return subprocess.CompletedProcess(
                argv,
                0,
                b"",
                b"initdb: error: could not re-execute with restricted token: error code 5\n",
            )

        with patch.object(
            postgres, "run_tool", side_effect=fake_tool
        ), self.assertRaises(ValueError):
            self.probe.initialize_once(
                postgres, Bundle(self.root), data, self.root, observation
            )
        self.assertEqual(observation["status"], "failed")

    def supervisor_fixture(self):
        (self.root / "logs").mkdir()
        (self.root / "runtime").mkdir()
        process = Mock(pid=23, returncode=1)
        process.poll.return_value = 1
        process.wait.return_value = 1
        owned = Mock()
        owned.cleanup.return_value = self.clean_record()
        report = self.probe.worker_template()
        report["stage"] = "initialize"
        report["initialization"] = self.probe.native_result(
            7, b"", b"private stdout and SID"
        )
        report["worker_cleanup"] = {"active_children": 0, "inspection_failed": False}
        return process, owned, report

    def test_supervisor_retains_failed_exit_and_closes_handle(self):
        process, owned, report = self.supervisor_fixture()
        with patch.object(
            self.probe.subprocess, "Popen", return_value=process
        ), patch.object(self.probe.smoke, "json_object", return_value=report):
            result = self.probe.supervise(self.root, {}, owned)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["initialization"]["child_exit_code"], 7)
        self.assertEqual(result["cleanup"]["status"], "passed")
        process.wait.assert_called_once()
        process._handle.Close.assert_called_once()
        self.assertNotIn("private stdout", json.dumps(result))

    def diagnostic_worker_fixture(self, passed):
        process, owned, report = self.supervisor_fixture()
        process.returncode = 0 if passed else 1
        process.poll.return_value = process.returncode
        process.wait.return_value = process.returncode
        report.update(status="passed" if passed else "failed")
        report["stage"] = "complete" if passed else "initialize"
        report["initialization"] = self.probe.native_result(
            0 if passed else 7,
            b"",
            (
                b""
                if passed
                else (
                    b"initdb: error: could not re-execute with restricted token: "
                    b"error code 5\n"
                )
            ),
        )
        for name in ("postgres", "initdb"):
            report["versions"][name] = dict(
                self.probe.native_result(0, b"", b""),
                literal=f"{name} (PostgreSQL) 18.6",
            )
        self.probe.safe_worker_result(report)
        (self.root / "worker.json").write_text(json.dumps(report), encoding="utf-8")
        return process, owned, report

    def check_parent_error(self, *, observer, passed):
        process, owned, report = self.diagnostic_worker_fixture(passed)
        failure = OSError("private supervision sentinel")
        if observer:
            process.poll.side_effect = [None, process.returncode]
            owned.observe.side_effect = failure
        else:
            owned.psutil.Process.side_effect = failure
        with patch.object(
            self.probe.subprocess, "Popen", return_value=process
        ) as launch:
            result = self.probe.supervise(self.root, {}, owned)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["worker_exit_code"], process.returncode)
        self.assertEqual(result["initialization"], report["initialization"])
        self.assertEqual(result["versions"], report["versions"])
        self.assertFalse(result["worker_report_rejected"])
        self.assertEqual(result["cleanup"]["status"], "failed")
        self.assertTrue(result["cleanup"]["inspection_failed"])
        self.assertFalse(result["cleanup"]["forced_cleanup"])
        self.assertNotIn("private supervision sentinel", json.dumps(result))
        process.wait.assert_called_once()
        process._handle.Close.assert_called_once()
        owned.cleanup.assert_called_once()
        self.assertTrue(launch.call_args.kwargs["stdout"].closed)

    def test_observation_error_preserves_failed_native_evidence(self):
        self.check_parent_error(observer=True, passed=False)

    def test_registration_error_preserves_failed_native_evidence(self):
        self.check_parent_error(observer=False, passed=False)

    def test_observation_error_cannot_be_overwritten_by_success(self):
        self.check_parent_error(observer=True, passed=True)

    def test_registration_error_cannot_be_overwritten_by_success(self):
        self.check_parent_error(observer=False, passed=True)

    def check_rejected_report_after_parent_error(self, raw):
        process, owned, _ = self.supervisor_fixture()
        owned.psutil.Process.side_effect = OSError("private lookup sentinel")
        if raw is not None:
            (self.root / "worker.json").write_bytes(raw)
        with patch.object(
            self.probe.subprocess, "Popen", return_value=process
        ) as launch:
            result = self.probe.supervise(self.root, {}, owned)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["worker_exit_code"], 1)
        self.assertEqual(result["initialization"]["status"], "unknown")
        self.assertIsNone(result["initialization"]["child_exit_code"])
        self.assertEqual(result["initialization"]["classification"], "unknown")
        self.assertTrue(result["worker_report_rejected"])
        self.assertTrue(result["cleanup"]["inspection_failed"])
        self.assertNotIn("private", json.dumps(result))
        process.wait.assert_called_once()
        process._handle.Close.assert_called_once()
        owned.cleanup.assert_called_once()
        self.assertTrue(launch.call_args.kwargs["stdout"].closed)

    def test_missing_report_after_parent_error_fails_unknown(self):
        self.check_rejected_report_after_parent_error(None)

    def test_malformed_report_after_parent_error_fails_unknown(self):
        self.check_rejected_report_after_parent_error(b'{"private": "sentinel"}')

    def check_reap_error(self, passed):
        process, owned, report = self.diagnostic_worker_fixture(passed)
        process.wait.side_effect = OSError("private reap sentinel")
        with patch.object(
            self.probe.subprocess, "Popen", return_value=process
        ) as launch:
            result = self.probe.supervise(self.root, {}, owned)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["worker_exit_code"], process.returncode)
        self.assertEqual(result["initialization"], report["initialization"])
        self.assertFalse(result["worker_report_rejected"])
        self.assertEqual(result["cleanup"]["status"], "failed")
        self.assertTrue(result["cleanup"]["inspection_failed"])
        self.assertNotIn("private reap sentinel", json.dumps(result))
        process.wait.assert_called_once()
        process._handle.Close.assert_called_once()
        owned.cleanup.assert_called_once()
        self.assertTrue(launch.call_args.kwargs["stdout"].closed)

    def test_reap_error_preserves_failed_native_evidence(self):
        self.check_reap_error(False)

    def test_reap_error_cannot_be_overwritten_by_success(self):
        self.check_reap_error(True)

    def check_interruption(self, location, error_type, *, cleanup_error=False):
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            private = Path(temporary)
            (private / "logs").mkdir()
            (private / "runtime").mkdir()
            process = Mock(pid=23, returncode=1)
            process.poll.return_value = 1
            process.wait.return_value = 1
            owned = Mock()
            owned.cleanup.return_value = self.clean_record()
            if cleanup_error:
                owned.cleanup.side_effect = OSError("private cleanup sentinel")
            interruption = error_type("private interruption sentinel")
            with patch.object(self.probe.time, "sleep") as sleep, patch.object(
                self.probe.smoke, "json_object"
            ) as reader, patch.object(
                self.probe.subprocess, "Popen", return_value=process
            ) as launch:
                targets = {
                    "lookup": owned.psutil.Process,
                    "remember": owned.remember,
                    "observe": owned.observe,
                    "sleep": sleep,
                    "wait": process.wait,
                    "close": process._handle.Close,
                    "report": reader,
                }
                if location in {"observe", "sleep"}:
                    process.poll.side_effect = [None, 1]
                targets[location].side_effect = interruption
                with self.assertRaises(error_type) as raised:
                    self.probe.supervise(private, {}, owned)
            self.assertIs(raised.exception, interruption)
            self.assertEqual(process.wait.call_count, 1)
            self.assertEqual(process._handle.Close.call_count, 1)
            self.assertTrue(launch.call_args.kwargs["stdout"].closed)
            self.assertEqual(owned.cleanup.call_count, 1)
            self.assertGreaterEqual(owned.cleanup.call_args.kwargs["timeout"], 0)
            self.assertLessEqual(owned.cleanup.call_args.kwargs["timeout"], 10)
            self.assertEqual(reader.call_count, int(location == "report"))

    def test_supervision_interruptions_always_clean_owned_tree(self):
        for error_type in (KeyboardInterrupt, SystemExit):
            for location in ("lookup", "remember", "observe", "sleep"):
                with self.subTest(error_type=error_type, location=location):
                    self.check_interruption(location, error_type)

    def test_finalization_interruptions_always_clean_owned_tree(self):
        for error_type in (KeyboardInterrupt, SystemExit):
            for location in ("wait", "close"):
                with self.subTest(error_type=error_type, location=location):
                    self.check_interruption(location, error_type)

    def test_report_interruptions_preserve_identity_and_cleanup(self):
        for error_type in (KeyboardInterrupt, SystemExit):
            with self.subTest(error_type=error_type):
                self.check_interruption("report", error_type)

    def test_cleanup_error_cannot_replace_supervision_interruption(self):
        for error_type in (KeyboardInterrupt, SystemExit):
            with self.subTest(error_type=error_type):
                self.check_interruption("observe", error_type, cleanup_error=True)

    def test_launch_interruption_preserves_identity_and_cleanup(self):
        process, owned, _ = self.supervisor_fixture()
        interruption = KeyboardInterrupt("private launch sentinel")
        with patch.object(
            self.probe.subprocess, "Popen", side_effect=interruption
        ) as launch, self.assertRaises(KeyboardInterrupt) as raised:
            self.probe.supervise(self.root, {}, owned)
        self.assertIs(raised.exception, interruption)
        self.assertTrue(launch.call_args.kwargs["stdout"].closed)
        self.assertEqual(owned.cleanup.call_count, 1)
        process.wait.assert_not_called()
        process._handle.Close.assert_not_called()

    def test_supervisor_timeout_kills_reaps_and_inspects(self):
        process, owned, _ = self.supervisor_fixture()
        process.poll.return_value = None
        with patch.object(
            self.probe.subprocess, "Popen", return_value=process
        ), patch.object(self.probe.time, "monotonic", side_effect=[0, 75, 75, 75, 75]):
            result = self.probe.supervise(self.root, {}, owned)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["timed_out"])
        self.assertTrue(result["cleanup"]["forced_cleanup"])
        process.kill.assert_called_once()
        process.wait.assert_called_once()
        process._handle.Close.assert_called_once()
        owned.cleanup.assert_called_once()

    def test_report_io_failure_cannot_prevent_cleanup(self):
        process, owned, _ = self.supervisor_fixture()
        with patch.object(
            self.probe.subprocess, "Popen", return_value=process
        ), patch.object(self.probe.smoke, "json_object", side_effect=OSError):
            result = self.probe.supervise(self.root, {}, owned)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["initialization"]["status"], "unknown")
        self.assertTrue(result["worker_report_rejected"])
        owned.cleanup.assert_called_once()
        process._handle.Close.assert_called_once()

    def test_outer_init_deadline_kills_worker_before_global_timeout(self):
        process, owned, _ = self.supervisor_fixture()
        process.poll.return_value = None
        (self.root / "initialize.started").write_text("0")
        with patch.object(
            self.probe.subprocess, "Popen", return_value=process
        ), patch.object(
            self.probe.time, "monotonic", side_effect=[0, 60, 60, 60, 60, 60]
        ):
            result = self.probe.supervise(self.root, {}, owned)
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["stage"], "initialize")
        self.assertEqual(result["initialization"]["status"], "timeout")
        process.kill.assert_called_once()
        process.wait.assert_called_once()

    def test_each_version_has_an_independent_five_second_outer_deadline(self):
        (self.root / "postgres.started").write_text("0")
        self.assertIsNone(self.probe.expired_native(self.root, 4.9))
        self.assertEqual(self.probe.expired_native(self.root, 5), "postgres")
        (self.root / "postgres.done").touch()
        (self.root / "initdb.started").write_text("5")
        self.assertEqual(self.probe.expired_native(self.root, 10), "initdb")

    def test_cleanup_error_does_not_replace_native_exit(self):
        process, owned, report = self.supervisor_fixture()
        owned.cleanup.side_effect = OSError("private secret")
        with patch.object(
            self.probe.subprocess, "Popen", return_value=process
        ), patch.object(self.probe.smoke, "json_object", return_value=report):
            result = self.probe.supervise(self.root, {}, owned)
        self.assertEqual(result["initialization"]["child_exit_code"], 7)
        self.assertEqual(result["cleanup"]["status"], "failed")
        self.assertNotIn("private secret", json.dumps(result))

    def test_worker_report_rejects_extra_fields_and_version_only_success(self):
        for report in (
            dict(self.probe.worker_template(), argv="secret"),
            dict(self.probe.worker_template(), status="passed", stage="complete"),
        ):
            with self.assertRaises(ValueError):
                self.probe.safe_worker_result(report)

    def test_honest_version_failure_does_not_claim_initialization(self):
        report = self.probe.worker_template()
        report["versions"]["postgres"].update(
            self.probe.native_result(1, b"", b"private")
        )
        self.assertEqual(
            self.probe.safe_worker_result(report)["initialization"]["status"], "not_run"
        )

    def run_mocked_worker(self, active_children):
        from autogpt_desktop import postgres, process
        from autogpt_desktop.layout import DataDir

        private = self.root / "private"
        (private / "runtime/postgres").mkdir(parents=True)
        (private / "logs").mkdir()
        data = DataDir(private / "data")

        def fake_tool(argv, **kwargs):
            if argv[-1] == "--version":
                name = Path(argv[0]).stem
                self.assertEqual(kwargs["timeout"], 5)
                return subprocess.CompletedProcess(
                    argv, 0, f"{name} (PostgreSQL) 18.6\n".encode(), b""
                )
            self.initialized_fixture(data)
            return subprocess.CompletedProcess(argv, 0, b"success\n", b"")

        with patch.object(self.probe.smoke, "require_clean_host"), patch.object(
            self.probe.smoke, "private_environment", return_value={"FIXTURE": "1"}
        ), patch.dict(os.environ), patch.object(
            self.probe.os, "chdir"
        ) as chdir, patch.object(
            process, "adopt_kill_on_exit_job"
        ), patch.object(
            process, "_job_handle", 1
        ), patch.object(
            self.probe.ctypes,
            "WinDLL",
            return_value=SimpleNamespace(SetErrorMode=Mock()),
            create=True,
        ), patch.object(
            self.probe, "job_children", return_value=active_children
        ), patch.object(
            postgres, "run_tool", side_effect=fake_tool
        ):
            code = self.probe.worker(private)
        chdir.assert_called_once_with(private / "runtime")
        report = self.probe.safe_worker_result(
            json.loads((private / "worker.json").read_text())
        )
        return code, report

    def test_worker_runs_versions_and_actual_initializer_before_success(self):
        code, report = self.run_mocked_worker(0)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["initialization"]["child_exit_code"], 0)
        self.assertEqual(report["stage"], "complete")

    def test_job_children_prevent_worker_success(self):
        code, report = self.run_mocked_worker(1)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["initialization"]["status"], "passed")
        self.assertEqual(report["worker_cleanup"]["active_children"], 1)

    def test_cleanup_can_fail_an_otherwise_successful_probe(self):
        process, owned, report = self.supervisor_fixture()
        process.returncode = 0
        process.poll.return_value = 0
        report.update(status="passed", stage="complete")
        report["initialization"] = self.probe.native_result(0, b"", b"")
        for name in ("postgres", "initdb"):
            report["versions"][name] = dict(
                self.probe.native_result(0, b"", b""),
                literal=f"{name} (PostgreSQL) 18.6",
            )
        owned.cleanup.return_value["leftovers_before_cleanup"] = 1
        with patch.object(
            self.probe.subprocess, "Popen", return_value=process
        ) as launch, patch.object(self.probe.smoke, "json_object", return_value=report):
            result = self.probe.supervise(self.root, {}, owned)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stage"], "cleanup")
        self.assertTrue(result["cleanup"]["forced_cleanup"])
        self.assertTrue(launch.call_args.kwargs["stdout"].closed)


if __name__ == "__main__":
    unittest.main()
