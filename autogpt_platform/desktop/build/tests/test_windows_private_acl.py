"""No native ACL calls: fake descriptors and exclusively owned scratch trees."""

import importlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

BUILD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUILD))


class FakeNative:
    user = b"private-user-sid"
    system = b"private-system-sid"
    administrators = b"private-admin-sid"
    owner_rights = b"private-owner-sid"
    default_owner = administrators

    def __init__(self, acl, baseline, control):
        self.acl = acl
        self.baseline = baseline
        self.control = control
        self.created = []
        self.override = None

    def token_summary(self):
        return {
            "observer": "parent_process",
            "elevated": True,
            "elevation_type": "full",
            "default_owner": "administrators",
            "administrators": "enabled",
            "child_token_lineage": "not_observed",
        }

    def create_directory(self, path):
        path.mkdir()
        self.created.append(path)

    def descriptor(self, path):
        control = path.is_relative_to(self.control)
        root = self.control if control else self.baseline
        relative = path.relative_to(root).as_posix()
        protected = relative in self.acl.PRIVATE_DIRS
        principals = (
            (self.user, self.system)
            if control
            else (self.system, self.administrators, self.owner_rights)
        )
        flags = (3 if path.is_dir() else 0) | (0 if protected else 16)
        record = {
            "owner": self.administrators,
            "protected": protected,
            "aces": tuple((0, flags, self.acl.FULL_ACCESS, sid) for sid in principals),
            "label": (),
        }
        if self.override:
            self.override(path, record)
        return record


class ACLTests(unittest.TestCase):
    def setUp(self):
        self.acl = importlib.import_module("windows_private_acl")
        self.temp = tempfile.TemporaryDirectory(
            dir=os.environ.get("PG_PROBE_TEST_ROOT")
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.baseline = self.root / "baseline"
        self.control = self.root / "baselinb"
        self.baseline.mkdir()
        for name in self.acl.PRIVATE_DIRS[1:]:
            (self.baseline / name).mkdir()
        for name in self.acl.DATA_DIRS:
            (self.baseline / "private/data" / name).mkdir(parents=True, exist_ok=True)
        (self.baseline / "private/runtime/postgres/bin").mkdir(parents=True)
        (self.baseline / "private/runtime/postgres/bin/initdb.exe").write_bytes(b"MZ")
        (self.baseline / "private/pair.secret").write_bytes(b"synthetic")
        self.payload = {"bin": ("directory", None), "bin/initdb.exe": ("file", "hash")}
        self.native = FakeNative(self.acl, self.baseline, self.control)
        guard = patch.object(
            self.acl.ctypes,
            "WinDLL",
            side_effect=AssertionError("native denied"),
            create=True,
        )
        guard.start()
        self.addCleanup(guard.stop)

    def pair(self):
        return self.acl.PairACL(
            self.baseline, self.control, float("inf"), native=self.native
        )

    def complete_fixture(self, pair):
        pair.inspect_baseline(self.payload)
        pair.create_private_tree()
        for name in self.acl.DATA_DIRS:
            (self.control / "private/data" / name).mkdir(parents=True, exist_ok=True)
        (self.control / "private/runtime/postgres/bin").mkdir(parents=True)
        (self.control / "private/runtime/postgres/bin/initdb.exe").write_bytes(b"MZ")
        (self.control / "private/pair.secret").write_bytes(b"synthetic")

    def test_unknown_baseline_owner_blocks_creation(self):
        self.native.override = lambda path, record: record.update(
            owner=b"unknown-private-sid"
        )
        pair = self.pair()
        with self.assertRaises(ValueError):
            pair.inspect_baseline(self.payload)
        with self.assertRaises(ValueError):
            pair.create_private_tree()
        self.assertEqual(self.native.created, [])
        self.assertFalse(self.control.exists())

    def test_new_control_has_matching_owner_integrity_and_only_exact_grants(self):
        pair = self.pair()
        self.complete_fixture(pair)
        report = pair.compare(self.payload)
        self.assertTrue(report["owner_match"] and report["integrity_match"])
        self.assertEqual(report["control"]["protected"], len(self.acl.PRIVATE_DIRS))
        self.assertEqual(
            report["control"]["objects"], report["control"]["exact_user_grants"]
        )
        self.assertEqual(report["baseline"]["exact_user_grants"], 0)
        self.assertNotIn("private-", json.dumps(report))
        self.assertNotIn(str(self.root), json.dumps(report))

    def test_owner_and_integrity_mismatches_are_inconclusive(self):
        pair = self.pair()
        self.complete_fixture(pair)
        for field, value in (
            ("owner", self.native.user),
            ("label", ((17, 0, 1, b"different-label"),)),
        ):
            with self.subTest(field=field):

                def override(path, record, field=field, value=value):
                    if path.is_relative_to(self.control):
                        record[field] = value

                self.native.override = override
                with self.assertRaises(ValueError):
                    pair.compare(self.payload)

    def test_unprotected_acl_missing_grant_or_extra_principal_is_rejected(self):
        pair = self.pair()
        self.complete_fixture(pair)
        for kind in ("unprotected", "missing", "extra", "not_inherited"):
            with self.subTest(kind=kind):

                def override(path, record, kind=kind):
                    if path == self.control / "private":
                        if kind == "unprotected":
                            record["protected"] = False
                        elif kind == "missing":
                            record["aces"] = record["aces"][1:]
                        elif kind == "extra":
                            record["aces"] += (
                                (
                                    0,
                                    3,
                                    self.acl.FULL_ACCESS,
                                    self.native.administrators,
                                ),
                            )
                    if (
                        kind == "not_inherited"
                        and path
                        == self.control / "private/runtime/postgres/bin/initdb.exe"
                    ):
                        record["aces"] = tuple(
                            (a, 0, c, d) for a, _, c, d in record["aces"]
                        )

                self.native.override = override
                with self.assertRaises(ValueError):
                    pair.compare(self.payload)

    def test_creation_cannot_touch_existing_or_outside_paths(self):
        pair = self.pair()
        pair.inspect_baseline(self.payload)
        for relative in (
            "../outside",
            "private/../../outside",
            str(self.root),
            "private/runtime",
        ):
            with self.assertRaises(ValueError):
                pair.create(relative)
        self.assertEqual(self.native.created, [])
        pair.create_private_tree()
        before = list(self.native.created)
        with self.assertRaises(ValueError):
            pair.create(".")
        self.assertEqual(self.native.created, before)

    def test_existing_sibling_is_rejected_without_native_access(self):
        self.control.mkdir()
        with self.assertRaises(ValueError):
            self.pair()
        self.assertEqual(self.native.created, [])

    def test_deadline_refuses_creation_and_keeps_owned_paths(self):
        pair = self.pair()
        pair.inspect_baseline(self.payload)
        pair.deadline = 1
        with patch.object(
            self.acl.time, "monotonic", return_value=1
        ), self.assertRaises(TimeoutError):
            pair.create_private_tree()
        self.assertEqual(self.native.created, [])

    def test_native_creation_sets_only_dacl_and_frees_descriptor_on_interruption(self):
        api = object.__new__(self.acl.NativeAccess)
        api.user_text = "S-1-5-21-100-200-300-400"
        api.p = lambda: SimpleNamespace(value=123)
        api.convert_sd = Mock(return_value=True)
        api.security_attributes = Mock(return_value="attributes")
        api.free = Mock(return_value=None)
        for failure in (None, OSError("private"), KeyboardInterrupt("private")):
            api.create = Mock(return_value=True, side_effect=failure)
            api.free.reset_mock()
            with patch.object(
                self.acl.ctypes, "byref", side_effect=lambda value: value, create=True
            ), patch.object(self.acl.ctypes, "sizeof", return_value=24, create=True):
                if failure is None:
                    api.create_directory(self.control)
                else:
                    with self.assertRaises(type(failure)) as raised:
                        api.create_directory(self.control)
                    self.assertIs(raised.exception, failure)
            self.assertEqual(
                api.convert_sd.call_args.args[0],
                "D:P(A;OICI;FA;;;S-1-5-21-100-200-300-400)(A;OICI;FA;;;SY)",
            )
            api.create.assert_called_once_with(str(self.control), "attributes")
            api.free.assert_called_once()

    def test_replaced_owned_directory_is_rejected_before_child_creation(self):
        pair = self.pair()
        pair.inspect_baseline(self.payload)
        pair.create(".")
        pair.created["."] += 1
        with self.assertRaises(ValueError):
            pair.create("private")
        self.assertEqual(self.native.created, [self.control])

    def test_control_preparation_uses_identical_extraction_secret_layout_and_environment(
        self,
    ):
        probe = importlib.import_module("postgres_windows_check")
        private = self.baseline / "private"
        (private / "runtime/postgres/bin/postgres.exe").write_bytes(b"MZ")
        secret = b"x" * 43
        (private / "pair.secret").write_bytes(secret)
        archive = private / "postgres.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            for name in ("payload/bin/initdb.exe", "payload/bin/postgres.exe"):
                entry = tarfile.TarInfo(name)
                entry.size = 2
                output.addfile(entry, io.BytesIO(b"MZ"))

        def environment(path, inherited):
            return {
                "PATH": "system-only",
                "HOME": str(path / "home"),
                "HOMEPATH": str(path / "home")[len(path.drive) :],
            }

        env = dict(
            environment(private, {}),
            GITHUB_ACTIONS="true",
            RUNNER_ENVIRONMENT="github-hosted",
        )
        with patch.object(probe.time, "monotonic", return_value=100), patch.object(
            probe, "PIN", probe.smoke.sha256(archive)
        ), patch.object(
            self.acl, "NativeAccess", return_value=self.native
        ), patch.object(
            probe.smoke, "private_environment", side_effect=environment
        ):
            payload = probe.payload_manifest(private / "runtime/postgres", 190)
            other, other_env, evidence = probe.prepare_control(
                private, archive, env, payload, 190
            )
            self.assertEqual(
                probe.payload_manifest(other / "runtime/postgres", 190), payload
            )
        self.assertEqual((other / "postgres.tar.gz").read_bytes(), archive.read_bytes())
        self.assertEqual((other / "pair.secret").read_bytes(), secret)
        self.assertFalse((other / "data/postgres").exists())
        self.assertEqual(other_env["PATH"], env["PATH"])
        self.assertEqual(len(self.native.created), len(self.acl.PRIVATE_DIRS))
        self.assertEqual(evidence["control"]["ace_shape"], "exact_user_system")
        self.assertNotIn(secret.decode(), json.dumps(evidence))
        self.assertEqual(probe.SOURCES["private_acl"], "build/windows_private_acl.py")


if __name__ == "__main__":
    unittest.main()
