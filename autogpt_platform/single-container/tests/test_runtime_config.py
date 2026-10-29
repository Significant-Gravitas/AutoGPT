from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import io
import os
import re
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).parents[1] / "runtime_config.py"
COMMON_PATH = Path(__file__).parents[1] / "common.sh"
SPEC = importlib.util.spec_from_file_location(
    "single_container_runtime_config", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
runtime_config = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_config)

LISTENER_PATH = Path(__file__).parents[1] / "fatal_listener.py"
LISTENER_SPEC = importlib.util.spec_from_file_location(
    "single_container_fatal_listener", LISTENER_PATH
)
assert LISTENER_SPEC is not None and LISTENER_SPEC.loader is not None
fatal_listener = importlib.util.module_from_spec(LISTENER_SPEC)
LISTENER_SPEC.loader.exec_module(fatal_listener)


class RuntimeConfigTest(unittest.TestCase):
    def test_shell_loader_allowlist_matches_generated_runtime_keys(self) -> None:
        common = COMMON_PATH.read_text(encoding="utf-8")
        match = re.search(
            r'case "\$\{name\}" in\s+(?P<keys>[A-Z0-9_ |]+)\)\s+'
            r'export "\$\{name\}=\$\{value\}"',
            common,
        )
        self.assertIsNotNone(match)
        assert match is not None
        shell_keys = {name.strip() for name in match.group("keys").split("|")}
        with tempfile.TemporaryDirectory() as directory:
            generated_keys = set(
                runtime_config.ensure_runtime_config(
                    Path(directory) / "runtime.env", {}
                )
            )

        self.assertEqual(shell_keys, generated_keys)

    def test_first_boot_generates_complete_private_config_and_reuses_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"

            first = runtime_config.ensure_runtime_config(path, {})
            second = runtime_config.ensure_runtime_config(path, {})

            self.assertEqual(first, second)
            self.assertEqual(
                set(first),
                {
                    "AUTOGPT_RUNTIME_CONFIG_VERSION",
                    "POSTGRES_PASSWORD",
                    "RABBITMQ_DEFAULT_USER",
                    "RABBITMQ_DEFAULT_PASS",
                    "REDIS_PASSWORD",
                    "BETTER_AUTH_SECRET",
                    "ENCRYPTION_KEY",
                    "UNSUBSCRIBE_SECRET_KEY",
                    "GRAPHITI_FALKORDB_PASSWORD",
                    "INTERNAL_API_KEY",
                    "VAPID_PRIVATE_KEY",
                    "VAPID_PUBLIC_KEY",
                },
            )
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_first_boot_honors_explicit_values(self) -> None:
        vapid_private = base64.urlsafe_b64encode(b"p" * 32).rstrip(b"=").decode()
        vapid_public = (
            base64.urlsafe_b64encode(b"\x04" + b"q" * 64).rstrip(b"=").decode()
        )
        environment = {
            "POSTGRES_PASSWORD": "p" * 40,
            "RABBITMQ_DEFAULT_USER": "self_hosted",
            "RABBITMQ_DEFAULT_PASS": "r" * 40,
            "REDIS_PASSWORD": "d" * 40,
            "BETTER_AUTH_SECRET": "b" * 40,
            "ENCRYPTION_KEY": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
            "UNSUBSCRIBE_SECRET_KEY": "u" * 40,
            "GRAPHITI_FALKORDB_PASSWORD": "f" * 40,
            "INTERNAL_API_KEY": "i" * 40,
            "VAPID_PRIVATE_KEY": vapid_private,
            "VAPID_PUBLIC_KEY": vapid_public,
        }
        with tempfile.TemporaryDirectory() as directory:
            values = runtime_config.ensure_runtime_config(
                Path(directory) / "runtime.env", environment
            )

        for name, value in environment.items():
            self.assertEqual(values[name], value)

    def test_first_boot_fsyncs_file_and_parent_directory(self) -> None:
        real_fsync = runtime_config.os.fsync
        events: list[tuple[str, bool] | tuple[str]] = []

        def record_fsync(descriptor: int) -> None:
            events.append(
                ("fsync", stat.S_ISDIR(runtime_config.os.fstat(descriptor).st_mode))
            )
            real_fsync(descriptor)

        def record_replace(source: Path, destination: Path) -> None:
            events.append(("replace",))
            runtime_config.os.rename(source, destination)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            with (
                mock.patch.object(runtime_config.os, "fsync", side_effect=record_fsync),
                mock.patch.object(
                    runtime_config.os, "replace", side_effect=record_replace
                ),
            ):
                runtime_config.ensure_runtime_config(path, {})

        self.assertIn(("fsync", False), events)
        self.assertIn(("fsync", True), events)
        self.assertLess(events.index(("fsync", False)), events.index(("replace",)))
        self.assertLess(events.index(("replace",)), events.index(("fsync", True)))

    def test_first_boot_closes_descriptor_when_fdopen_fails(self) -> None:
        real_close = runtime_config.os.close
        closed_descriptors: list[int] = []

        def record_close(descriptor: int) -> None:
            closed_descriptors.append(descriptor)
            real_close(descriptor)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            with (
                mock.patch.object(
                    runtime_config.os, "fdopen", side_effect=MemoryError("test")
                ),
                mock.patch.object(runtime_config.os, "close", side_effect=record_close),
                self.assertRaisesRegex(MemoryError, "test"),
            ):
                runtime_config.ensure_runtime_config(path, {})

            self.assertEqual(len(closed_descriptors), 1)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_existing_config_rejects_secret_rotation_by_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            values = runtime_config.ensure_runtime_config(path, {})
            environment = {"POSTGRES_PASSWORD": values["POSTGRES_PASSWORD"] + "x"}

            with self.assertRaisesRegex(ValueError, "persisted on first boot"):
                runtime_config.ensure_runtime_config(path, environment)

    def test_rejects_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            target.write_text("not a config", encoding="ascii")
            link = Path(directory) / "runtime.env"
            link.symlink_to(target)

            with self.assertRaisesRegex(ValueError, "refusing symlink"):
                runtime_config.ensure_runtime_config(link, {})

    def test_rejects_unsafe_explicit_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "POSTGRES_PASSWORD"):
                runtime_config.ensure_runtime_config(
                    Path(directory) / "runtime.env",
                    {"POSTGRES_PASSWORD": "contains whitespace and shell syntax $(id)"},
                )

    def test_rejects_builtin_rabbitmq_guest_user(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "built-in guest"):
                runtime_config.ensure_runtime_config(
                    Path(directory) / "runtime.env",
                    {"RABBITMQ_DEFAULT_USER": "guest"},
                )

    def test_rejects_one_sided_vapid_configuration(self) -> None:
        cases = [
            {"VAPID_PRIVATE_KEY": "p" * 43},
            {"VAPID_PUBLIC_KEY": "q" * 87},
        ]
        for environment in cases:
            with (
                self.subTest(environment=environment),
                tempfile.TemporaryDirectory() as directory,
                self.assertRaisesRegex(ValueError, "must be set together"),
            ):
                runtime_config.ensure_runtime_config(
                    Path(directory) / "runtime.env", environment
                )

    def test_rejects_corrupt_existing_config(self) -> None:
        invalid_vapid_public = (
            base64.urlsafe_b64encode(b"\x03" + b"q" * 64).rstrip(b"=").decode()
        )
        cases = [
            (
                "duplicate key",
                lambda content: content + "POSTGRES_PASSWORD=duplicate\n",
                "invalid runtime configuration line",
            ),
            (
                "malformed line",
                lambda content: content + "malformed\n",
                "invalid runtime configuration line",
            ),
            (
                "unsupported version",
                lambda content: content.replace(
                    "AUTOGPT_RUNTIME_CONFIG_VERSION=2",
                    "AUTOGPT_RUNTIME_CONFIG_VERSION=3",
                ),
                "unsupported runtime configuration version",
            ),
            (
                "invalid encryption key",
                lambda content: _replace_config_value(
                    content, "ENCRYPTION_KEY", "A" * 32
                ),
                "ENCRYPTION_KEY",
            ),
            (
                "invalid VAPID public key",
                lambda content: _replace_config_value(
                    content, "VAPID_PUBLIC_KEY", invalid_vapid_public
                ),
                "VAPID_PUBLIC_KEY",
            ),
        ]
        for name, mutate, error in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "runtime.env"
                runtime_config.ensure_runtime_config(path, {})
                path.write_text(
                    mutate(path.read_text(encoding="ascii")), encoding="ascii"
                )
                with self.assertRaisesRegex(ValueError, error):
                    runtime_config.ensure_runtime_config(path, {})

    def test_rejects_non_regular_existing_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            path.mkdir()
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                runtime_config.ensure_runtime_config(path, {})

    def test_version_1_config_gains_an_internal_key_and_keeps_the_rest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            original = _write_version_1_config(path)
            path.chmod(0o644)

            upgraded = runtime_config.ensure_runtime_config(path, {})

            key = upgraded.pop("INTERNAL_API_KEY")
            self.assertGreaterEqual(len(key), 32)
            self.assertRegex(key, r"^[A-Za-z0-9_-]+$")
            self.assertEqual(upgraded["AUTOGPT_RUNTIME_CONFIG_VERSION"], "2")
            self.assertEqual(
                {**upgraded, "AUTOGPT_RUNTIME_CONFIG_VERSION": "1"}, original
            )
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            again = runtime_config.ensure_runtime_config(path, {})
            self.assertEqual(again["INTERNAL_API_KEY"], key)
            self.assertEqual(again["AUTOGPT_RUNTIME_CONFIG_VERSION"], "2")

    def test_version_1_config_takes_a_configured_internal_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            _write_version_1_config(path)
            configured = {"INTERNAL_API_KEY": "k" * 40}

            upgraded = runtime_config.ensure_runtime_config(path, configured)

            self.assertEqual(upgraded["INTERNAL_API_KEY"], "k" * 40)
            self.assertEqual(
                runtime_config.ensure_runtime_config(path, configured),
                upgraded,
            )

    def test_rejects_version_1_config_that_does_not_match_version_1(self) -> None:
        cases = [
            ("extra key", lambda values: {**values, "SURPRISE": "s" * 40}),
            (
                "missing key",
                lambda values: {
                    name: value
                    for name, value in values.items()
                    if name != "REDIS_PASSWORD"
                },
            ),
            (
                "already has an internal key",
                lambda values: {**values, "INTERNAL_API_KEY": "i" * 40},
            ),
            (
                "invalid secret",
                lambda values: {**values, "REDIS_PASSWORD": "short"},
            ),
        ]
        for name, mutate in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "runtime.env"
                values = mutate(_version_1_values())
                _write_config_lines(path, values)
                before = path.read_bytes()

                with self.assertRaises(ValueError):
                    runtime_config.ensure_runtime_config(path, {})
                self.assertEqual(path.read_bytes(), before)

    def test_the_version_1_step_produces_version_2_after_later_bumps(self) -> None:
        with mock.patch.object(runtime_config, "CONFIG_VERSION", "3"):
            upgraded = runtime_config._upgraded_from_version_1(_version_1_values(), {})

        self.assertEqual(upgraded["AUTOGPT_RUNTIME_CONFIG_VERSION"], "2")

    def test_a_refused_boot_leaves_a_version_1_config_as_it_was(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            _write_version_1_config(path)
            before = path.read_bytes()

            with self.assertRaisesRegex(ValueError, "persisted on first boot"):
                runtime_config.ensure_runtime_config(
                    path, {"POSTGRES_PASSWORD": "x" * 40}
                )
            self.assertEqual(path.read_bytes(), before)

    def test_rejects_version_1_upgrade_with_an_unsafe_configured_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            _write_version_1_config(path)
            before = path.read_bytes()

            with self.assertRaisesRegex(ValueError, "INTERNAL_API_KEY"):
                runtime_config.ensure_runtime_config(
                    path, {"INTERNAL_API_KEY": "too-short"}
                )
            self.assertEqual(path.read_bytes(), before)


class PublicUrlTest(unittest.TestCase):
    def test_normalizes_origin(self) -> None:
        self.assertEqual(
            runtime_config.validate_public_url("https://Example.COM:8443/"),
            "https://example.com:8443",
        )

    def test_normalizes_idna_hostname(self) -> None:
        self.assertEqual(
            runtime_config.validate_public_url("https://BÜCHER.example/"),
            "https://xn--bcher-kva.example",
        )

    def test_accepts_ipv6_origin(self) -> None:
        self.assertEqual(
            runtime_config.validate_public_url("http://[::1]:3000"),
            "http://[::1]:3000",
        )

    def test_rejects_non_origin_values(self) -> None:
        invalid = [
            "ftp://example.com",
            "https://example.com/path",
            "https://example.com?token=secret",
            "https://user:password@example.com",
            "http:///missing-host",
            "http://bad_host.example",
            "http://bad$variable.example",
            "http://example..com",
            "http://example.com:99999",
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime_config.validate_public_url(value)


class FatalListenerTest(unittest.TestCase):
    def test_fatal_event_acknowledges_before_terminating_supervisor(self) -> None:
        payload = "processname:rest groupname:runtime from_state:BACKOFF"
        header = f"eventname:PROCESS_STATE_FATAL len:{len(payload)}\n"
        input_stream = io.StringIO(payload)
        output_stream = io.StringIO()
        calls: list[str] = []

        with mock.patch.object(fatal_listener.sys, "stderr", io.StringIO()):
            fatal_listener.handle_event(
                header,
                input_stream,
                output_stream,
                lambda: calls.append(output_stream.getvalue()),
            )

        self.assertEqual(output_stream.getvalue(), "RESULT 2\nOK")
        self.assertEqual(calls, ["RESULT 2\nOK"])

    def test_fatal_event_does_not_echo_untrusted_payload(self) -> None:
        payload = "processname:rest\nsecret-value groupname:runtime"
        header = f"eventname:PROCESS_STATE_FATAL len:{len(payload)}\n"
        with mock.patch.object(fatal_listener.sys, "stderr", io.StringIO()) as stderr:
            fatal_listener.handle_event(
                header,
                io.StringIO(payload),
                io.StringIO(),
                lambda: None,
            )

        self.assertNotIn("secret-value", stderr.getvalue())
        self.assertIn("unknown", stderr.getvalue())

    def test_unexpected_bootstrap_exit_terminates_supervisor(self) -> None:
        payload = "processname:bootstrap groupname:runtime expected:0"
        output_stream = io.StringIO()
        calls: list[str] = []

        with mock.patch.object(fatal_listener.sys, "stderr", io.StringIO()):
            fatal_listener.handle_event(
                f"eventname:PROCESS_STATE_EXITED len:{len(payload)}\n",
                io.StringIO(payload),
                output_stream,
                lambda: calls.append("terminated"),
            )

        self.assertEqual(output_stream.getvalue(), "RESULT 2\nOK")
        self.assertEqual(calls, ["terminated"])

    def test_expected_bootstrap_exit_is_ignored(self) -> None:
        self._assert_exit_ignored("processname:bootstrap expected:1")

    def test_other_process_exit_is_ignored(self) -> None:
        self._assert_exit_ignored("processname:rest expected:0")

    def test_supervisor_subscribes_to_exited_events(self) -> None:
        config = (
            Path(__file__).parents[1] / "supervisor" / "supervisord.conf"
        ).read_text(encoding="utf-8")
        self.assertIn("events=PROCESS_STATE_FATAL,PROCESS_STATE_EXITED", config)

    def test_rejects_malformed_events_without_terminating(self) -> None:
        cases = [
            ("eventname:PROCESS_STATE_FATAL\n", "", "invalid payload length"),
            (
                "eventname:PROCESS_STATE_FATAL len:not-a-number\n",
                "",
                "invalid payload length",
            ),
            (
                "eventname:PROCESS_STATE_FATAL "
                f"len:{fatal_listener.MAX_PAYLOAD_LENGTH + 1}\n",
                "",
                "payload is too large",
            ),
            (
                "eventname:PROCESS_STATE_FATAL len:5\n",
                "abc",
                "ended unexpectedly",
            ),
            ("eventname:TICK_5_SECONDS len:0\n", "", "unsupported type"),
        ]
        for header, payload, error in cases:
            terminate = mock.Mock()
            with (
                self.subTest(header=header),
                self.assertRaisesRegex(RuntimeError, error),
            ):
                fatal_listener.handle_event(
                    header,
                    io.StringIO(payload),
                    io.StringIO(),
                    terminate,
                )
            terminate.assert_not_called()

    def _assert_exit_ignored(self, payload: str) -> None:
        output_stream = io.StringIO()
        calls: list[str] = []

        fatal_listener.handle_event(
            f"eventname:PROCESS_STATE_EXITED len:{len(payload)}\n",
            io.StringIO(payload),
            output_stream,
            lambda: calls.append("terminated"),
        )

        self.assertEqual(output_stream.getvalue(), "RESULT 2\nOK")
        self.assertEqual(calls, [])


class FillEnvSecretsTest(unittest.TestCase):
    """`make init-env` bootstrap: .env.default ships secrets blank (SECRT-2611)."""

    def test_fills_only_blank_known_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# comment\n"
                "ENCRYPTION_KEY=\n"
                "UNSUBSCRIBE_SECRET_KEY=\n"
                "OPENAI_API_KEY=\n"
                "DB_PASS=already-set\n",
                encoding="utf-8",
            )

            filled = runtime_config.fill_env_blanks(path)

            self.assertEqual(filled, ["ENCRYPTION_KEY", "UNSUBSCRIBE_SECRET_KEY"])
            values = dict(
                line.split("=", 1)
                for line in path.read_text(encoding="utf-8").splitlines()
                if "=" in line
            )
            self.assertEqual(values["DB_PASS"], "already-set")
            # Not a secret this bootstrap owns: a blank must stay blank.
            self.assertEqual(values["OPENAI_API_KEY"], "")
            self.assertEqual(
                len(base64.urlsafe_b64decode(values["ENCRYPTION_KEY"])), 32
            )
            self.assertTrue(values["UNSUBSCRIBE_SECRET_KEY"])

    def test_rerunning_setup_does_not_rotate_existing_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("ENCRYPTION_KEY=\n", encoding="utf-8")

            runtime_config.fill_env_blanks(path)
            after_first = path.read_text(encoding="utf-8")
            self.assertEqual(runtime_config.fill_env_blanks(path), [])
            self.assertEqual(path.read_text(encoding="utf-8"), after_first)

    def test_generated_values_differ_between_installs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            keys = []
            for index in range(2):
                path = Path(directory) / f"{index}.env"
                path.write_text("ENCRYPTION_KEY=\n", encoding="utf-8")
                runtime_config.fill_env_blanks(path)
                keys.append(path.read_text(encoding="utf-8"))
            self.assertNotEqual(keys[0], keys[1])

    def test_generated_file_becomes_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("ENCRYPTION_KEY=\n", encoding="utf-8")
            path.chmod(0o644)

            runtime_config.fill_env_blanks(path)

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_published_digests_match_the_backend_startup_check(self) -> None:
        """The backend warns about the same published passwords this replaces."""
        guard = (
            Path(__file__).parents[2] / "backend/backend/util/secrets_guard.py"
        ).read_text(encoding="utf-8")
        assignment = next(
            node
            for node in ast.walk(ast.parse(guard))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_PUBLISHED_CREDENTIAL_DIGESTS"
                for target in node.targets
            )
        )

        digests = ast.literal_eval(assignment.value)
        self.assertEqual(digests, set(runtime_config.PUBLISHED_CREDENTIAL_DIGESTS))
        self.assertTrue(digests)
        self.assertTrue(
            all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in digests)
        )

    def test_check_env_defaults_rejects_populated_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative_path, names in runtime_config.BLANK_IN_ENV_DEFAULT.items():
                (root / relative_path).parent.mkdir(parents=True, exist_ok=True)
                (root / relative_path).write_text(
                    "".join(f"{name}=\n" for name in names), encoding="utf-8"
                )
            backend = root / "backend/.env.default"
            backend.write_text(
                "ENCRYPTION_KEY=not-blank\n" + backend.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            self.assertEqual(
                runtime_config.check_env_defaults(root),
                ["backend/.env.default:ENCRYPTION_KEY"],
            )

    def test_env_default_ships_its_secrets_blank_and_generatable(self) -> None:
        """`.env.default` is public, so these must ship blank (SECRT-2611) —
        and every blank one must be fillable, or setup leaves a broken stack."""
        platform_root = Path(__file__).parents[2]
        self.assertEqual(runtime_config.check_env_defaults(platform_root), [])

        with tempfile.TemporaryDirectory() as directory:
            # A fresh install: each .env copied from its .env.default.
            env_files = {
                relative_path: _copy_env_default(
                    platform_root, Path(directory), relative_path
                )
                for relative_path in runtime_config.BLANK_IN_ENV_DEFAULT
            }
            for relative_path, names in runtime_config.BLANK_IN_ENV_DEFAULT.items():
                # Blank is only safe if setup can actually fill it back in.
                self.assertEqual(
                    sorted(runtime_config.fill_env_blanks(env_files[relative_path])),
                    sorted(names),
                )


class FillEnvServicePasswordsTest(unittest.TestCase):
    """`make init-env` gives the database, broker and caches their passwords,
    and moves an existing install off the ones .env.default used to ship."""

    # Stands in for a value a .env.default once shipped: the real ones are
    # never written into a tracked file, this test's included.
    PUBLISHED = "pretend-this-shipped-in-a-public-env-default"

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        published = mock.patch.object(
            runtime_config,
            "PUBLISHED_CREDENTIAL_DIGESTS",
            frozenset({hashlib.sha256(self.PUBLISHED.encode()).hexdigest()}),
        )
        published.start()
        self.addCleanup(published.stop)

    def test_fresh_install_shares_one_database_password_across_files(self) -> None:
        platform_root = Path(__file__).parents[2]
        for relative_path in runtime_config.BLANK_IN_ENV_DEFAULT:
            _copy_env_default(platform_root, self.root, relative_path)

        _run_fill_env(
            self.root / ".env", self.root / "backend/.env", self.root / "frontend/.env"
        )

        root, backend, frontend = self._values()
        password = root["POSTGRES_PASSWORD"]
        self.assertGreaterEqual(len(password), 32)
        self.assertEqual(backend["DB_PASS"], password)
        self.assertEqual(
            frontend["DATABASE_URL"],
            f"postgresql://postgres:{password}@localhost:5432/postgres",
        )
        service_passwords = [
            backend[name]
            for name in (
                "RABBITMQ_DEFAULT_PASS",
                "REDIS_PASSWORD",
                "GRAPHITI_FALKORDB_PASSWORD",
            )
        ]
        self.assertTrue(all(len(value) >= 32 for value in service_passwords))
        self.assertEqual(len({password, *service_passwords}), 4)
        # backend/.env builds its own connection strings from DB_PASS.
        self.assertIn("${DB_PASS}", backend["DATABASE_URL"])

    def test_database_password_agrees_whatever_order_files_are_filled(self) -> None:
        self._write(
            root="POSTGRES_PASSWORD=\n",
            backend="DB_PASS=\n",
            frontend="DATABASE_URL=\n",
        )

        for path in ("frontend/.env", "backend/.env", ".env"):
            runtime_config.fill_env_blanks(self.root / path)

        root, backend, frontend = self._values()
        self.assertEqual(backend["DB_PASS"], root["POSTGRES_PASSWORD"])
        self.assertIn(f":{root['POSTGRES_PASSWORD']}@", frontend["DATABASE_URL"])

    def test_published_service_passwords_are_replaced_and_keys_are_not(self) -> None:
        backend_url = '"postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:5432/postgres"'
        self._write(
            root=f"POSTGRES_PASSWORD={self.PUBLISHED}\n",
            backend=(
                f"DB_PASS={self.PUBLISHED}\n"
                f"DATABASE_URL={backend_url}\n"
                f"RABBITMQ_DEFAULT_PASS={self.PUBLISHED}\n"
                f"REDIS_PASSWORD={self.PUBLISHED}\n"
                f"GRAPHITI_FALKORDB_PASSWORD={self.PUBLISHED}\n"
                # Encrypts stored data: replacing it would make that unreadable.
                f"ENCRYPTION_KEY={self.PUBLISHED}\n"
            ),
            frontend=(
                "DATABASE_URL=postgresql://postgres:"
                f"{self.PUBLISHED}@localhost:5432/postgres\n"
            ),
        )

        _run_fill_env(
            self.root / ".env", self.root / "backend/.env", self.root / "frontend/.env"
        )

        root, backend, frontend = self._values()
        password = root["POSTGRES_PASSWORD"]
        self.assertNotEqual(password, self.PUBLISHED)
        self.assertEqual(backend["DB_PASS"], password)
        self.assertEqual(
            frontend["DATABASE_URL"],
            f"postgresql://postgres:{password}@localhost:5432/postgres",
        )
        for name in (
            "RABBITMQ_DEFAULT_PASS",
            "REDIS_PASSWORD",
            "GRAPHITI_FALKORDB_PASSWORD",
        ):
            self.assertNotEqual(backend[name], self.PUBLISHED, name)
            self.assertGreaterEqual(len(backend[name]), 32, name)
        self.assertEqual(backend["ENCRYPTION_KEY"], self.PUBLISHED)
        self.assertEqual(backend["DATABASE_URL"], backend_url)

    def test_database_password_an_operator_set_is_kept_and_shared(self) -> None:
        self._write(
            root=f"POSTGRES_PASSWORD={self.PUBLISHED}\n",
            backend="DB_PASS=operator-chosen-password\n",
            frontend="DATABASE_URL=\n",
        )

        _run_fill_env(
            self.root / ".env", self.root / "backend/.env", self.root / "frontend/.env"
        )

        root, backend, frontend = self._values()
        self.assertEqual(root["POSTGRES_PASSWORD"], "operator-chosen-password")
        self.assertEqual(backend["DB_PASS"], "operator-chosen-password")
        self.assertIn(":operator-chosen-password@", frontend["DATABASE_URL"])

    def test_connection_string_with_a_published_password_takes_the_new_one(
        self,
    ) -> None:
        self._write(
            root="POSTGRES_PASSWORD=\n",
            backend=(
                f'DIRECT_URL="postgresql://postgres:{self.PUBLISHED}'
                '@localhost:5432/postgres?schema=platform"\n'
            ),
        )

        _run_fill_env(self.root / ".env", self.root / "backend/.env")

        root, backend, _ = self._values()
        self.assertEqual(
            backend["DIRECT_URL"],
            f'"postgresql://postgres:{root["POSTGRES_PASSWORD"]}'
            '@localhost:5432/postgres?schema=platform"',
        )

    def test_credentials_an_older_file_lacks_are_added_but_keys_are_not(self) -> None:
        (self.root / "backend").mkdir()
        (self.root / "backend/.env.default").write_text(
            "REDIS_PASSWORD=\nGRAPHITI_FALKORDB_PASSWORD=\nENCRYPTION_KEY=\n",
            encoding="utf-8",
        )
        # A file copied before these existed: the shards' password was
        # commented out, FalkorDB's came from .env.default.
        self._write(backend="# REDIS_PASSWORD=\nOPENAI_API_KEY=sk-test")

        filled = runtime_config.fill_env_blanks(self.root / "backend/.env")

        self.assertEqual(filled, ["REDIS_PASSWORD", "GRAPHITI_FALKORDB_PASSWORD"])
        content = (self.root / "backend/.env").read_text(encoding="utf-8")
        self.assertTrue(
            content.startswith("# REDIS_PASSWORD=\nOPENAI_API_KEY=sk-test\n")
        )
        _, backend, _ = self._values()
        self.assertGreaterEqual(len(backend["REDIS_PASSWORD"]), 32)
        # The backend refuses to start without it, and the upgrade guide moves
        # stored credentials to a new one; generating it here would skip that.
        self.assertNotIn("ENCRYPTION_KEY", backend)

    def test_second_run_changes_nothing(self) -> None:
        (self.root / "backend").mkdir()
        (self.root / "backend/.env.default").write_text(
            "REDIS_PASSWORD=\n", encoding="utf-8"
        )
        self._write(
            root=f"POSTGRES_PASSWORD={self.PUBLISHED}\n",
            backend="DB_PASS=\n",
            frontend="DATABASE_URL=\n",
        )
        paths = [
            self.root / ".env",
            self.root / "backend/.env",
            self.root / "frontend/.env",
        ]
        _run_fill_env(*paths)
        first = [path.read_text(encoding="utf-8") for path in paths]

        self.assertEqual(
            [runtime_config.fill_env_blanks(path) for path in paths], [[], [], []]
        )
        self.assertEqual([path.read_text(encoding="utf-8") for path in paths], first)

    def test_crlf_files_from_a_windows_checkout_are_filled(self) -> None:
        self._write(
            root=f"# Database\r\nPOSTGRES_PASSWORD={self.PUBLISHED}\r\nPOSTGRES_PORT=5432\r\n",
            backend="DB_PASS=\r\n",
        )

        _run_fill_env(self.root / ".env", self.root / "backend/.env")

        root, backend, _ = self._values()
        self.assertNotIn(self.PUBLISHED, root["POSTGRES_PASSWORD"])
        self.assertGreaterEqual(len(root["POSTGRES_PASSWORD"].strip()), 32)
        self.assertEqual(backend["DB_PASS"].strip(), root["POSTGRES_PASSWORD"].strip())
        self.assertEqual(root["POSTGRES_PORT"].strip(), "5432")

    def test_values_are_read_as_compose_reads_them(self) -> None:
        """A comment, quotes or `export` must not hide a published password,
        and quotes around nothing must not pass for a password."""
        self._write(
            root=f"export POSTGRES_PASSWORD={self.PUBLISHED}  # from the old file\n",
            backend=(
                f"RABBITMQ_DEFAULT_PASS={self.PUBLISHED} # legacy\n"
                f"GRAPHITI_FALKORDB_PASSWORD='{self.PUBLISHED}'\n"
                'REDIS_PASSWORD=""\n'
                "DB_PASS =\n"
            ),
        )

        _run_fill_env(self.root / ".env", self.root / "backend/.env")

        root_text = (self.root / ".env").read_text(encoding="utf-8")
        self.assertTrue(root_text.startswith("export POSTGRES_PASSWORD="))
        self.assertTrue(root_text.endswith("  # from the old file\n"))
        self.assertNotIn(self.PUBLISHED, root_text)
        _, backend, _ = self._values()
        for name in (
            "RABBITMQ_DEFAULT_PASS",
            "GRAPHITI_FALKORDB_PASSWORD",
            "REDIS_PASSWORD",
        ):
            self.assertNotIn(self.PUBLISHED, backend[name], name)
            self.assertGreaterEqual(len(backend[name]), 32, name)
        password = root_text.split("=", 1)[1].split(" #")[0].strip()
        self.assertEqual(backend["DB_PASS"], password)

    def test_the_last_of_several_assignments_is_the_one_shared(self) -> None:
        self._write(
            root="POSTGRES_PASSWORD=earlier-value\nPOSTGRES_PASSWORD=later-value\n",
            backend="DB_PASS=\n",
        )

        _run_fill_env(self.root / ".env", self.root / "backend/.env")

        _, backend, _ = self._values()
        self.assertEqual(backend["DB_PASS"], "later-value")

    def test_a_password_a_connection_string_cant_carry_is_refused(self) -> None:
        bell = "DATABASE_URL=postgresql://postgres:pa%07ss@localhost:5432/postgres\n"
        for root, backend, frontend in (
            ("POSTGRES_PASSWORD=pa?ss\n", "DB_PASS=\n", ""),
            ("POSTGRES_PASSWORD=\n", "DB_PASS=pa#ss\n", ""),
            # A control character, percent-encoded in the frontend's copy.
            ("POSTGRES_PASSWORD=\n", "DB_PASS=\n", bell),
        ):
            with self.subTest(root=root, backend=backend, frontend=frontend):
                for path in self.root.rglob(".env"):
                    path.unlink()
                self._write(root=root, backend=backend, frontend=frontend)
                arguments = ["fill-env", "--path", str(self.root / ".env")]
                with (
                    mock.patch.object(sys, "argv", ["runtime_config.py", *arguments]),
                    mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
                ):
                    self.assertEqual(runtime_config.main(), 2)

                self.assertIn("connection string", stderr.getvalue())
                self.assertEqual((self.root / ".env").read_text(encoding="utf-8"), root)

    def test_a_database_of_ones_own_keeps_its_password(self) -> None:
        external = (
            f"postgresql://postgres:{self.PUBLISHED}@db.example.com:5432/postgres"
        )
        self._write(
            root="POSTGRES_PASSWORD=\n",
            backend=(
                "DB_HOST=db.example.com\n"
                f"DB_PASS={self.PUBLISHED}\n"
                f"DIRECT_URL={external}\n"
            ),
        )

        _run_fill_env(self.root / ".env", self.root / "backend/.env")

        root, backend, _ = self._values()
        self.assertEqual(backend["DB_PASS"], self.PUBLISHED)
        self.assertEqual(backend["DIRECT_URL"], external)
        # The bundled database's own password is still generated.
        self.assertGreaterEqual(len(root["POSTGRES_PASSWORD"]), 32)

    def test_services_of_ones_own_keep_their_passwords(self) -> None:
        (self.root / "backend").mkdir()
        (self.root / "backend/.env.default").write_text(
            "REDIS_PASSWORD=\nRABBITMQ_DEFAULT_PASS=\nGRAPHITI_FALKORDB_PASSWORD=\n",
            encoding="utf-8",
        )
        # A Redis that needs no password, a broker still on a value an older
        # .env.default shipped, and a graph whose password Compose takes from
        # the environment: each is a service of the operator's, not the stack's.
        backend = (
            "REDIS_HOST=redis.internal\n"
            "RABBITMQ_HOST=mq.example.com\n"
            f"RABBITMQ_DEFAULT_PASS={self.PUBLISHED}\n"
            "GRAPHITI_FALKORDB_HOST=graph.example.com\n"
            "GRAPHITI_FALKORDB_PASSWORD=${GRAPH_PASSWORD}\n"
        )
        self._write(backend=backend)

        filled = runtime_config.fill_env_blanks(self.root / "backend/.env")

        self.assertEqual(filled, [])
        self.assertEqual(
            (self.root / "backend/.env").read_text(encoding="utf-8"), backend
        )

    def test_a_service_whose_host_compose_takes_at_start_keeps_its_password(
        self,
    ) -> None:
        # The name alone on a line: Compose takes the host from the environment
        # it starts in, which may point anywhere.
        backend = "REDIS_HOST\nREDIS_PASSWORD=\n"
        self._write(backend=backend)

        filled = runtime_config.fill_env_blanks(self.root / "backend/.env")

        self.assertEqual(filled, [])
        self.assertEqual(
            (self.root / "backend/.env").read_text(encoding="utf-8"), backend
        )

    def test_the_cluster_host_the_backend_reads_first_decides(self) -> None:
        # The backend takes REDIS_CLUSTER_HOST over REDIS_HOST, and
        # RABBITMQ_CLUSTER_HOST over RABBITMQ_HOST, unless it is blank.
        for name, cluster_setting, host_setting, compose_name in (
            ("REDIS_PASSWORD", "REDIS_CLUSTER_HOST", "REDIS_HOST", "redis-0"),
            (
                "RABBITMQ_DEFAULT_PASS",
                "RABBITMQ_CLUSTER_HOST",
                "RABBITMQ_HOST",
                "rabbitmq",
            ),
        ):
            for cluster, host, generated in (
                ("managed.example.com", "localhost", False),
                (compose_name, "managed.example.com", True),
                ("", "managed.example.com", False),
                ("", "localhost", True),
            ):
                with self.subTest(name=name, cluster=cluster, host=host):
                    self._write(
                        backend=(
                            f"{cluster_setting}={cluster}\n"
                            f"{host_setting}={host}\n{name}=\n"
                        )
                    )

                    filled = runtime_config.fill_env_blanks(self.root / "backend/.env")

                    self.assertEqual(filled, [name] if generated else [])

    def test_bundled_services_get_passwords_under_any_name_they_go_by(self) -> None:
        for name, host_setting, compose_name in (
            ("DB_PASS", "DB_HOST", "db"),
            ("REDIS_PASSWORD", "REDIS_HOST", "redis-0"),
            ("REDIS_PASSWORD", "REDIS_HOST", "redis-2"),
            ("RABBITMQ_DEFAULT_PASS", "RABBITMQ_HOST", "rabbitmq"),
            ("GRAPHITI_FALKORDB_PASSWORD", "GRAPHITI_FALKORDB_HOST", "falkordb"),
        ):
            for host in (
                "",
                "localhost",
                "127.0.0.1",
                "::1",
                "LOCALHOST.",
                compose_name,
            ):
                with self.subTest(name=name, host=host):
                    self._write(backend=f"{host_setting}={host}\n{name}=\n")

                    filled = runtime_config.fill_env_blanks(self.root / "backend/.env")

                    self.assertEqual(filled, [name])

    def test_a_reference_to_another_setting_is_not_taken_for_a_password(
        self,
    ) -> None:
        self._write(
            root="POSTGRES_PASSWORD=\n",
            frontend=(
                "DATABASE_URL=postgresql://postgres:${DB_PASS}@localhost:5432/postgres\n"
            ),
        )

        _run_fill_env(self.root / ".env")

        root, _, _ = self._values()
        self.assertGreaterEqual(len(root["POSTGRES_PASSWORD"]), 32)
        self.assertNotIn("$", root["POSTGRES_PASSWORD"])

    def test_dotenv_values_are_parsed_as_compose_parses_them(self) -> None:
        # (as written after `=`, what Compose takes, whether the line alone
        # tells: Compose fills in a `$` reference when the stack starts).
        cases = [
            ("plain", "plain", True),
            ("  padded  ", "padded", True),
            ("value # comment", "value", True),
            ("value#not-a-comment", "value#not-a-comment", True),
            ("value\t# not a comment either", "value\t# not a comment either", True),
            ("'single $X \\n \\\\'", "single $X \\n \\\\", True),
            ("'it\\'s'", "it's", True),
            ('"a\\nb\\tc\\\\d\\"e\\$X"', 'a\nb\tc\\d"e$X', True),
            ('"kept \\x \\q"', "kept \\x \\q", True),
            ('"octal \\0101, not \\012"', "octal A, not \\12", True),
            ('"quoted" # comment', "quoted", True),
            ("pa$$word", "pa$word", True),
            ('"pa$$word"', "pa$word", True),
            ("$1 p$ $! $", "$1 p$ $! $", True),
            ('""', "", True),
            ("", "", True),
            ("$UNSET", None, False),
            ("${UNSET}", None, False),
            ("p$unset", None, False),
            ('"${UNSET:-fallback}"', None, False),
            ("pa$$$UNSET", None, False),
            # Go's case-insensitive [_a-z] takes the Kelvin sign and the long
            # s as letters of a name, but not the dotless i.
            ("pa$\u212ass", None, False),
            ("pa$\u017fss", None, False),
            ("pa$\u0131ss", "pa$\u0131ss", True),
        ]
        for raw, value, literal in cases:
            with self.subTest(raw=raw):
                setting = _compose_settings(f"X={raw}\n")["X"]
                self.assertEqual(setting.literal, literal)
                if value is not None:
                    self.assertEqual(setting.value, value)
        # Compose refuses a whole file with an unclosed quote in it.
        for raw in ("'unclosed", '"unclosed \\"'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _compose_settings(f"X={raw}\nY=1\n")

    def test_whole_files_are_read_as_compose_reads_them(self) -> None:
        """Each as `docker compose config` reads it: it is not lines that
        matter but statements, which may be indented, run on over lines,
        or share one."""
        cases = [
            ("\ufeffA=1\n", {"A": "1"}),
            ("B: yaml-style\n", {"B": "yaml-style"}),
            ("  C=indented\n\tC2=tab\n", {"C": "indented", "C2": "tab"}),
            (
                "D=\"x\" E=two-on-a-line\nF='y'G=no-space\n",
                {"D": "x", "E": "two-on-a-line", "F": "y", "G": "no-space"},
            ),
            (
                'H="multi\nI=inside\nline"\nJ=after\n',
                {"H": "multi\nI=inside\nline", "J": "after"},
            ),
            ("export\tL=export-tab\n", {"L": "export-tab"}),
            ("M = spaced\n", {"M": "spaced"}),
            ("N=crlf\r\nO=too\r\n", {"N": "crlf", "O": "too"}),
            ("P=no-final-newline", {"P": "no-final-newline"}),
            ("S= # not a comment\n", {"S": "# not a comment"}),
            ("T.u-v[w]=key-characters\n", {"T.u-v[w]": "key-characters"}),
        ]
        for text, values in cases:
            with self.subTest(text=text):
                settings = _compose_settings(text)
                self.assertEqual({k: s.value for k, s in settings.items()}, values)
                self.assertTrue(all(s.literal for s in settings.values()))
        # The name alone takes the environment's value, if there is one.
        self.assertFalse(_compose_settings("K\nK2=set\n")["K"].literal)
        # A template no `}` closes runs to the last `}` on its line, so these
        # braces belong to its default.
        for text in (
            "A=${FOO:-${BAR}}\n",
            'A="${FOO:-a}b}"\n',
            "A=$${\n",
            "A=${FOO:-{}}\n",
            "A=${FOO:-{x}\n",
            "A=${FOO:-{}x}\n",
            "A=${FOO:-$${}\n",
        ):
            with self.subTest(text=text):
                _compose_settings(text)
        # Compose refuses these files outright.
        for text in (
            "U V=1\n",
            "W$=1\n",
            "A=${}\n",
            'A="${FOO"\n',
            "A=${1}\n",
            "A=x${FOO.BAR}\n",
            # One `}` can't close both.
            "A=${FOO:-${BAR}\n",
            # What follows the `}` that closes one is read as well.
            "A=${FOO:-a}${}\n",
            # A lone `{` opens a brace too, so no `}` closes this one: its
            # default runs to the last `}`, leaving `${BAR:-x` without one.
            "A=${FOO:-{}${BAR:-x}\n",
            "A=${\u0131}\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                _compose_settings(text)

    def test_a_service_password_compose_works_out_at_startup_is_refused(
        self,
    ) -> None:
        """Compose fills it in when the stack starts, possibly with nothing,
        so the service could end up with no password at all. A line break is
        refused too: no service can take one."""
        plain = "not a plain value"
        for root, backend, message in (
            ("POSTGRES_PASSWORD=$PG_PASSWORD\n", "", plain),
            ("", "REDIS_PASSWORD=$UNSET\n", plain),
            ("", "REDIS_PASSWORD=${UNSET}\n", plain),
            ("", "REDIS_PASSWORD=p$unset # one letter once Compose is done\n", plain),
            # The Kelvin sign starts a name as K does: the service would get
            # `strong`.
            ("", "REDIS_PASSWORD=strong$\u212a\n", plain),
            ("", 'RABBITMQ_DEFAULT_PASS="${UNSET:-short}"\n', plain),
            ("", "DB_HOST=localhost\nDB_PASS=${UNSET}\n", plain),
            # The name alone: from the environment, if it is set there.
            ("", "REDIS_PASSWORD\n", plain),
            # Indented, it is still a setting, and the last one.
            (
                "POSTGRES_PASSWORD=strong-one\n  POSTGRES_PASSWORD=${UNSET:-x}\n",
                "",
                plain,
            ),
            ("", 'GRAPHITI_FALKORDB_PASSWORD="line-one\\nline-two"\n', plain),
            # Compose refuses these files outright.
            ("", "GRAPHITI_FALKORDB_PASSWORD='unclosed\n", "can't read"),
            ("", "MY KEY=1\nREDIS_PASSWORD=\n", "can't read"),
            ("", "UNRELATED=${}\nREDIS_PASSWORD=\n", "can't read"),
        ):
            with self.subTest(root=root, backend=backend):
                for path in self.root.rglob(".env"):
                    path.unlink()
                self._write(root=root or "POSTGRES_PASSWORD=\n", backend=backend)
                target = self.root / (".env" if root else "backend/.env")
                before = target.read_text(encoding="utf-8")
                arguments = ["fill-env", "--path", str(target)]
                with (
                    mock.patch.object(sys, "argv", ["runtime_config.py", *arguments]),
                    mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
                ):
                    self.assertEqual(runtime_config.main(), 2)

                self.assertIn(message, stderr.getvalue())
                self.assertEqual(target.read_text(encoding="utf-8"), before)

    def test_a_database_password_set_in_the_shell_is_pointed_out(self) -> None:
        """Compose takes POSTGRES_PASSWORD from the shell over the stack's
        .env, so one exported there would replace the generated password."""
        self._write(root="POSTGRES_PASSWORD=\n")
        root = self.root / ".env"
        warning = "POSTGRES_PASSWORD is also set in this shell"
        with mock.patch.dict(os.environ):
            os.environ.pop("POSTGRES_PASSWORD", None)
            self.assertNotIn(warning, _run_fill_env(root))
            os.environ["POSTGRES_PASSWORD"] = "exported-for-another-project"
            self.assertIn(warning, _run_fill_env(root))
            os.environ["POSTGRES_PASSWORD"] = _env_values(root)["POSTGRES_PASSWORD"]
            self.assertNotIn(warning, _run_fill_env(root))

    def test_a_setting_inside_another_value_is_not_one(self) -> None:
        """Compose reads a quoted value on over lines: a `NAME=` line inside
        it belongs to that value, so the password it seems to set is still
        missing and is added, and the value is left as it was."""
        (self.root / "backend").mkdir()
        (self.root / "backend/.env.default").write_text(
            "REDIS_PASSWORD=\n", encoding="utf-8"
        )
        other = 'OTHER="begins\nREDIS_PASSWORD=looks-strong\nends"\n'
        self._write(backend=other)

        _run_fill_env(self.root / "backend/.env")

        text = (self.root / "backend/.env").read_text(encoding="utf-8")
        self.assertTrue(text.startswith(other))
        settings = _compose_settings(text)
        self.assertEqual(
            settings["OTHER"].value, "begins\nREDIS_PASSWORD=looks-strong\nends"
        )
        self.assertGreaterEqual(len(settings["REDIS_PASSWORD"].value), 32)

    def test_values_change_in_place(self) -> None:
        """Indentation, `export`, `:` for `=`, quotes, comments, a statement
        sharing the line and CRLF line endings stay: only values change."""
        self._write(
            backend=(
                f'  export RABBITMQ_DEFAULT_PASS: "{self.PUBLISHED}"  # old\r\n'
                "REDIS_PASSWORD=\r\n"
                f'A="x" GRAPHITI_FALKORDB_PASSWORD={self.PUBLISHED}\r\n'
            ),
        )

        _run_fill_env(self.root / "backend/.env")

        text = (self.root / "backend/.env").read_bytes().decode("utf-8")
        values = {k: s.value for k, s in _compose_settings(text).items()}
        rabbitmq = values["RABBITMQ_DEFAULT_PASS"]
        redis = values["REDIS_PASSWORD"]
        falkordb = values["GRAPHITI_FALKORDB_PASSWORD"]
        self.assertEqual(
            text.split("\r\n"),
            [
                f"  export RABBITMQ_DEFAULT_PASS:{rabbitmq}  # old",
                f"REDIS_PASSWORD={redis}",
                f'A="x" GRAPHITI_FALKORDB_PASSWORD={falkordb}',
                "",
            ],
        )
        for value in (rabbitmq, redis, falkordb):
            self.assertGreaterEqual(len(value), 32)

    def test_an_indented_db_host_still_names_a_database_of_its_own(self) -> None:
        self._write(
            root="POSTGRES_PASSWORD=\n",
            backend="  DB_HOST=db.example.com\nDB_PASS=external-one\n",
        )

        _run_fill_env(self.root / ".env", self.root / "backend/.env")

        root, backend, _ = self._values()
        self.assertGreaterEqual(len(root["POSTGRES_PASSWORD"]), 32)
        self.assertEqual(backend["DB_PASS"], "external-one")

    def test_a_template_connection_string_is_left_alone(self) -> None:
        """A template with a bracketed host can't be taken apart before Compose
        fills it in: it holds no password to share, and no reason to stop."""
        for url in (
            "postgresql://${DB_USER}:${DB_PASS}@[${DB_HOST}]:5432/postgres",
            # Nor does one with a broken host, as written.
            "postgresql://postgres:operator-chosen@[::1:5432/postgres",
        ):
            with self.subTest(url=url):
                self._write(
                    root="POSTGRES_PASSWORD=\n", frontend=f"DATABASE_URL={url}\n"
                )

                _run_fill_env(self.root / ".env", self.root / "frontend/.env")

                root, _, frontend = self._values()
                self.assertGreaterEqual(len(root["POSTGRES_PASSWORD"]), 32)
                self.assertEqual(frontend["DATABASE_URL"], url)

    def test_a_dollar_written_literally_is_kept(self) -> None:
        backend = (
            "REDIS_PASSWORD='pa$sword-in-single-quotes'\n"
            "RABBITMQ_DEFAULT_PASS=pa$$word-escaped\n"
            'GRAPHITI_FALKORDB_PASSWORD="pa\\$word-escaped"\n'
            # A database of the backend's own may use whatever it likes.
            "DB_HOST=db.example.com\n"
            "DB_PASS=${EXTERNAL_DB_PASSWORD}\n"
        )
        self._write(root="POSTGRES_PASSWORD=\n", backend=backend)

        _run_fill_env(self.root / "backend/.env")

        self.assertEqual(
            (self.root / "backend/.env").read_text(encoding="utf-8"), backend
        )

    def test_double_quoted_escapes_are_read_as_compose_decodes_them(self) -> None:
        """`\\n` in double quotes is a line break to Compose, so the password
        can't go into a connection string; read as `n` it would have passed."""
        for value, message in (
            ('"pa\\nss"', "control character"),
            ('"pa\\tss"', "control character"),
            ('"pa\\\\ss"', "connection string"),
        ):
            with self.subTest(value=value):
                root = f"POSTGRES_PASSWORD={value}\n"
                self._write(root=root)
                arguments = ["fill-env", "--path", str(self.root / ".env")]
                with (
                    mock.patch.object(sys, "argv", ["runtime_config.py", *arguments]),
                    mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
                ):
                    self.assertEqual(runtime_config.main(), 2)

                self.assertIn(message, stderr.getvalue())
                self.assertEqual((self.root / ".env").read_text(encoding="utf-8"), root)

    def test_an_external_databases_password_is_never_copied(self) -> None:
        """A backend or frontend pointed at a database of its own holds that
        database's password: it must not become the bundled database's."""
        frontend_url = "postgresql://app:frontend-own-secret@pg.example.com:5432/app"
        self._write(
            root="POSTGRES_PASSWORD=\n",
            backend="DB_HOST=db.example.com\nDB_PASS=ext@rnal-secret\n",
            frontend=f"DATABASE_URL={frontend_url}\n",
        )

        _run_fill_env(
            self.root / ".env", self.root / "backend/.env", self.root / "frontend/.env"
        )

        root, backend, frontend = self._values()
        password = root["POSTGRES_PASSWORD"]
        self.assertGreaterEqual(len(password), 32)
        self.assertNotIn("secret", password)
        self.assertEqual(backend["DB_PASS"], "ext@rnal-secret")
        self.assertEqual(frontend["DATABASE_URL"], frontend_url)

    def test_the_bundled_database_is_recognised_however_its_host_is_written(
        self,
    ) -> None:
        for host in ("LOCALHOST", "localhost.", "DB"):
            with self.subTest(host=host):
                self._write(
                    root="POSTGRES_PASSWORD=operator-chosen-password\n",
                    backend=f"DB_HOST={host}\nDB_PASS=\n",
                )

                _run_fill_env(self.root / "backend/.env")

                _, backend, _ = self._values()
                self.assertEqual(backend["DB_PASS"], "operator-chosen-password")

    def test_a_connection_string_keeps_its_quotes_and_comment(self) -> None:
        self._write(
            root="POSTGRES_PASSWORD=operator-chosen-password\n",
            frontend=(
                "DATABASE_URL='postgresql://postgres:"
                f"{self.PUBLISHED}@localhost:5432/postgres' # bundled database\n"
            ),
        )

        _run_fill_env(self.root / "frontend/.env")

        self.assertEqual(
            (self.root / "frontend/.env").read_text(encoding="utf-8"),
            "DATABASE_URL='postgresql://postgres:operator-chosen-password"
            "@localhost:5432/postgres' # bundled database\n",
        )

    @unittest.skipIf(sys.platform == "win32", "symlinks need a privilege there")
    def test_a_symlinked_file_is_read_but_never_written(self) -> None:
        target = self.root / "shared.env"
        target.write_text(
            "POSTGRES_PASSWORD=operator-chosen-password\n", encoding="utf-8"
        )
        (self.root / ".env").symlink_to(target)
        self._write(backend="DB_PASS=\n", frontend="DATABASE_URL=\n")

        with mock.patch("sys.stderr", new_callable=io.StringIO):
            _run_fill_env(
                self.root / ".env",
                self.root / "backend/.env",
                self.root / "frontend/.env",
            )

        _, backend, frontend = self._values()
        self.assertEqual(backend["DB_PASS"], "operator-chosen-password")
        self.assertIn(":operator-chosen-password@", frontend["DATABASE_URL"])
        self.assertTrue((self.root / ".env").is_symlink())

    def _write(self, root: str = "", backend: str = "", frontend: str = "") -> None:
        for relative_path, content in (
            (".env", root),
            ("backend/.env", backend),
            ("frontend/.env", frontend),
        ):
            if content:
                path = self.root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content.encode())

    def _values(self) -> list[dict[str, str]]:
        return [
            _env_values(self.root / relative_path)
            for relative_path in (".env", "backend/.env", "frontend/.env")
        ]


def _copy_env_default(platform_root: Path, root: Path, relative_path: str) -> Path:
    env_path = root / relative_path.replace(".env.default", ".env")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        (platform_root / relative_path).read_text(encoding="utf-8"), encoding="utf-8"
    )
    return env_path


def _run_fill_env(*paths: Path) -> str:
    """Run `fill-env` on `paths`, and return what it wrote to stderr."""
    arguments = ["runtime_config.py", "fill-env"]
    for path in paths:
        arguments += ["--path", str(path)]
    with (
        mock.patch.object(sys, "argv", arguments),
        mock.patch("sys.stdout", new_callable=io.StringIO),
        mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
    ):
        assert runtime_config.main() == 0
    return stderr.getvalue()


def _compose_settings(text: str) -> dict[str, runtime_config.Setting]:
    return runtime_config._effective(runtime_config._settings(text))


def _env_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        name.strip(): value
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.startswith("#")
        for name, value in [line.split("=", 1)]
    }


def _version_1_values() -> dict[str, str]:
    """Every key a version 1 image wrote, before INTERNAL_API_KEY existed."""
    return {
        "AUTOGPT_RUNTIME_CONFIG_VERSION": "1",
        "POSTGRES_PASSWORD": "p" * 40,
        "RABBITMQ_DEFAULT_USER": "autogpt",
        "RABBITMQ_DEFAULT_PASS": "r" * 40,
        "REDIS_PASSWORD": "d" * 40,
        "BETTER_AUTH_SECRET": "b" * 40,
        "ENCRYPTION_KEY": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
        "UNSUBSCRIBE_SECRET_KEY": "u" * 40,
        "GRAPHITI_FALKORDB_PASSWORD": "f" * 40,
        "VAPID_PRIVATE_KEY": base64.urlsafe_b64encode(b"p" * 32).rstrip(b"=").decode(),
        "VAPID_PUBLIC_KEY": (
            base64.urlsafe_b64encode(b"\x04" + b"q" * 64).rstrip(b"=").decode()
        ),
    }


def _write_version_1_config(path: Path) -> dict[str, str]:
    values = _version_1_values()
    _write_config_lines(path, values)
    return values


def _write_config_lines(path: Path, values: dict[str, str]) -> None:
    lines = ["# Generated once by the AutoGPT all-in-one image."]
    lines += [f"{name}={value}" for name, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    path.chmod(0o600)


def _replace_config_value(content: str, name: str, value: str) -> str:
    prefix = f"{name}="
    lines = [
        f"{prefix}{value}" if line.startswith(prefix) else line
        for line in content.splitlines()
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    unittest.main()
