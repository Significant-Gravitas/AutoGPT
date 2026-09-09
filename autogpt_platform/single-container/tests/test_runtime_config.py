from __future__ import annotations

import base64
import importlib.util
import io
import re
import stat
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
                    "AUTOGPT_RUNTIME_CONFIG_VERSION=1",
                    "AUTOGPT_RUNTIME_CONFIG_VERSION=2",
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

            filled = runtime_config.fill_env_secrets(path)

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

            runtime_config.fill_env_secrets(path)
            after_first = path.read_text(encoding="utf-8")
            self.assertEqual(runtime_config.fill_env_secrets(path), [])
            self.assertEqual(path.read_text(encoding="utf-8"), after_first)

    def test_generated_values_differ_between_installs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            keys = []
            for index in range(2):
                path = Path(directory) / f"{index}.env"
                path.write_text("ENCRYPTION_KEY=\n", encoding="utf-8")
                runtime_config.fill_env_secrets(path)
                keys.append(path.read_text(encoding="utf-8"))
            self.assertNotEqual(keys[0], keys[1])

    def test_preserves_file_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("ENCRYPTION_KEY=\n", encoding="utf-8")
            path.chmod(0o600)

            runtime_config.fill_env_secrets(path)

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_env_default_ships_its_secrets_blank_and_generatable(self) -> None:
        """`.env.default` is public, so these must ship blank (SECRT-2611) —
        and every blank one must be fillable, or setup leaves a broken stack."""
        platform_root = Path(__file__).parents[2]
        self.assertEqual(runtime_config.check_env_defaults(platform_root), [])

        with tempfile.TemporaryDirectory() as directory:
            for relative_path, names in runtime_config.BLANK_IN_ENV_DEFAULT.items():
                content = (platform_root / relative_path).read_text(encoding="utf-8")
                copy = Path(directory) / relative_path.replace("/", "_")
                copy.write_text(content, encoding="utf-8")

                # Blank is only safe if setup can actually fill it back in.
                self.assertEqual(
                    sorted(runtime_config.fill_env_secrets(copy)), sorted(names)
                )


def _replace_config_value(content: str, name: str, value: str) -> str:
    prefix = f"{name}="
    lines = [
        f"{prefix}{value}" if line.startswith(prefix) else line
        for line in content.splitlines()
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    unittest.main()
