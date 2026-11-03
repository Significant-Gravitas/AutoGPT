import tempfile
import unittest
from pathlib import Path

import yaml
from frontend_ci_env import RUNTIME_SETTINGS, prepare_build_environment


class FrontendBuildEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.compose = Path(self.directory.name) / "compose.yml"
        self.env = Path(self.directory.name) / ".env"

    def write_run(self, suffix):
        runtime = {
            "BETTER_AUTH_SECRET": f"ephemeral-auth-{suffix}",
            "DATABASE_URL": f"postgresql://postgres:password-{suffix}@db:5432/postgres",
        }
        self.compose.write_text(
            yaml.safe_dump({"services": {"frontend": {"environment": runtime}}}),
            encoding="utf-8",
        )
        self.env.write_text(
            "".join(f"{key}={value}\n" for key, value in runtime.items())
            + "NEXT_PUBLIC_FRONTEND_BASE_URL=http://localhost:3000\n",
            encoding="utf-8",
        )

    def test_fresh_runtime_credentials_produce_identical_build_inputs(self):
        outputs = []
        for suffix in ("first", "second"):
            self.write_run(suffix)
            runtime = self.compose.read_bytes()
            prepare_build_environment(self.compose, self.env)
            self.assertEqual(self.compose.read_bytes(), runtime)
            outputs.append(self.env.read_text(encoding="utf-8"))
        self.assertEqual(outputs[0], outputs[1])
        self.assertIn("NEXT_PUBLIC_FRONTEND_BASE_URL=http://localhost:3000", outputs[0])
        for name in RUNTIME_SETTINGS:
            self.assertIn(f"{name}=\n", outputs[0])

    def test_missing_runtime_credentials_fail_before_changing_the_build(self):
        for name in RUNTIME_SETTINGS:
            for value in (None, "", " "):
                with self.subTest(name=name, value=value):
                    self.write_run("first")
                    runtime = yaml.safe_load(self.compose.read_text(encoding="utf-8"))
                    runtime["services"]["frontend"]["environment"][name] = value
                    self.compose.write_text(yaml.safe_dump(runtime), encoding="utf-8")
                    original = self.env.read_bytes()
                    with self.assertRaises(ValueError):
                        prepare_build_environment(self.compose, self.env)
                    self.assertEqual(self.env.read_bytes(), original)

    def test_missing_or_repeated_build_setting_is_rejected(self):
        for name in RUNTIME_SETTINGS:
            for duplicate in (False, True):
                with self.subTest(name=name, duplicate=duplicate):
                    self.write_run("first")
                    text = self.env.read_text(encoding="utf-8")
                    if duplicate:
                        text += f"{name}=duplicate\n"
                    else:
                        text = "\n".join(
                            line
                            for line in text.splitlines()
                            if not line.startswith(name + "=")
                        )
                    self.env.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        prepare_build_environment(self.compose, self.env)

    def test_public_build_configuration_changes_remain_visible(self):
        self.write_run("first")
        prepare_build_environment(self.compose, self.env)
        first = self.env.read_text(encoding="utf-8")
        self.env.write_text(
            first.replace("localhost:3000", "example.invalid"), encoding="utf-8"
        )
        prepare_build_environment(self.compose, self.env)
        self.assertNotEqual(first, self.env.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
