import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frontend_typecheck import compatibility_config, main


class TypecheckCompatibilityTests(unittest.TestCase):
    def test_implicit_global_types_and_aliases_remain_in_scope(self):
        config = compatibility_config(
            {"paths": {"@/*": ["src/*"], "shared": ["../shared", "./local"]}},
            ["node", "react", "vitest"],
        )
        self.assertEqual(config["extends"], "./tsconfig.json")
        self.assertEqual(
            config["compilerOptions"]["types"], ["node", "react", "vitest"]
        )
        self.assertEqual(
            config["compilerOptions"]["paths"],
            {"@/*": ["./src/*"], "shared": ["../shared", "./local"]},
        )
        self.assertFalse(config["compilerOptions"]["noUncheckedSideEffectImports"])

    def test_explicit_project_choices_override_legacy_defaults(self):
        for types in ([], ["node"]):
            config = compatibility_config(
                {"types": types, "noUncheckedSideEffectImports": True}, ["react"]
            )
            self.assertEqual(config["compilerOptions"]["types"], types)
            self.assertTrue(config["compilerOptions"]["noUncheckedSideEffectImports"])

    def test_type_errors_fail_the_job_and_remove_the_temporary_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "tsconfig.ci-typecheck.json"
            with (
                patch("frontend_typecheck.Path", return_value=config),
                patch(
                    "frontend_typecheck.subprocess.check_output",
                    side_effect=[json.dumps({"compilerOptions": {}}), '["node"]'],
                ),
                patch(
                    "frontend_typecheck.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 2),
                ),
            ):
                self.assertEqual(main(), 2)
            self.assertFalse(config.exists())


if __name__ == "__main__":
    unittest.main()
