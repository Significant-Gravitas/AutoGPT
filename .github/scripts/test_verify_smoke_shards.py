import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

import yaml
from test_validate_playwright_json import valid_report
from verify_smoke_shards import SHARD_COUNT, verify, verify_directory, verify_seed


def reports():
    inventory = valid_report()
    inventory["config"]["projects"][0]["repeatEach"] = 1
    inventory["stats"]["expected"] = 32
    template = inventory["suites"][0]["specs"][0]
    specs = []
    for number in range(32):
        spec = copy.deepcopy(template)
        spec["id"] = f"test-{number}"
        spec["tests"][0]["projectName"] = "chromium"
        specs.append(spec)
    inventory["suites"][0]["specs"] = specs
    shards = {}
    for number in range(1, SHARD_COUNT + 1):
        report = copy.deepcopy(inventory)
        report["config"]["shard"] = {"current": number, "total": SHARD_COUNT}
        report["suites"][0]["specs"] = specs[number - 1 :: SHARD_COUNT]
        report["stats"]["expected"] = len(report["suites"][0]["specs"])
        shards[number] = report
    return inventory, shards


class SmokeShardTests(unittest.TestCase):
    def test_complete_shards_pass(self):
        self.assertEqual(verify(*reports()), 32)

    def test_silently_omitted_passing_test_fails(self):
        inventory, shards = reports()
        shards[4]["suites"][0]["specs"].pop()
        shards[4]["stats"]["expected"] -= 1
        with self.assertRaisesRegex(ValueError, "Shard union"):
            verify(inventory, shards)

    def test_duplicate_or_misidentified_shard_fails(self):
        for duplicate in (True, False):
            with self.subTest(duplicate=duplicate):
                inventory, shards = reports()
                if duplicate:
                    shards[4]["suites"] = shards[1]["suites"]
                else:
                    shards[4]["config"]["shard"]["current"] = 1
                with self.assertRaises(ValueError):
                    verify(inventory, shards)

    def test_missing_shard_or_failed_test_fails(self):
        inventory, shards = reports()
        with self.assertRaises(ValueError):
            verify(inventory, {key: value for key, value in shards.items() if key != 4})
        for status in ("skipped", "failed", "timedOut"):
            with self.subTest(status=status):
                inventory, shards = reports()
                test = shards[1]["suites"][0]["specs"][0]["tests"][0]
                test["results"][0]["status"] = status
                with self.assertRaises(ValueError):
                    verify(inventory, shards)

    def test_partial_or_broken_discovery_fails(self):
        for broken in (True, False):
            inventory, shards = reports()
            if broken:
                inventory["errors"] = [{"message": "collection failed"}]
            else:
                inventory["config"]["shard"] = {"current": 1, "total": SHARD_COUNT}
            with self.assertRaises(ValueError):
                verify(inventory, shards)

    def test_missing_artifact_and_changed_discovery_fail(self):
        inventory, shards = reports()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for number, report in shards.items():
                folder = root / f"playwright-test-results-smoke-{number}"
                folder.mkdir()
                (folder / "results.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
                (folder / "smoke-inventory.json").write_text(
                    json.dumps(inventory), encoding="utf-8"
                )
            self.assertEqual(verify_directory(root), 32)
            path = root / "playwright-test-results-smoke-4" / "smoke-inventory.json"
            path.unlink()
            with self.assertRaises(ValueError):
                verify_directory(root)
            inventory["suites"][0]["specs"][0]["id"] = "different"
            path.write_text(json.dumps(inventory), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_directory(root)

    def test_seed_requires_the_complete_suite_on_the_database_being_published(self):
        inventory, shards = reports()
        self.assertEqual(verify_seed(inventory, inventory), 32)
        with self.assertRaises(ValueError):
            verify_seed(inventory, shards[1])
        partial = copy.deepcopy(inventory)
        partial["suites"][0]["specs"].pop()
        partial["stats"]["expected"] -= 1
        with self.assertRaises(ValueError):
            verify_seed(inventory, partial)

    def test_workflow_requires_every_shard_and_complete_seed_validation(self):
        root = Path(__file__).resolve().parents[2]
        text = (root / ".github/workflows/platform-fullstack-ci.yml").read_text(
            encoding="utf-8"
        )
        workflow = yaml.safe_load(text)
        legs = json.loads(re.findall(r"'(\[\{[^\n]+\}\])'", text)[0])
        self.assertEqual(
            {leg["partition"] for leg in legs}, set(range(1, SHARD_COUNT + 1))
        )
        for leg in legs:
            self.assertEqual(
                leg["playwright_args"], f"--shard={leg['partition']}/{SHARD_COUNT}"
            )
        gate = workflow["jobs"]["validate-smoke-shards"]
        self.assertEqual(gate["needs"], "e2e_test")
        self.assertIn("always()", gate["if"])
        self.assertIn("!cancelled()", gate["if"])
        steps = workflow["jobs"]["e2e_test"]["steps"]
        seed = next(step for step in steps if step.get("id") == "e2e-seed-suite")
        shard = next(step for step in steps if step.get("id") == "e2e-playwright")
        self.assertLess(steps.index(seed), steps.index(shard))
        self.assertNotIn("--shard", seed["run"])
        self.assertIn("verify_smoke_shards.py seed", seed["run"])
        publish = next(
            step for step in steps if step.get("name") == "Push E2E test data to GHCR"
        )
        self.assertIn("steps.e2e-seed-suite.outcome == 'success'", publish["if"])
        codecov = yaml.safe_load((root / "codecov.yml").read_text(encoding="utf-8"))
        self.assertEqual(
            codecov["flags"]["platform-frontend-e2e"]["after_n_builds"], SHARD_COUNT
        )


if __name__ == "__main__":
    unittest.main()
