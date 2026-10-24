import contextlib
import copy
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from security_release_notes import BASE_URL, END, START, main, release_notes

TAG = "autogpt-platform-beta-v0.9.0"


def advisory(ghsa="GHSA-aaaa-bbbb-cccc"):
    return {
        "ghsa_id": ghsa,
        "state": "published",
        "published_at": "2026-10-24T12:00:00Z",
        "withdrawn_at": None,
        "summary": "Example fix",
        "description": f"Fixed in [{TAG}]({BASE_URL}/releases/tag/{TAG}).",
        "vulnerabilities": [{"patched_versions": "0.9.0"}],
        "credits_detailed": [
            {"user": {"login": "researcher"}, "state": "accepted", "type": "reporter"}
        ],
    }


class ReleaseNotesTest(unittest.TestCase):
    def test_publication_and_consent_boundaries(self):
        public = advisory()
        public["credits_detailed"] += [
            {"user": {"login": "pending"}, "state": "pending", "type": "reporter"},
            {"user": {"login": "declined"}, "state": "declined", "type": "reporter"},
            {
                "user": {"login": "maintainer"},
                "state": "accepted",
                "type": "remediation_developer",
            },
        ]
        excluded = []
        for state in ("triage", "draft", "closed"):
            item = advisory()
            item.update(state=state, summary=f"SECRET-{state}")
            excluded.append(item)
        for field in ("published_at", "withdrawn_at"):
            item = advisory()
            item.update(summary=f"SECRET-{field}")
            item[field] = None if field == "published_at" else "2026-10-24"
            excluded.append(item)
        result = release_notes([public, *excluded], TAG)
        self.assertIn("Example fix", result)
        self.assertIn("@researcher", result)
        for forbidden in ("SECRET", "@pending", "@declined", "@maintainer"):
            self.assertNotIn(forbidden, result)

    def test_deleted_credit_users_and_missing_ids_do_not_hide_valid_advisories(self):
        valid = advisory()
        valid["credits_detailed"] += [
            {"user": None, "state": "accepted", "type": "reporter"},
            {"state": "accepted", "type": "reporter"},
        ]
        missing_id = advisory()
        del missing_id["ghsa_id"]
        null_id = advisory(None)
        result = release_notes([missing_id, valid, null_id], TAG)
        self.assertEqual(result.count("Example fix"), 1)
        self.assertIn("@researcher", result)

    def test_only_valid_cve_identifiers_are_rendered(self):
        item = advisory()
        item["cve_id"] = "CVE-2026-12345"
        self.assertIn("(CVE-2026-12345)", release_notes([item], TAG))
        for cve in (None, "CVE-2026-12", "CVE-2026-12345<script>"):
            with self.subTest(cve=cve):
                item["cve_id"] = cve
                self.assertNotIn("CVE-", release_notes([item], TAG))

    def test_requires_matching_fixed_version_and_exact_release_reference(self):
        good = advisory()
        wrong_version = copy.deepcopy(good)
        wrong_version["vulnerabilities"][0]["patched_versions"] = "0.8.0"
        wrong_tag = copy.deepcopy(good)
        wrong_tag["description"] = wrong_tag["description"].replace(TAG, TAG + "0")
        missing_reference = copy.deepcopy(good)
        missing_reference["description"] = "Unrelated release"
        for item in (wrong_version, wrong_tag, missing_reference):
            with self.subTest(item=item), self.assertRaises(ValueError):
                release_notes([item], TAG)
        for fixed in ("0.9.0", "v0.9.0", TAG, ">= " + TAG):
            good["vulnerabilities"][0]["patched_versions"] = fixed
            self.assertIn("Example fix", release_notes([good], TAG))

    def test_repeated_generation_preserves_surrounding_notes(self):
        original = "# Features\n\nExisting text.\n"
        once = release_notes([advisory()], TAG, original)
        twice = release_notes([advisory()], TAG, once)
        self.assertEqual(once, twice)
        self.assertTrue(twice.startswith(original))
        with_suffix = once + "\n## Contributors\nKeep this too.\n"
        updated = advisory()
        updated["summary"] = "Updated fix"
        result = release_notes([updated], TAG, with_suffix)
        self.assertTrue(result.endswith("## Contributors\nKeep this too.\n"))
        self.assertNotIn("Example fix", result)
        self.assertEqual(result.count(START), 1)

    def test_multiple_packages_require_a_release_reference_for_the_patched_row(self):
        item = advisory()
        platform = {"ecosystem": "other", "name": "autogpt-platform"}
        classic = {"ecosystem": "other", "name": "autogpt-classic"}
        item["vulnerabilities"] = [
            {"package": platform, "patched_versions": "0.8.0"},
            {"package": classic, "patched_versions": "0.9.0"},
        ]
        with self.assertRaisesRegex(ValueError, "package-specific release reference"):
            release_notes([item], TAG)

        reference = json.dumps({"tag": TAG, "package": platform})
        item["description"] += f"\n<!-- security-release: {reference} -->"
        with self.assertRaisesRegex(ValueError, "No published advisories match"):
            release_notes([item], TAG)

        item["vulnerabilities"][0]["patched_versions"] = "0.9.0"
        item["vulnerabilities"][1]["patched_versions"] = None
        self.assertIn("Example fix", release_notes([item], TAG))

        item["description"] = advisory()["description"]
        reference = json.dumps({"tag": TAG + "0", "package": platform})
        item["description"] += f"\n<!-- security-release: {reference} -->"
        with self.assertRaisesRegex(ValueError, "package-specific release reference"):
            release_notes([item], TAG)

    def test_multiple_package_references_are_validated(self):
        item = advisory()
        package = {"ecosystem": "other", "name": "significantgravitas/autogpt"}
        item["vulnerabilities"] = [
            {"package": package, "patched_versions": "0.9.0"},
            {"package": {"ecosystem": "other", "name": "autogpt-platform"}},
        ]
        for invalid in ("not-json", "null", "[]", '{"tag": "x"}'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                item["description"] = advisory()["description"]
                item["description"] += f"\n<!-- security-release: {invalid} -->"
                release_notes([item], TAG)

        reference = json.dumps({"tag": TAG, "package": package})
        item["description"] = advisory()["description"]
        item["description"] += f"\n<!-- security-release: {reference} -->"
        self.assertIn("Example fix", release_notes([item], TAG))

    def test_cli_preserves_output_when_package_attribution_is_ambiguous(self):
        item = advisory()
        item["vulnerabilities"] *= 2
        with TemporaryDirectory() as directory:
            notes = Path(directory) / "notes.md"
            output = Path(directory) / "candidate.md"
            notes.write_text("# Existing notes\n", encoding="utf-8")
            output.write_text("Previous candidate\n", encoding="utf-8")
            argv = [
                "security_release_notes.py",
                "--tag",
                TAG,
                "--notes",
                str(notes),
                "--output",
                str(output),
            ]
            with patch("sys.argv", argv), patch(
                "security_release_notes.fetch_advisories",
                return_value=[advisory(), item],
            ), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "Previous candidate\n")
            with patch("sys.argv", argv), patch(
                "security_release_notes.fetch_advisories", return_value=[advisory()]
            ):
                main()
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                release_notes([advisory()], TAG, "# Existing notes\n"),
            )

    def test_release_links_allow_prose_punctuation_but_reject_tag_extensions(self):
        item = advisory()
        url = f"{BASE_URL}/releases/tag/{TAG}"
        for suffix in (
            ".",
            ", next",
            "; next",
            ": next",
            "!",
            "]",
            '"',
            "'",
            "`",
            '."',
            "?download=1",
            "#notes",
        ):
            with self.subTest(suffix=suffix):
                item["description"] = url + suffix
                self.assertIn("Example fix", release_notes([item], TAG))
        for suffix in ("0", ".1", ".other", "-rc1", "/different", ",next"):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                item["description"] = url + suffix
                release_notes([item], TAG)

    def test_malformed_markers_fail_without_overwriting_notes(self):
        for notes in (START, END, END + START, START + START + END):
            with self.subTest(notes=notes), self.assertRaises(ValueError):
                release_notes([advisory()], TAG, notes)

    def test_titles_are_text_not_arbitrary_markdown(self):
        item = advisory()
        item["summary"] = "[click](https://example.invalid)\n<script>"
        result = release_notes([item], TAG)
        self.assertNotIn("[click](", result)
        self.assertNotIn("\n<script>", result)


if __name__ == "__main__":
    unittest.main()
