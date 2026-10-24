import copy
import unittest

from security_release_notes import BASE_URL, END, START, release_notes

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
        for forbidden in ("SECRET", "@pending", "@declined"):
            self.assertNotIn(forbidden, result)

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
