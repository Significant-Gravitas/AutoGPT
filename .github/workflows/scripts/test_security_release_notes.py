import copy
import json
import unittest

import security_public_sync as sync
from security_public_api import PublicDataError, PublicErrorCode
from security_release_notes import BASE_URL, END, START

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


def release_tags(item):
    return sync.public_snapshot([item])["advisories"][0]["release_tags"]


def notes(items, existing=""):
    return sync.public_release_notes(sync.public_snapshot(items), TAG, existing)


class ReleaseAttributionTest(unittest.TestCase):
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
        excluded = [dict(advisory(), state=s) for s in ("triage", "draft", "closed")]
        excluded += [
            dict(advisory(), published_at=None),
            dict(advisory(), withdrawn_at="2026-10-24"),
        ]
        snapshot = sync.public_snapshot([public, *excluded])
        self.assertEqual(len(snapshot["advisories"]), 1)
        result = sync.public_release_notes(snapshot, TAG, "")
        self.assertIn("@researcher", result)
        for forbidden in ("@pending", "@declined", "@maintainer", "Example fix"):
            self.assertNotIn(forbidden, result)

    def test_deleted_credit_users_are_skipped_and_bad_ids_fail_closed(self):
        valid = advisory()
        valid["credits_detailed"] += [
            {"user": None, "state": "accepted", "type": "reporter"},
            {"state": "accepted", "type": "reporter"},
        ]
        self.assertEqual(
            sync.public_snapshot([valid])["advisories"][0]["researchers"],
            ["researcher"],
        )
        missing = advisory()
        del missing["ghsa_id"]
        for broken in (missing, advisory(None), advisory("GHSA-not-valid")):
            with self.subTest(ghsa=broken.get("ghsa_id")), self.assertRaises(
                PublicDataError
            ) as raised:
                sync.public_snapshot([valid, broken])
            self.assertIs(raised.exception.code, PublicErrorCode.ADVISORY_ID)

    def test_only_valid_cve_identifiers_are_rendered(self):
        item = advisory()
        item["cve_id"] = "CVE-2026-12345"
        self.assertIn("(CVE-2026-12345)", notes([item]))
        for cve in (None, "CVE-2026-12", "CVE-2026-12345<script>"):
            with self.subTest(cve=cve):
                item["cve_id"] = cve
                self.assertNotIn("CVE-", notes([item]))

    def test_requires_matching_fixed_version_and_exact_release_reference(self):
        good = advisory()
        wrong_version = copy.deepcopy(good)
        wrong_version["vulnerabilities"][0]["patched_versions"] = "0.8.0"
        wrong_tag = copy.deepcopy(good)
        wrong_tag["description"] = wrong_tag["description"].replace(TAG, TAG + "0")
        missing_reference = copy.deepcopy(good)
        missing_reference["description"] = "Unrelated release"
        for item in (wrong_version, wrong_tag, missing_reference):
            with self.subTest(description=item["description"]):
                self.assertEqual(release_tags(item), [])
        for fixed in ("0.9.0", "v0.9.0", TAG, ">= " + TAG):
            good["vulnerabilities"][0]["patched_versions"] = fixed
            self.assertEqual(release_tags(good), [TAG])

    def test_repeated_generation_preserves_surrounding_notes(self):
        original = "# Features\n\nExisting text.\n"
        once = notes([advisory()], original)
        self.assertTrue(once.startswith(original))
        self.assertEqual(notes([advisory()], once), once)
        with_suffix = once + "\n## Contributors\nKeep this too.\n"
        updated = advisory()
        updated["cve_id"] = "CVE-2026-12345"
        result = notes([updated], with_suffix)
        self.assertTrue(result.startswith(original))
        self.assertTrue(result.endswith("## Contributors\nKeep this too.\n"))
        self.assertIn("(CVE-2026-12345)", result)
        self.assertEqual(result.count(START), 1)

    def test_rows_fixed_in_different_versions_need_a_mapping_for_the_patched_row(self):
        item = advisory()
        platform = {"ecosystem": "other", "name": "autogpt-platform"}
        classic = {"ecosystem": "other", "name": "autogpt-classic"}
        item["vulnerabilities"] = [
            {"package": platform, "patched_versions": "0.8.0"},
            {"package": classic, "patched_versions": "0.9.0"},
        ]
        with self.assertRaises(PublicDataError) as raised:
            release_tags(item)
        self.assertIs(raised.exception.code, PublicErrorCode.RELEASE_MAPPING)

        # The mapped row was fixed in a different release, so this one isn't credited.
        reference = json.dumps({"tag": TAG, "package": platform})
        item["description"] += f"\n<!-- security-release: {reference} -->"
        self.assertEqual(release_tags(item), [])

        item["vulnerabilities"][0]["patched_versions"] = "0.9.0"
        item["vulnerabilities"][1]["patched_versions"] = "0.8.0"
        self.assertEqual(release_tags(item), [TAG])

        item["description"] = advisory()["description"]
        reference = json.dumps({"tag": TAG + "0", "package": platform})
        item["description"] += f"\n<!-- security-release: {reference} -->"
        with self.assertRaises(PublicDataError):
            release_tags(item)

    def test_package_references_are_validated(self):
        item = advisory()
        package = {"ecosystem": "other", "name": "significantgravitas/autogpt"}
        item["vulnerabilities"] = [
            {"package": package, "patched_versions": "0.9.0"},
            {
                "package": {"ecosystem": "other", "name": "autogpt-platform"},
                "patched_versions": "0.8.0",
            },
        ]
        for invalid in ("not-json", "null", "[]", '{"tag": "x"}'):
            item["description"] = advisory()["description"]
            item["description"] += f"\n<!-- security-release: {invalid} -->"
            with self.subTest(invalid=invalid), self.assertRaises(PublicDataError):
                release_tags(item)
        reference = json.dumps({"tag": TAG, "package": package})
        item["description"] = advisory()["description"]
        item["description"] += f"\n<!-- security-release: {reference} -->"
        self.assertEqual(release_tags(item), [TAG])

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
                self.assertEqual(release_tags(item), [TAG])
        for suffix in ("0", ".1", ".other", "-rc1", "/different", ",next"):
            with self.subTest(suffix=suffix):
                item["description"] = url + suffix
                self.assertEqual(release_tags(item), [])

    def test_malformed_markers_fail_without_overwriting_notes(self):
        snapshot = sync.public_snapshot([advisory()])
        for existing in (START, END, END + START, START + START + END):
            with self.subTest(existing=existing), self.assertRaises(PublicDataError):
                sync.public_release_notes(snapshot, TAG, existing)


if __name__ == "__main__":
    unittest.main()
