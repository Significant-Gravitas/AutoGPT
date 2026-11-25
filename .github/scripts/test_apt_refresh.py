import contextlib
import hashlib
import http.client
import io
import re
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import apt_refresh

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"
NOW = datetime(2026, 11, 23, 19, 1, tzinfo=timezone.utc)
# Files the trixie-security Release file lists, a sample of each kind.
LISTED_FILES = (
    "contrib/binary-amd64/Packages.xz",
    "main/Contents-amd64.gz",
    "main/binary-all/Packages",
    "main/binary-all/Packages.xz",
    "main/binary-all/Release",
    "main/binary-amd64/Packages",
    "main/binary-amd64/Packages.xz",
    "main/binary-amd64/Release",
    "main/binary-arm64/Packages",
    "main/binary-arm64/Packages.xz",
    "main/binary-arm64/Release",
    "main/binary-s390x/Packages.xz",
    "main/debian-installer/binary-amd64/Packages.xz",
    "main/i18n/Translation-en.xz",
    "main/source/Sources.xz",
    "non-free-firmware/binary-amd64/Packages.xz",
)
# The package indexes of the main component, which the images install from.
PACKAGE_INDEXES = {
    path
    for path in LISTED_FILES
    if re.fullmatch(r"main/binary-\w+/Packages(\.xz)?", path)
}


def release(
    *,
    codename: str = "trixie-security",
    signed: str = "Mon, 23 Nov 2026 15:45:44 UTC",
    published: frozenset[str] = frozenset(),
) -> str:
    """An InRelease file like the one Debian serves for trixie-security.

    `signed` is the Date it was signed on, and each file in `published` gets a
    new checksum and size, as when Debian publishes an update to it.
    """
    entries = "".join(
        f" {hashlib.sha256(f'{path}{path in published}'.encode()).hexdigest()}"
        f" {len(path) * 1000 + (path in published):>8} {path}\n"
        for path in LISTED_FILES
    )
    return (
        "-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA256\n\n"
        "Origin: Debian\nLabel: Debian-Security\nSuite: stable-security\n"
        f"Version: 13\nCodename: {codename}\nDate: {signed}\n"
        f"Valid-Until: {signed.replace('23 Nov', '30 Nov')}\nAcquire-By-Hash: yes\n"
        "Architectures: amd64 arm64 armel armhf i386 ppc64el riscv64 s390x\n"
        "Components: updates/main updates/contrib updates/non-free-firmware"
        " updates/non-free\nDescription: Debian 13 - Security Updates\n"
        f"SHA256:\n{entries}"
        "-----BEGIN PGP SIGNATURE-----\n\n"
        f"{hashlib.sha256(signed.encode()).hexdigest()}\n"
        "-----END PGP SIGNATURE-----\n"
    )


class RefreshKeyTests(unittest.TestCase):
    def key(self, *reads: str | Exception, now: datetime = NOW) -> str:
        """The key when each read of the index returns or raises in turn."""
        answers = iter(reads)

        def fetch(url: str) -> str:
            self.assertEqual(url, apt_refresh.INDEX_URL)
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return answer

        self.sleeps: list[float] = []
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            key = apt_refresh.refresh_key(
                fetch=fetch, now=lambda: now, sleep=self.sleeps.append
            )
        # The workflows read the key from stdout, so nothing else goes there.
        self.assertEqual(stdout.getvalue(), "")
        self.log = stderr.getvalue()
        return key

    def test_the_key_is_the_utc_date_and_a_hash_of_the_package_indexes(self):
        self.assertRegex(self.key(release()), r"^2026-11-23\+[0-9a-f]{12}$")
        self.assertEqual(self.sleeps, [])
        self.assertIn("Mon, 23 Nov 2026 15:45:44 UTC", self.log)

    def test_the_date_is_the_utc_one(self):
        in_utc_plus_5 = datetime(
            2026, 11, 24, 1, 30, tzinfo=timezone(timedelta(hours=5))
        )
        self.assertTrue(
            self.key(release(), now=in_utc_plus_5).startswith("2026-11-23+")
        )

    def test_a_security_publication_changes_the_key(self):
        before = self.key(release())
        for path in sorted(PACKAGE_INDEXES):
            with self.subTest(path=path):
                after = self.key(release(published=frozenset({path})))
                self.assertNotEqual(after, before)
                self.assertTrue(after.startswith("2026-11-23+"))

    def test_a_re_signed_file_keeps_the_key(self):
        # Debian re-signs the file, with a new Date, without new packages.
        self.assertEqual(
            self.key(release(signed="Mon, 23 Nov 2026 20:04:24 UTC")),
            self.key(release()),
        )

    def test_files_the_images_do_not_install_from_keep_the_key(self):
        before = self.key(release())
        for path in sorted(set(LISTED_FILES) - PACKAGE_INDEXES):
            with self.subTest(path=path):
                self.assertEqual(self.key(release(published=frozenset({path}))), before)

    def test_a_failed_read_is_retried_then_the_date_alone_is_used(self):
        failures = (
            urllib.error.URLError("[Errno -3] Temporary failure in name resolution"),
            TimeoutError("The read operation timed out"),
            http.client.IncompleteRead(b"-----BEGIN PGP"),
        )
        self.assertEqual(self.key(*failures), "2026-11-23")
        self.assertEqual(self.sleeps, [2, 4])
        self.assertRegex(self.log, r"(?m)^::warning title=[^:]+::.*IncompleteRead")

    def test_a_read_that_recovers_uses_the_index(self):
        recovered = self.key(
            urllib.error.HTTPError(apt_refresh.INDEX_URL, 503, "busy", {}, None),
            release(),
        )
        self.assertEqual(self.sleeps, [2])
        self.assertEqual(recovered, self.key(release()))

    def test_a_file_that_is_not_the_suites_release_file_is_refused(self):
        for answer in (
            release(codename="forky-security"),
            "<html><body>503 Service Unavailable</body></html>",
            release().split("SHA256:")[0],
        ):
            with self.subTest(answer=answer[:60]):
                self.assertEqual(self.key(answer, answer, answer), "2026-11-23")
                self.assertIn("::warning ", self.log)


class WorkflowTests(unittest.TestCase):
    def test_every_build_takes_apt_refresh_from_this_script(self):
        # A build that passed the date alone would keep a fix Debian publishes
        # during the day behind that day's cached package layer until midnight.
        step_output, variable = "${{ steps.apt-refresh.outputs.key }}", "$apt_refresh"
        passed = 0
        for workflow in sorted(WORKFLOWS.glob("*.y*ml")):
            steps = re.split(r"\n\s+- name: ", workflow.read_text(encoding="utf-8"))
            for step in steps:
                values = [
                    value.strip('"')
                    for value in re.findall(r"APT_REFRESH=(\$\{\{[^}]*\}\}|\S+)", step)
                ]
                passed += len(values)
                for value in values:
                    with self.subTest(workflow=workflow.name, value=value):
                        self.assertIn(value, {step_output, variable})
                if variable in values:
                    with self.subTest(workflow=workflow.name, step=step.split("\n")[0]):
                        # A shell variable doesn't outlive its step.
                        self.assertRegex(
                            step,
                            r'apt_refresh="\$\(python3 "?[\w$/.]*'
                            r'\.github/scripts/apt_refresh\.py"?\)"',
                        )
        self.assertGreater(passed, 0, "no workflow passes APT_REFRESH")


if __name__ == "__main__":
    unittest.main()
