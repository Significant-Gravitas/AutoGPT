import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from check_frontend_server_errors import main, server_errors

STARTUP = [
    "   ▲ Next.js 15.5.28",
    "   - Local:        http://localhost:3000",
    "",
    " ✓ Starting...",
    " ✓ Ready in 421ms",
    ' ⚠ The "images.domains" configuration is deprecated.',
    "Request failed on server: POST /api/auth/user 401: This account is closed.",
]

MODULE_ERROR = [
    " ⨯ ReferenceError: Cannot access 's' before initialization",
    "    at Object.I (.next/server/chunks/3944.js:7:193478)",
    "    at c.a (.next/server/webpack-runtime.js:1:721) {",
    "  digest: '1841523647'",
    "}",
]

IMAGE_ERROR = [
    (
        " ⨯ The requested resource isn't a valid image for /integrations/codex.png"
        " received null"
    ),
]


def run(lines: list[str]) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "frontend.log"
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = StringIO()
        with redirect_stdout(out):
            code = main([str(log)])
    return code, out.getvalue()


class CheckFrontendServerErrorsTests(unittest.TestCase):
    def test_a_clean_log_passes(self):
        code, out = run(STARTUP)
        self.assertEqual(code, 0)
        self.assertIn("no Next.js server errors", out)

    def test_each_server_error_fails_the_run(self):
        for error in (MODULE_ERROR, IMAGE_ERROR):
            with self.subTest(error=error[0]):
                code, out = run(STARTUP + error)
                self.assertEqual(code, 1)
                self.assertIn("::error title=Next.js server error (1x)::", out)

    def test_repeats_are_counted_once_with_their_stack(self):
        code, out = run(STARTUP + MODULE_ERROR * 3 + IMAGE_ERROR)
        self.assertEqual(code, 1)
        self.assertIn(
            "::error title=Next.js server error (3x)::ReferenceError: Cannot "
            "access 's' before initialization",
            out,
        )
        self.assertEqual(out.count("at Object.I"), 1)
        self.assertIn("2 distinct", out)

    def test_compose_prefixes_and_colours_are_read_through(self):
        errors = server_errors(
            [
                "frontend-1  |  \x1b[31m⨯\x1b[39m TypeError: boom",
                "frontend-1  |     at handler (.next/server/app/page.js:1:2)",
            ]
        )
        self.assertEqual(
            errors,
            [("TypeError: boom", ["    at handler (.next/server/app/page.js:1:2)"])],
        )

    def test_a_stack_ends_at_an_unrelated_log_entry(self):
        errors = server_errors(
            [
                " ⨯ image failed",
                "Error: authentication failed",
                "    at authenticate (auth.js:1:2)",
                " ⨯ TypeError: boom",
                "    at handler (page.js:1:2)",
            ]
        )
        self.assertEqual(
            errors,
            [
                ("image failed", []),
                ("TypeError: boom", ["    at handler (page.js:1:2)"]),
            ],
        )

    def test_a_wrapped_errors_cause_stays_in_its_stack(self):
        errors = server_errors(
            [
                " ⨯ Error: request failed",
                "    at fetchUser (page.js:1:2) {",
                "  [cause]: TypeError: fetch failed",
                "}",
            ]
        )
        self.assertEqual(
            errors[0][1],
            ["    at fetchUser (page.js:1:2) {", "  [cause]: TypeError: fetch failed"],
        )

    def test_an_empty_log_fails(self):
        for lines in ([], ["", "   "]):
            with self.subTest(lines=lines):
                code, out = run(lines)
                self.assertEqual(code, 1)
                self.assertIn("the log is empty", out)

    def test_an_external_image_timing_out_is_a_warning_not_a_failure(self):
        timeout = (
            " ⨯ upstream image response timed out for https://storage.googleapis"
            ".com/agpt-prod-public-site-media/users/u/images/i.png"
        )
        code, out = run(STARTUP + [timeout, timeout])
        self.assertEqual(code, 0)
        self.assertIn("::warning title=External image timed out (2x)::", out)
        self.assertIn("no Next.js server errors", out)

    def test_a_dns_name_with_a_numeric_first_label_is_external(self):
        for host in ("169.254.images.example.com", "10.cdn.example.com"):
            with self.subTest(host=host):
                line = f" ⨯ upstream image response timed out for https://{host}/a.png"
                code, out = run(STARTUP + [line])
                self.assertEqual(code, 0)
                self.assertIn("::warning title=External image timed out (1x)::", out)

    def test_a_local_timeout_or_a_failed_fetch_still_fails(self):
        for line in (
            " ⨯ upstream image response timed out for http://localhost:3000/a.png",
            " ⨯ upstream image response timed out for http://rest_server:8006/a.png",
            " ⨯ upstream image response timed out for http://172.18.0.4/a.png",
            " ⨯ upstream image response timed out for http://0.0.0.0:3000/a.png",
            " ⨯ upstream image response timed out for http://169.254.1.1/a.png",
            (
                " ⨯ upstream image response timed out for "
                "http://host.docker.internal:8006/a.png"
            ),
            " ⨯ upstream image response timed out for http://[::1]:3000/a.png",
            " ⨯ upstream image response failed for https://example.com/a.png 404",
        ):
            with self.subTest(line=line):
                code, out = run(STARTUP + [line])
                self.assertEqual(code, 1)
                self.assertIn("::error title=Next.js server error (1x)::", out)

    def test_an_external_timeout_doesnt_hide_a_real_error(self):
        timeout = (
            " ⨯ upstream image response timed out for https://images.unsplash"
            ".com/photo.jpg"
        )
        code, out = run(STARTUP + [timeout] + MODULE_ERROR)
        self.assertEqual(code, 1)
        self.assertIn("::warning title=External image timed out (1x)::", out)
        self.assertIn("ReferenceError: Cannot access 's' before initialization", out)
        self.assertIn("logged 1 error(s)", out)

    def test_a_missing_log_fails_naming_only_the_file(self):
        out = StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["/nonexistent/private/frontend.log"]), 1)
        self.assertIn("frontend.log: could not read the log", out.getvalue())
        self.assertNotIn("/nonexistent/private", out.getvalue())


if __name__ == "__main__":
    unittest.main()
