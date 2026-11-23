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

    def test_a_missing_log_fails(self):
        out = StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["/nonexistent/frontend.log"]), 1)


if __name__ == "__main__":
    unittest.main()
