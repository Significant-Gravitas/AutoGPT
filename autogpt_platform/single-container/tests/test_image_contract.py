from __future__ import annotations

import unittest
from pathlib import Path


SINGLE_CONTAINER_DIR = Path(__file__).resolve().parents[1]


class ImageContractTest(unittest.TestCase):
    def test_falkordb_is_required_and_uses_upstream_module_defaults(self) -> None:
        dockerfile = (
            SINGLE_CONTAINER_DIR.parent / "backend" / "Dockerfile"
        ).read_text()
        entrypoint = (SINGLE_CONTAINER_DIR / "entrypoint.sh").read_text()
        compose = (
            SINGLE_CONTAINER_DIR.parent / "docker-compose.single-container.yml"
        ).read_text()

        self.assertIn(
            "loadmodule /opt/falkordb/falkordb.so MAX_QUEUED_QUERIES 25 "
            "TIMEOUT 1000 RESULTSET_SIZE 10000",
            entrypoint,
        )
        self.assertIn("FORCE_FLAG_GRAPHITI_MEMORY=true", dockerfile)
        self.assertNotIn("AUTOGPT_ENABLE_FALKORDB", entrypoint)
        self.assertNotIn("AUTOGPT_ENABLE_FALKORDB", compose)

    def test_unsupported_email_verification_fails_closed(self) -> None:
        entrypoint = (SINGLE_CONTAINER_DIR / "entrypoint.sh").read_text()
        compose = (
            SINGLE_CONTAINER_DIR.parent / "docker-compose.single-container.yml"
        ).read_text()

        self.assertIn(
            "email verification is not supported by the single-container distribution",
            entrypoint,
        )
        self.assertNotIn("AUTH_REQUIRE_EMAIL_VERIFICATION:", compose)

    def test_runtime_google_picker_is_explicitly_unsupported(self) -> None:
        compose = (
            SINGLE_CONTAINER_DIR.parent / "docker-compose.single-container.yml"
        ).read_text()
        example = (SINGLE_CONTAINER_DIR / ".env.example").read_text()

        self.assertNotIn("GOOGLE_API_KEY", compose)
        self.assertNotIn("GOOGLE_APP_ID", compose)
        self.assertIn("does not support configuring Google Picker at runtime", example)


if __name__ == "__main__":
    unittest.main()
