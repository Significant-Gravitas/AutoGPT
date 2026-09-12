import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[3] / ".github/scripts/single_container_cache.py"
)
SPEC = importlib.util.spec_from_file_location("single_container_cache", SCRIPT)
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)
IMAGE = "ghcr.io/example/appliance-cache"


class RegistryCacheTests(unittest.TestCase):
    def settings(self, event, ref="refs/heads/dev", arch="amd64"):
        return cache.cache_settings(event, ref, IMAGE, arch)

    def test_pull_requests_never_read_or_write_registry_cache(self):
        self.assertEqual(self.settings("pull_request"), [])

    def test_only_dev_push_writes_the_release_cache(self):
        settings = self.settings("push")
        self.assertIn(
            f"single-container.cache-to=type=registry,ref={IMAGE}:v4-amd64,mode=max,oci-mediatypes=true,image-manifest=false",
            settings,
        )
        self.assertFalse(
            any("cache-to" in s for s in self.settings("push", "refs/heads/test"))
        )

    def test_dispatch_cache_is_stable_branch_scoped_and_never_release_cache(self):
        for ref in ("refs/heads/dev", "refs/heads/test"):
            settings = self.settings("workflow_dispatch", ref)
            writes = [s for s in settings if "cache-to" in s]
            self.assertEqual(len(writes), 1)
            self.assertIn(":v4-branch-", writes[0])
            self.assertIn("mode=max", writes[0])
            self.assertIn(
                f"single-container.cache-from=type=registry,ref={IMAGE}:v4-amd64",
                settings,
            )
            self.assertEqual(settings, self.settings("workflow_dispatch", ref))
        self.assertNotEqual(
            self.settings("workflow_dispatch", "refs/heads/one"),
            self.settings("workflow_dispatch", "refs/heads/two"),
        )

    def test_release_only_reads_trusted_cache_and_architectures_are_separate(self):
        settings = self.settings("release", "refs/tags/v1")
        self.assertTrue(all("cache-from" in s and ":v4-amd64" in s for s in settings))
        self.assertTrue(
            all(":v4-arm64" in s for s in self.settings("release", arch="arm64"))
        )
