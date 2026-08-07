from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


SINGLE_CONTAINER_DIR = Path(__file__).resolve().parents[1]
LICENSE_DIR = SINGLE_CONTAINER_DIR / "licenses"
DOCKERFILE = SINGLE_CONTAINER_DIR.parent / "backend" / "Dockerfile"

EXPECTED_HASHES = {
    "FalkorDB-SSPL-1.0.txt": "34e94c5087ba6e9fb34f35ae71df5e6533c5fc7cbbf6c44186a71e82806b69e1",
    "Redis-LICENSE.txt": "4a0e416b9537688f30dfe69ddaceb2ca64d96b7df02a0a6760d376890ddc4e40",
    "Redis-REDISCONTRIBUTIONS.txt": "aa6a56234e5ca27f09010693d1c31ca83988d7a0dc80fb253603e052d1d7f0d1",
    "redis/deps/fpconv/LICENSE.txt": "c9bff75738922193e67fa726fa225535870d2aa1059f91452c411736284ad566",
    "redis/deps/hdr_histogram/COPYING.txt": "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499",
    "redis/deps/hdr_histogram/LICENSE.txt": "c124afa369aae960fa33f6944c82e161e482f2e998d6f07b37ae2c018f3c6c69",
    "redis/deps/hiredis/COPYING": "dca05ce8fc87a8261783b4aed0deef8becc9350b6aa770bc714d0c1833b896eb",
    "redis/deps/jemalloc/COPYING": "94aa2caa98c25d942f58b956c71dba6a99ff98fc3a31cbc669fe2a4cd0268b53",
    "redis/deps/lua/COPYRIGHT": "ee5e3e82af1e1b543c4f216e399d7c8cfee797711913f349e385101c4ae60a79",
    "redis/deps/xxhash/LICENSE": "6ffedbc0f7878612d2b23589f1ff2ab15633e1df7963a5d9fc750ec5500c7e7a",
    "redis/deps/fast_float/LICENSE-MIT": "e562f3f974ced7e69dd1db77b820b36bcf8f30377f1aa105723fba449c53c4e6",
    "redis/src/lzf_c.c": "9355e97eff48e649ebb290f3dc76866a00ae9ab42855fff8e892df3e487e58a8",
    "redis/src/lzf_d.c": "5bc74a3cb8a2cf60e174a2e67dc32cc092b1c5dc4474f7bf88f81d30b0cfc194",
}


class LicenseAssetsTest(unittest.TestCase):
    def test_vendored_license_assets_match_pinned_upstream_bytes(self) -> None:
        for relative_path, expected_hash in EXPECTED_HASHES.items():
            with self.subTest(path=relative_path):
                contents = (LICENSE_DIR / relative_path).read_bytes()
                self.assertEqual(hashlib.sha256(contents).hexdigest(), expected_hash)

    def test_falkordb_license_is_copied_from_local_build_context(self) -> None:
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")

        self.assertIn("FalkorDB-SSPL-1.0.txt", dockerfile)
        self.assertIn("Redis-LICENSE.txt", dockerfile)
        self.assertNotIn("raw.githubusercontent.com/FalkorDB", dockerfile)

    def test_source_notices_pin_falkordb_and_redis_commits(self) -> None:
        falkordb_notice = (LICENSE_DIR / "FalkorDB-SOURCE.txt").read_text()
        redis_notice = (LICENSE_DIR / "Redis-SOURCE.txt").read_text()

        self.assertIn("efea8d99f8492f3eeaadcafeae29967c66d825a0", falkordb_notice)
        self.assertIn("bd3b38d41070b478c58bc8b72d2af89cbccd1a40", redis_notice)


if __name__ == "__main__":
    unittest.main()
