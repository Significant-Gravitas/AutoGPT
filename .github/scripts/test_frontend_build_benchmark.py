import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark_frontend_builds import build_report, compare_builds, reset_output


def write_build(frontend: Path, *, route: str = "/page", maps: bool = True) -> None:
    output = frontend / ".next"
    for directory in ("server", "standalone", "static", "cache"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    (output / "standalone/server.js").write_text("server")
    (output / "server/app-paths-manifest.json").write_text(json.dumps({route: "a.js"}))
    (output / "server/pages-manifest.json").write_text(json.dumps({"/_error": "b.js"}))
    (output / "cache/dependencies").write_text("cached input")
    if maps:
        (output / "static/a.js.map").write_text("{}")


class FrontendBuildBenchmarkTests(unittest.TestCase):
    def test_rebuild_removes_outputs_but_retains_only_requested_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            frontend = Path(directory)
            write_build(frontend)
            reset_output(frontend, keep_cache=True)
            self.assertEqual(
                (frontend / ".next/cache/dependencies").read_text(), "cached input"
            )
            self.assertFalse((frontend / ".next/standalone").exists())
            self.assertFalse((frontend / ".next/server").exists())
            reset_output(frontend, keep_cache=False)
            self.assertEqual(list((frontend / ".next").iterdir()), [])

    def test_successful_command_without_source_maps_is_not_valid_build_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            frontend = Path(directory)
            write_build(frontend, maps=False)
            with self.assertRaisesRegex(ValueError, "source maps"):
                build_report(frontend)

    def test_comparison_rejects_missing_routes_and_restores_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            frontend = Path(directory)
            source = frontend / "src/lib/keyboard.ts"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"export const key = 'Enter';\n")
            original = source.read_bytes()
            (frontend / "next.config.mjs").write_text("export default {}")
            benchmark = SimpleNamespace(output=frontend, records=[], canaries={})
            observations = []

            def run(name, command):
                observations.append(
                    (
                        name,
                        (frontend / ".next/cache/dependencies").exists(),
                        (frontend / ".next/standalone/server.js").exists(),
                        source.read_bytes() != original,
                    )
                )
                benchmark.records.append({"name": name, "exit_code": 0})
                route = "/lost/page" if name == "webpack-edited" else "/page"
                write_build(frontend, route=route)
                return 0

            benchmark.run = run
            compare_builds(benchmark, frontend, "turbopack")
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(
                [item[1:] for item in observations],
                [(False, False, False), (True, False, False), (True, False, True)] * 2,
            )
            self.assertTrue(
                benchmark.canaries["all-six-builds-produced-standalone-and-sourcemaps"]
            )
            self.assertFalse(benchmark.canaries["same-built-route-inventory"])

    def test_interrupted_edited_build_restores_the_source(self):
        with tempfile.TemporaryDirectory() as directory:
            frontend = Path(directory)
            source = frontend / "src/lib/keyboard.ts"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"original\r\n")
            (frontend / "next.config.mjs").write_text("export default {}")
            benchmark = SimpleNamespace(output=frontend, records=[], canaries={})

            def run(name, command):
                if name.endswith("edited"):
                    self.assertIn(b"build-cache probe", source.read_bytes())
                    raise RuntimeError("interrupted build")
                benchmark.records.append({"name": name, "exit_code": 0})
                write_build(frontend)
                return 0

            benchmark.run = run
            with self.assertRaisesRegex(RuntimeError, "interrupted build"):
                compare_builds(benchmark, frontend, "webpack")
            self.assertEqual(source.read_bytes(), b"original\r\n")


if __name__ == "__main__":
    unittest.main()
