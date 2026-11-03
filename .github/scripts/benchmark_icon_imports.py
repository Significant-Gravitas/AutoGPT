import json
import os
import sys
from pathlib import Path

from benchmark_frontend import FRONTEND, Benchmark
from benchmark_frontend_builds import build_report, reset_output


if os.environ.get("GITHUB_ACTIONS") != "true" or sys.platform != "linux":
    raise SystemExit("This diagnostic runs only in hosted Linux CI")

benchmark = Benchmark(Path(os.environ["RUNNER_TEMP"]) / "icon-import-report", mode="icon-imports")
configuration = FRONTEND / "next.config.mjs"
original = configuration.read_bytes()
marker = b"const nextConfig = {\n"
assert original.count(marker) == 1
first = os.environ["IMPORTS_FIRST"]
assert first in ("baseline", "direct")
order = [first, "direct" if first == "baseline" else "baseline"]
routes = []
try:
    mapping_path = benchmark.output / "icon-transform.json"
    benchmark.run("verify-icon-exports", ["node", "../../.github/scripts/verify_icon_exports.mjs", str(mapping_path)], required=True)
    transform = json.loads(mapping_path.read_text())
    config = {"@hugeicons/core-free-icons": {"transform": transform, "preventFullImport": True}}
    candidate = original.replace(marker, marker + ("  modularizeImports: " + json.dumps(config) + ",\n").encode())
    for name in order:
        configuration.write_bytes(candidate if name == "direct" else original)
        reset_output(FRONTEND, keep_cache=False)
        benchmark.run(name, ["pnpm", "exec", "next", "build", "--turbopack"], required=True)
        report = build_report(FRONTEND)
        benchmark.records[-1].update(report)
        routes.append(report["routes"])
    benchmark.canaries["both-builds-produced-standalone-and-sourcemaps"] = len(routes) == 2
    benchmark.canaries["same-built-route-inventory"] = len(routes) == 2 and routes[0] == routes[1]
finally:
    configuration.write_bytes(original)
    benchmark.finish()
    path = benchmark.output / "measurements.json"
    report = json.loads(path.read_text())
    report["first"] = first
    report["limitations"] = [
        "Both builds use Turbopack and a deleted compiler cache on the same runner.",
        "The matrix reverses baseline and direct-import ordering.",
        "Only the Hugeicons import transform changes between commands.",
        "All used icon values must equal their root-package exports.",
        "Route and output checks are structural, not runtime validation.",
        "Command timings exclude setup, Docker export, tests and scheduling.",
    ]
    path.write_text(json.dumps(report, indent=2) + "\n")
if not routes or not all(benchmark.canaries.values()):
    raise SystemExit("Icon import comparison failed validation")
