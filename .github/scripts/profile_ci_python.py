import cProfile
import json
import os
import pathlib
import pstats
import runpy
import sys
import time

root = pathlib.Path(os.environ["RUNNER_TEMP"]) / "python-profile"
root.mkdir(exist_ok=True)
label = sys.argv[1]
profiler = cProfile.Profile()
started = time.perf_counter()
profiler.enable()
if label == "prisma":
    import prisma
else:
    sys.argv = ["report_backend_coverage.py", "--data-file", str(root / ".coverage-util-executor-3"), "--output", str(root / "coverage.xml")]
    runpy.run_path("../../.github/scripts/report_backend_coverage.py", run_name="__main__")
profiler.disable()
elapsed = time.perf_counter() - started
profiler.dump_stats(str(root / f"{label}.prof"))
records = []
for (filename, line, function), (primitive, calls, own, cumulative, callers) in pstats.Stats(profiler).stats.items():
    records.append({"file": pathlib.Path(filename).name, "line": line, "function": function[:80], "calls": calls, "own": round(own, 3), "cumulative": round(cumulative, 3)})
for key in ("own", "cumulative"):
    data = {"elapsed_seconds": elapsed, "python": sys.version.split()[0], "top": sorted(records, key=lambda row: row[key], reverse=True)[:12]}
    (root / f"{label}-{key}.json").write_text(json.dumps(data, indent=2))
    value = json.dumps(data, separators=(",", ":")).replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D")
    print(f"::notice title={label}-{key}::{value}")
