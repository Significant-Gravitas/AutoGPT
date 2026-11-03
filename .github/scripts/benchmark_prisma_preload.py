import argparse
import json
import os
import pathlib
import subprocess
import time
from xml.etree import ElementTree

from coverage import CoverageData

parser = argparse.ArgumentParser()
parser.add_argument("--order", required=True)
args = parser.parse_args()
output = pathlib.Path(os.environ["RUNNER_TEMP"]) / "prisma-preload"
output.mkdir()
labels = ["baseline", "candidate"] if args.order == "baseline-first" else ["candidate", "baseline"]
results = {}
for label in labels:
    report = output / f"junit-{label}.xml"
    data_file = output / f".coverage-{label}"
    environment = {**os.environ, "BENCHMARK_PRELOAD_PRISMA": "1" if label == "candidate" else "0", "COVERAGE_FILE": str(data_file), "BACKEND_TEST_METRICS": str(output / f"metrics-{label}.json")}
    started = time.perf_counter()
    with (output / f"{label}.log").open("w") as log:
        result = subprocess.run(["python", "../../.github/scripts/backend_test_shard.py", "--layout", "paths", "--partition", "3", "billing-threads", "--", "-p", "no:cacheprovider", "--cov=backend", "--cov-branch", "--cov-report=", f"--junitxml={report}", "-q"], env=environment, stdout=log, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - started
    if result.returncode:
        print((output / f"{label}.log").read_text())
        raise SystemExit(result.returncode)
    cases = list(ElementTree.parse(report).iter("testcase"))
    if not cases or any(case.find("failure") is not None or case.find("error") is not None for case in cases):
        raise RuntimeError("Missing or failed test cases")
    results[label] = {"seconds": seconds, "cases": sorted((c.get("classname"), c.get("name"), c.find("skipped") is not None) for c in cases)}
if results["baseline"]["cases"] != results["candidate"]["cases"]:
    raise RuntimeError("Preloading changed case identities or skips")
coverage = {}
for label in labels:
    data = CoverageData(basename=str(output / f".coverage-{label}"))
    data.read()
    if not data.has_arcs():
        raise RuntimeError("Missing branch coverage data")
    coverage[label] = {name: set(data.arcs(name) or []) for name in data.measured_files()}
changes = []
for name in sorted(coverage["baseline"].keys() | coverage["candidate"].keys()):
    before = coverage["baseline"].get(name, set())
    after = coverage["candidate"].get(name, set())
    if before != after:
        changes.append({"file": name, "lost_arcs": sorted(before - after), "added_arcs": sorted(after - before)})
(output / "coverage-diff.json").write_text(json.dumps(changes, indent=2))
compact = {label: {"seconds": value["seconds"], "cases": len(value["cases"])} for label, value in results.items()}
compact["coverage_changed_files"] = len(changes)
print("::notice title=prisma-preload-comparison::" + json.dumps(compact))
(output / "comparison.json").write_text(json.dumps(compact, indent=2))
if changes:
    print("::notice title=prisma-preload-coverage-diff::" + json.dumps(changes[:4])[:3500])
    raise SystemExit("Branch coverage parity needs investigation")
