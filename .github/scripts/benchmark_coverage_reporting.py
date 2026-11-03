import difflib
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from xml.etree import ElementTree

root = pathlib.Path(os.environ["RUNNER_TEMP"]) / "coverage-comparison"
root.mkdir()
data_file = pathlib.Path(os.environ["RUNNER_TEMP"]) / "coverage-input/.coverage-util-executor-3"
labels = ["baseline", "candidate"] if sys.argv[1] == "baseline-first" else ["candidate", "baseline"]
results = {}


def rss_tree(pid):
    try:
        status = pathlib.Path(f"/proc/{pid}/status").read_text()
        own = int(next(line for line in status.splitlines() if line.startswith("VmRSS:")).split()[1])
        children = pathlib.Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        return own + sum(rss_tree(child) for child in children)
    except (OSError, StopIteration):
        return 0


for label in labels:
    script = "report_backend_coverage.py" if label == "baseline" else "parallel_coverage_candidate.py"
    started = time.perf_counter()
    peak = 0
    with (root / f"{label}.log").open("w") as log:
        process = subprocess.Popen([sys.executable, f"../../.github/scripts/{script}", "--data-file", str(data_file), "--output", str(root / f"{label}.xml")], stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            peak = max(peak, rss_tree(process.pid))
            time.sleep(0.1)
    results[label] = {"seconds": time.perf_counter() - started, "peak_process_rss_kb": peak, "exit_code": process.returncode}
    if process.returncode:
        print((root / f"{label}.log").read_text())
        raise SystemExit(process.returncode)
canonical = {label: ElementTree.canonicalize(from_file=root / f"{label}.xml", strip_text=True, exclude_attrs={"timestamp"}) for label in labels}
parity = canonical["baseline"] == canonical["candidate"]
for label in labels:
    results[label]["canonical_sha256"] = hashlib.sha256(canonical[label].encode()).hexdigest()
results["xml_parity"] = parity
(root / "comparison.json").write_text(json.dumps(results, indent=2))
print("::notice title=coverage-report-comparison::" + json.dumps(results))
if not parity:
    changes = list(difflib.unified_diff(canonical["baseline"].replace("><", ">\n<").splitlines(), canonical["candidate"].replace("><", ">\n<").splitlines()))
    (root / "xml-diff.txt").write_text("\n".join(changes[:200]))
    print("::notice title=coverage-xml-diff::" + "\n".join(changes[:25]).replace("%", "%25").replace("\n", "%0A"))
    raise SystemExit("Parallel coverage XML differs from the baseline")
