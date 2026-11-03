import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from backend_test_shard import report_shards


def weights_from_reports(paths: list[Path]) -> dict:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not reports or any(report["exit_code"] != 0 for report in reports):
        raise ValueError("Durations require successful shard reports")
    sources = {report["source_sha"] for report in reports}
    layouts = {report["layout"] for report in reports}
    if len(sources) != 1 or None in sources or len(layouts) != 1:
        raise ValueError("Use one source revision and one layout per calibration")
    observations = defaultdict(list)
    versions = defaultdict(set)
    partitioned = any(report.get("partition", 0) for report in reports)
    for report in reports:
        version = ".".join(report["python"].split(".")[:2])
        shard = (
            f"{report['shard']}-{report.get('partition', 0)}"
            if partitioned
            else report["shard"]
        )
        if shard in versions[version]:
            raise ValueError(f"Duplicate shard for Python {version}")
        versions[version].add(shard)
        for name, values in report["files"].items():
            if values["cases"]:
                observations[name].append(
                    values["collection_seconds"] + values["test_seconds"]
                )
    if any(shards != set(report_shards(partitioned)) for shards in versions.values()):
        raise ValueError("Calibration requires every shard for each Python version")
    return {
        "version": 1,
        "source_sha": sources.pop(),
        "measurement": "Median per-file collection, setup, call, and teardown seconds",
        "seconds": {
            name: round(statistics.median(values), 3)
            for name, values in sorted(observations.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("backend-test-durations.json"),
    )
    args = parser.parse_args()
    weights = weights_from_reports(
        sorted(args.reports_dir.rglob("shard-metrics-*.json"))
    )
    args.output.write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
