import argparse
import json
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree

APPLICATION_SHA = "e514ac6ce0b4cc64f017d635065442763122db08"
APPLICATION_TREE = "88971556809482f4d8a7eb17a490895c85216360"


def prepare(reports, manifest, backend):
    expected = {item["name"] for item in json.loads(manifest.read_text())}
    assert len(expected) == 66
    assert {path.name for path in reports.iterdir()} == expected
    actual_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD:autogpt_platform"], text=True
    ).strip()
    assert actual_tree == APPLICATION_TREE
    cases = {"3.12": set(), "3.13": set()}
    for name in sorted(expected):
        report = reports / name
        metrics_paths = list(report.glob("shard-metrics-*.json"))
        assert len(metrics_paths) == 1
        metrics = json.loads(metrics_paths[0].read_text())
        assert metrics["source_sha"] == APPLICATION_SHA
        version = ".".join(metrics["python"].split(".")[:2])
        assert name.startswith(f"backend-test-reports-py{version}-")
        junit_paths = list(report.glob("junit-*.xml"))
        assert len(junit_paths) == 1
        junit = ElementTree.parse(junit_paths[0])
        found = 0
        for case in junit.iter("testcase"):
            assert case.find("failure") is None and case.find("error") is None
            identity = case.get("classname"), case.get("name")
            assert identity not in cases[version]
            cases[version].add(identity)
            found += 1
        assert found > 0
        raw_paths = list(report.glob(".coverage-*"))
        assert len(raw_paths) == 1 and raw_paths[0].stat().st_size > 0
        shutil.copyfile(raw_paths[0], backend / f".coverage-diagnostic.{name}")
    counts = {version: len(values) for version, values in cases.items()}
    assert counts == {"3.12": 43062, "3.13": 43062}, counts
    print(json.dumps({"reports": len(expected), "cases": counts}))


def verify(xml_path):
    root = ElementTree.parse(xml_path)
    sources = [node.text for node in root.findall("sources/source")]
    assert len(sources) == 1 and sources[0].endswith(
        "/autogpt_platform/backend/backend"
    )
    files = set()
    counts = {"valid": 0, "fully_hit": 0, "partial": 0, "missing": 0}
    for item in root.findall("packages/package/classes/class"):
        filename = item.attrib["filename"]
        assert filename not in files
        files.add(filename)
        for line in item.findall("lines/line"):
            counts["valid"] += 1
            if int(line.attrib["hits"]) == 0:
                counts["missing"] += 1
            elif line.get("branch") == "true" and not line.attrib[
                "condition-coverage"
            ].startswith("100%"):
                counts["partial"] += 1
            else:
                counts["fully_hit"] += 1
    result = {"files": len(files), **counts, "source_sha": APPLICATION_SHA}
    print(json.dumps(result, indent=2))
    xml_path.with_suffix(".summary.json").write_text(json.dumps(result, indent=2))
    assert len(files) == 3736
    assert counts == {
        "valid": 467895,
        "fully_hit": 438778,
        "partial": 5809,
        "missing": 23308,
    }, counts
    print("Combined raw branch data exactly matches the independent 66-report union.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--backend", type=Path)
    parser.add_argument("--xml", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if not all((args.reports, args.manifest, args.backend)):
            parser.error("prepare requires reports, manifest and backend")
        prepare(args.reports, args.manifest, args.backend)
    else:
        if not args.xml:
            parser.error("verify requires xml")
        verify(args.xml)
