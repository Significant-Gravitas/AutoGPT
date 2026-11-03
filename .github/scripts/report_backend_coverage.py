#!/usr/bin/env python3

import argparse
import copy
import json
import os
import re
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from xml.etree import ElementTree

from coverage import Coverage


def report_part(data_file: str, paths: list[str], output: str) -> str:
    coverage = Coverage(data_file=data_file, source=["backend"], branch=True)
    coverage.load()
    coverage.xml_report(morfs=paths, outfile=output, ignore_errors=True)
    return output


def counts(classes: list[ElementTree.Element]) -> tuple[int, int, int, int]:
    valid = covered = branches = covered_branches = 0
    for item in classes:
        for line in item.findall("lines/line"):
            valid += 1
            covered += int(line.attrib["hits"]) > 0
            if line.get("branch") == "true":
                match = re.search(r"\((\d+)/(\d+)\)", line.attrib["condition-coverage"])
                if match is None:
                    raise ValueError("Unknown branch-count format")
                covered_branches += int(match[1])
                branches += int(match[2])
    return valid, covered, branches, covered_branches


def rate(covered: int, valid: int) -> str:
    return format(covered / valid, ".4g") if valid else "1"


def merge_reports(reports: list[str], output: str) -> None:
    roots = [ElementTree.parse(report).getroot() for report in reports]
    root = copy.deepcopy(roots[0])
    packages_node = root.find("packages")
    if packages_node is None:
        raise ValueError("Coverage report has no packages")
    packages_node.clear()
    packages: dict[str, ElementTree.Element] = {}
    seen_files: set[str] = set()
    for part in roots:
        if ElementTree.tostring(part.find("sources")) != ElementTree.tostring(
            root.find("sources")
        ):
            raise ValueError("Report source roots differ")
        for package in part.findall("packages/package"):
            name = package.attrib["name"]
            merged = packages.setdefault(
                name, ElementTree.Element("package", package.attrib)
            )
            classes = merged.find("classes")
            if classes is None:
                classes = ElementTree.SubElement(merged, "classes")
            for item in package.findall("classes/class"):
                filename = item.attrib["filename"]
                if filename in seen_files:
                    raise ValueError(f"Duplicate file: {filename}")
                seen_files.add(filename)
                classes.append(item)
    all_classes = []
    for _, package in sorted(packages.items()):
        classes = package.find("classes")
        if classes is None:
            raise ValueError("Coverage package has no classes")
        classes[:] = sorted(classes, key=lambda item: item.attrib["filename"])
        valid, covered, branches, covered_branches = counts(list(classes))
        package.set("line-rate", rate(covered, valid))
        package.set("branch-rate", rate(covered_branches, branches))
        packages_node.append(package)
        all_classes.extend(classes)
    valid, covered, branches, covered_branches = counts(all_classes)
    root.set("lines-valid", str(valid))
    root.set("lines-covered", str(covered))
    root.set("branches-valid", str(branches))
    root.set("branches-covered", str(covered_branches))
    root.set("line-rate", rate(covered, valid))
    root.set("branch-rate", rate(covered_branches, branches))
    ElementTree.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def parallel_xml(data_file: str, files: list[str], output: str) -> None:
    if not files:
        raise ValueError("Backend coverage contains no measured files")
    workers = min(4, os.cpu_count() or 1, len(files))
    weights = {}
    for path in files:
        try:
            weights[path] = Path(path).stat().st_size
        except OSError:
            weights[path] = 0
    chunks: list[list[str]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for path in sorted(files, key=lambda value: (-weights[value], value)):
        index = min(range(workers), key=loads.__getitem__)
        chunks[index].append(path)
        loads[index] += weights[path] + 10000
    with tempfile.TemporaryDirectory() as directory:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    report_part, data_file, paths, str(Path(directory) / f"{i}.xml")
                )
                for i, paths in enumerate(chunks)
            ]
            reports = [future.result() for future in futures]
        merge_reports(reports, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    coverage = Coverage(data_file=args.data_file, source=["backend"], branch=True)
    coverage.load()
    coverage.combine()
    if not coverage.get_data().has_arcs():
        raise SystemExit("Backend branch coverage data is missing")
    coverage.save()
    combined = time.perf_counter()
    parallel_xml(
        args.data_file, list(coverage.get_data().measured_files()), args.output
    )
    print(
        json.dumps(
            {
                "combine_seconds": combined - started,
                "xml_seconds": time.perf_counter() - combined,
                "measured_files": len(coverage.get_data().measured_files()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
