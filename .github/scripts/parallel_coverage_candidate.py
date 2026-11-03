import argparse
import concurrent.futures
import copy
import pathlib
import re
import tempfile
from xml.etree import ElementTree

from coverage import Coverage


def report_part(data_file, paths, output):
    coverage = Coverage(data_file=data_file, source=["backend"], branch=True)
    coverage.load()
    coverage.xml_report(morfs=paths, outfile=output, ignore_errors=True)
    return output


def counts(classes):
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


def rate(covered, valid):
    return format(covered / valid, ".4g") if valid else "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    coverage = Coverage(data_file=args.data_file, source=["backend"], branch=True)
    coverage.load()
    coverage.combine()
    if not coverage.get_data().has_arcs():
        raise SystemExit("Backend branch coverage data is missing")
    coverage.save()
    files = list(coverage.get_data().measured_files())
    weights = {}
    for path in files:
        try:
            weights[path] = pathlib.Path(path).stat().st_size
        except OSError:
            weights[path] = 0
    chunks = [[] for _ in range(4)]
    loads = [0] * 4
    for path in sorted(files, key=lambda value: (-weights[value], value)):
        index = min(range(4), key=loads.__getitem__)
        chunks[index].append(path)
        loads[index] += weights[path] + 10000
    with tempfile.TemporaryDirectory() as directory:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(report_part, args.data_file, paths, str(pathlib.Path(directory) / f"{i}.xml")) for i, paths in enumerate(chunks) if paths]
            roots = [ElementTree.parse(future.result()).getroot() for future in futures]
        root = copy.deepcopy(roots[0])
        packages_node = root.find("packages")
        packages_node.clear()
        packages = {}
        seen_files = set()
        for part in roots:
            if ElementTree.tostring(part.find("sources")) != ElementTree.tostring(root.find("sources")):
                raise ValueError("Report source roots differ")
            for package in part.findall("packages/package"):
                name = package.attrib["name"]
                merged = packages.setdefault(name, ElementTree.Element("package", package.attrib))
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
        for name, package in sorted(packages.items()):
            classes = package.find("classes")
            classes[:] = sorted(classes, key=lambda item: item.attrib["filename"])
            valid, covered, branches, covered_branches = counts(classes)
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
        ElementTree.ElementTree(root).write(args.output, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
