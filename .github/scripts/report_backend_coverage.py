#!/usr/bin/env python3

import argparse
import json
import time

from coverage import Coverage


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
    coverage.xml_report(outfile=args.output, ignore_errors=True)
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
