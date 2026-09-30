import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def nonnegative_integer(element: ET.Element, attribute: str) -> int:
    value = element.get(attribute, "")
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"{element.tag}.{attribute} must be a non-negative integer")
    return int(value)


def validate_report(root: ET.Element) -> None:
    if root.tag != "coverage":
        raise ValueError("report is not Cobertura coverage XML")
    valid = nonnegative_integer(root, "lines-valid")
    covered = nonnegative_integer(root, "lines-covered")
    if valid == 0 or covered > valid:
        raise ValueError("coverage must contain lines with valid coverage counts")
    rate = float(root.get("line-rate", "nan"))
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("coverage.line-rate must be between 0 and 1")

    classes = root.findall("packages//class")
    if not classes:
        raise ValueError("coverage contains no source classes")
    line_count = 0
    covered_count = 0
    for source in classes:
        if not source.get("filename", "").strip():
            raise ValueError("coverage class is missing a filename")
        for line in source.findall("lines/line"):
            if nonnegative_integer(line, "number") == 0:
                raise ValueError("coverage line numbers must be positive")
            covered_count += nonnegative_integer(line, "hits") > 0
            line_count += 1
    if line_count == 0:
        raise ValueError("coverage contains no source lines")
    if valid != line_count or covered != covered_count:
        raise ValueError("coverage summary counts do not match source lines")
    # Istanbul truncates percentages to two decimals before converting to a rate.
    if not math.isclose(rate, covered_count / line_count, rel_tol=0, abs_tol=1e-4):
        raise ValueError("coverage.line-rate does not match source lines")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_report(ET.parse(args.report).getroot())
    except (OSError, ET.ParseError, ValueError) as error:
        print(f"{args.report}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
