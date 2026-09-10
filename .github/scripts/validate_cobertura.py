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

    classes = root.findall("packages/package/classes/class")
    if not classes:
        raise ValueError("coverage contains no source classes")
    line_count = 0
    for source in classes:
        if not source.get("filename", "").strip():
            raise ValueError("coverage class is missing a filename")
        for line in source.findall("lines/line"):
            if nonnegative_integer(line, "number") == 0:
                raise ValueError("coverage line numbers must be positive")
            nonnegative_integer(line, "hits")
            line_count += 1
    if line_count == 0:
        raise ValueError("coverage contains no source lines")


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
