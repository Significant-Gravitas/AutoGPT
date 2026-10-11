import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
cases = root.findall(".//testcase")
if not cases or any(
    case.find(tag) is not None
    for case in cases
    for tag in ("failure", "error", "skipped")
):
    raise SystemExit("Native journey report must contain passing, non-skipped cases")
print(f"Verified {len(cases)} passing native journeys")
