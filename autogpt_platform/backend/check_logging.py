import pathlib
import re

PATTERN = re.compile(r"logging\.getLogger\(")
ROOT = pathlib.Path(__file__).resolve().parent / "backend"

errors = []

for path in ROOT.rglob("*.py"):
    if path.name == "logging.py":
        continue
    text = path.read_text()
    if PATTERN.search(text) and "TruncatedLogger" not in text:
        errors.append(str(path.relative_to(ROOT)))

if errors:
    print("Plain logging.getLogger usage detected in:\n" + "\n".join(errors))
