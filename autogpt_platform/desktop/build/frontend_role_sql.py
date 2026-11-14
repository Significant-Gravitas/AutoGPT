"""Extract the frontend database-role policy from the appliance's bootstrap.sh.

The policy (which tables the Next server's `autogpt_frontend` role may touch)
is security-relevant and already has one home and one test
(single-container/tests/test_frontend_environment.py reads it out of
bootstrap.sh the same way). The desktop bundle ships a copy extracted at
build time instead of maintaining a second version by hand.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HEREDOC = re.compile(
    r"configure_frontend_database_role\(\) \{.*?<<'SQL'\n(?P<sql>.*?)\nSQL\n",
    re.DOTALL,
)
REQUIRED = (
    'platform."UserAuthSession"',
    "NOSUPERUSER",
    "GRANT CONNECT ON DATABASE postgres TO autogpt_frontend",
)


def extract(bootstrap_script: Path) -> str:
    match = HEREDOC.search(bootstrap_script.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"no frontend role policy found in {bootstrap_script}")
    sql = match.group("sql")
    missing = [snippet for snippet in REQUIRED if snippet not in sql]
    if missing:
        raise ValueError(f"frontend role policy is missing {missing}")
    return sql + "\n"


def main() -> int:
    Path(sys.argv[2]).write_text(extract(Path(sys.argv[1])), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
