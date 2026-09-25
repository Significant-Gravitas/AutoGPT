"""Reject a second source of production marketplace skill content."""

import ast
from pathlib import Path


def check(root: Path) -> list[str]:
    store = root / "backend/api/features/store"
    experts = root / "backend/api/features/experts"
    errors = []
    if (store / "starter_skills").exists():
        errors.append("Production starter skills belong in skills-catalog")
    for path in (
        store / "skill_seed.py",
        experts / "seed.py",
        experts / "roster_wave_three.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict) and any(
                isinstance(key, ast.Constant) and key.value == "bundled_skills"
                for key in node.keys
            ):
                errors.append(
                    f"{path.name}: expert skill assignments belong in the catalogue"
                )
            if isinstance(node, ast.Name) and node.id in {
                "STARTER_SKILLS",
                "RETIRED_STARTER_SLUGS",
            }:
                errors.append(f"{path.name}: hard-coded skill inventory is forbidden")
    return sorted(set(errors))


if __name__ == "__main__":
    violations = check(Path(__file__).resolve().parents[1])
    for violation in violations:
        print(violation)
    raise SystemExit(bool(violations))
