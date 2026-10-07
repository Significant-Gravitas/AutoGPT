"""Build a skills catalog checkout on disk for tests: packages, expert files
and a ``release.json`` whose hashes are computed the way the catalog's own
``tools/release.py`` computes them. Test-only; imported by the loader,
publisher and roster tests."""

import hashlib
import json
from pathlib import Path

import yaml

from backend.data.skill_package import package_tree_sha256

SKILL_MD_TEMPLATE = """---
name: {slug}
description: "{slug} description"
triggers: ["{slug} trigger"]
version: "1"
---

# {slug}

Body of {slug}.
"""

EXPERT_TEMPLATE = """key: {key}
name: {name}
role: Sales
job_title: Sales Rep
tagline: Finds leads.
avatar_url: /avatars/{key}.webp
categories:
- sales
bio: I find leads.
identity: You are {name}, a sales expert.
voice_preferences: Direct.
voice_samples:
- label: Punchy
  text: Stop guessing.
boundaries: Never invent numbers.
day_one: []
preloads: []
routines: []
skills:
{skills}
"""


def skill_md(slug: str, body: str | None = None) -> str:
    text = SKILL_MD_TEMPLATE.format(slug=slug)
    return text if body is None else text.split("\n\n", 1)[0] + "\n\n" + body


def expert_yaml(key: str, skills: list[str], *, name: str | None = None) -> str:
    listed = "\n".join(f"- {slug}" for slug in skills) if skills else "[]"
    if not skills:
        return EXPERT_TEMPLATE.replace("skills:\n{skills}\n", "skills: []\n").format(
            key=key, name=name or key.title(), skills=""
        )
    return EXPERT_TEMPLATE.format(key=key, name=name or key.title(), skills=listed)


def write_catalog(
    root: Path,
    packages: dict[str, dict[str, bytes | str]],
    *,
    experts: dict[str, str] | None = None,
    retirements: list[str] | None = None,
    retired_experts: list[str] | None = None,
    release_key: str = "test-release",
    categories: dict[str, list[str]] | None = None,
    executable: set[str] | None = None,
) -> Path:
    """Write *packages* (slug → relative path → content; ``SKILL.md`` is
    added when absent) and *experts* (key → YAML text) under *root* with a
    matching ``catalog.yml`` and ``release.json``. Returns *root*."""
    executable = executable or set()
    skills_dir = root / "skills"
    manifest_packages = []
    for slug in sorted(packages):
        files = dict(packages[slug])
        if "SKILL.md" not in files:
            files["SKILL.md"] = skill_md(slug)
        listed = []
        for path in sorted(files):
            content = files[path]
            data = content.encode("utf-8") if isinstance(content, str) else content
            target = skills_dir / slug / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            listed.append(
                {
                    "path": path,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "executable": f"{slug}/{path}" in executable,
                }
            )
        manifest_packages.append(
            {
                "slug": slug,
                "tree_sha256": package_tree_sha256(
                    (f["path"], f["sha256"], f["executable"]) for f in listed
                ),
                "files": listed,
            }
        )
    catalog = {
        "skills": [
            {
                "slug": slug,
                "categories": (categories or {}).get(slug, ["sales"]),
                "required_providers": [],
                "source": "platform",
            }
            for slug in sorted(packages)
        ]
    }
    catalog_bytes = yaml.safe_dump(catalog, sort_keys=False).encode("utf-8")
    (root / "catalog.yml").write_bytes(catalog_bytes)
    manifest_experts = []
    experts_dir = root / "experts"
    experts_dir.mkdir(exist_ok=True)
    for key in sorted(experts or {}):
        text = (experts or {})[key]
        data = text.encode("utf-8")
        (experts_dir / f"{key}.yml").write_bytes(data)
        manifest_experts.append(
            {
                "key": key,
                "sha256": hashlib.sha256(data).hexdigest(),
                "skills": list(yaml.safe_load(text).get("skills") or []),
            }
        )
    manifest = {
        "schema_version": 2,
        "release_key": release_key,
        "provenance": {"catalogue_repository": "test"},
        "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "packages": manifest_packages,
        "experts": manifest_experts,
        "retirements": sorted(retirements or []),
        "retired_experts": sorted(retired_experts or []),
        "system_packages": [],
    }
    (root / "release.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return root
