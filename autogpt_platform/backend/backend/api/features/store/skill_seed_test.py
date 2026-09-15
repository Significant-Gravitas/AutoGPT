"""Tests for the catalog loader: a skill is a directory with a SKILL.md and
the files beside it, and a broken one fails the seed rather than the install."""

import pathlib

import pytest

from backend.api.features.store.skill_seed import CatalogEntry, _load, load_catalog
from backend.copilot.tools.skills import SkillPackageError

SKILL_MD = "---\nname: demo\ndescription: A demo skill.\n---\n\n# Demo\n"
ENTRY = CatalogEntry(slug="demo", categories=["content"], required_providers=[])


def _write(root: pathlib.Path, relative: str, content: str) -> pathlib.Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_a_skill_loads_its_siblings(tmp_path):
    """A package's resources travel with it, executable bit and all, at the
    paths its SKILL.md references them by."""
    _write(tmp_path, "skills/demo/SKILL.md", SKILL_MD)
    _write(tmp_path, "skills/demo/references/API.md", "# API\n")
    script = _write(tmp_path, "skills/demo/scripts/run.py", "print(1)\n")
    script.chmod(0o755)

    parsed, files = _load(tmp_path, ENTRY)

    assert (parsed.name, parsed.body.strip()) == ("demo", "# Demo")
    assert [(f.relative_path, f.content, f.is_executable) for f in files] == [
        ("references/API.md", b"# API\n", False),
        ("scripts/run.py", b"print(1)\n", True),
    ]


def test_a_skill_breaking_a_package_rule_fails_the_seed(tmp_path):
    """A hidden file would install into the skill folder unseen, so the seed
    refuses it here rather than at every user's install."""
    _write(tmp_path, "skills/demo/SKILL.md", SKILL_MD)
    _write(tmp_path, "skills/demo/.env", "SECRET=1\n")

    with pytest.raises(SkillPackageError):
        _load(tmp_path, ENTRY)


def test_a_skill_whose_frontmatter_name_differs_is_refused(tmp_path):
    """The frontmatter name becomes the installed skill's name, so a mismatch
    would install a skill under a name the marketplace never shows."""
    _write(tmp_path, "skills/demo/SKILL.md", SKILL_MD.replace("demo", "other"))

    with pytest.raises(ValueError, match="must match the catalog slug"):
        _load(tmp_path, ENTRY)


def test_a_listed_skill_without_a_skill_md_is_refused(tmp_path):
    _write(tmp_path, "skills/demo/references/API.md", "# API\n")

    with pytest.raises(ValueError, match="is missing"):
        _load(tmp_path, ENTRY)


def test_the_catalog_folds_categories_onto_the_canonical_set(tmp_path):
    _write(
        tmp_path,
        "catalog.yml",
        "skills:\n"
        "  - slug: demo\n"
        "    categories: [writing]\n"
        "    required_providers: [google]\n"
        "    source: platform\n",
    )

    assert load_catalog(tmp_path) == [
        CatalogEntry(slug="demo", categories=["content"], required_providers=["google"])
    ]


@pytest.mark.parametrize(
    "catalog, message",
    [
        ("skills:\n  - slug: demo\n    categories: [nonsense]\n", "category"),
        (
            "skills:\n  - slug: demo\n    categories: [sales]\n"
            "  - slug: demo\n    categories: [sales]\n",
            "listed twice",
        ),
        ("skills: []\n", "lists no skills"),
        ("skills:\n  - categories: [sales]\n", "has no slug"),
        (
            "skills:\n  - slug: ../demo\n    categories: [sales]\n",
            "name must be a slug",
        ),
        (
            "skills:\n  - slug: demo\n    categories: [sales]\n"
            "    required_providers: google\n",
            "required_providers must be a list of strings",
        ),
        (
            "skills:\n  - slug: demo\n    categories: [sales]\n"
            "    required_providers: [google, 1]\n",
            "required_providers must be a list of strings",
        ),
        (
            'skills:\n  - slug: demo\n    categories: ""\n',
            "categories must be a list of strings",
        ),
        (
            "skills:\n  - slug: demo\n    categories: [sales]\n"
            '    required_providers: ""\n',
            "required_providers must be a list of strings",
        ),
    ],
    ids=[
        "unknown-category",
        "duplicate-slug",
        "empty",
        "no-slug",
        "invalid-slug",
        "scalar-provider",
        "mixed-providers",
        "falsey-scalar-categories",
        "falsey-scalar-providers",
    ],
)
def test_a_malformed_catalog_fails_before_anything_is_written(
    tmp_path, catalog: str, message: str
):
    _write(tmp_path, "catalog.yml", catalog)

    with pytest.raises(ValueError, match=message):
        load_catalog(tmp_path)
