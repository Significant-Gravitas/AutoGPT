"""Tests for the starter-skill loader: a starter is either a flat SKILL.md or
a package directory, and a broken one fails the seed rather than the install."""

import pathlib

import pytest

from backend.api.features.store.skill_model import skill_title
from backend.api.features.store.skill_seed import STARTER_SKILLS, _load
from backend.copilot.tools.skills import SkillPackageError

SKILL_MD = "---\nname: demo\ndescription: A demo starter.\n---\n\n# Demo\n"


def _write(root: pathlib.Path, relative: str, content: str) -> pathlib.Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_a_flat_starter_loads_with_no_package_files(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.api.features.store.skill_seed._CONTENT_DIR", tmp_path)
    _write(tmp_path, "demo.md", SKILL_MD)

    parsed, files = _load("demo")

    assert (parsed.name, parsed.body.strip()) == ("demo", "# Demo")
    assert files == []


def test_a_directory_starter_loads_its_siblings(monkeypatch, tmp_path):
    """A package directory's resources travel with it, executable bit and all,
    at the paths its SKILL.md references them by."""
    monkeypatch.setattr("backend.api.features.store.skill_seed._CONTENT_DIR", tmp_path)
    _write(tmp_path, "demo/SKILL.md", SKILL_MD)
    _write(tmp_path, "demo/references/API.md", "# API\n")
    script = _write(tmp_path, "demo/scripts/run.py", "print(1)\n")
    script.chmod(0o755)

    parsed, files = _load("demo")

    assert parsed.name == "demo"
    assert [(f.relative_path, f.content, f.is_executable) for f in files] == [
        ("references/API.md", b"# API\n", False),
        ("scripts/run.py", b"print(1)\n", True),
    ]


def test_a_directory_starter_breaking_a_package_rule_fails_the_seed(
    monkeypatch, tmp_path
):
    """A hidden file would install into the skill folder unseen, so the seed
    refuses it here rather than at every user's install."""
    monkeypatch.setattr("backend.api.features.store.skill_seed._CONTENT_DIR", tmp_path)
    _write(tmp_path, "demo/SKILL.md", SKILL_MD)
    _write(tmp_path, "demo/.env", "SECRET=1\n")

    with pytest.raises(SkillPackageError):
        _load("demo")


def test_a_starter_whose_frontmatter_name_differs_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.api.features.store.skill_seed._CONTENT_DIR", tmp_path)
    _write(tmp_path, "demo/SKILL.md", SKILL_MD.replace("demo", "something-else"))

    with pytest.raises(ValueError, match="must match the listing slug"):
        _load("demo")


@pytest.mark.parametrize("entry", STARTER_SKILLS, ids=lambda e: e["slug"])
def test_every_shipped_starter_still_loads(entry):
    parsed, files = _load(entry["slug"])
    assert parsed.name == entry["slug"]
    assert files == []


@pytest.mark.parametrize(
    ("slug", "title"),
    [
        ("seo-content-brief", "SEO content brief"),
        ("on-page-seo-audit", "On-page SEO audit"),
        ("icp-and-positioning", "ICP and positioning"),
        ("brand-voice-guide", "Brand voice guide"),
    ],
)
def test_a_shipped_starter_keeps_the_casing_its_author_wrote(slug, title):
    """A starter's frontmatter name is its slug, so a title derived from it
    reads back as "Seo content brief"."""
    parsed, _ = _load(slug)
    assert skill_title(parsed.name, parsed.body) == title
