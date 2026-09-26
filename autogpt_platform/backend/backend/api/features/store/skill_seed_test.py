"""Tests for the catalog loader: a skill is a directory with a SKILL.md and
the files beside it, and a broken one fails the seed rather than the install."""

import io
import pathlib
import tarfile
from unittest.mock import Mock

import httpx
import pytest

from backend.api.features.store.skill_model import skill_title
from backend.api.features.store.skill_seed import (
    RETIRED_STARTER_SLUGS,
    STARTER_SKILLS,
    CatalogEntry,
    _attribution_value,
    _download_catalog,
    _extract_catalog_archive,
    _load,
    _load_starter,
    load_catalog,
)
from backend.copilot.tools.skills import (
    MAX_BODY_CHARS,
    MAX_DESCRIPTION_CHARS,
    MAX_TRIGGER_CHARS,
    MAX_TRIGGERS,
    ParsedSkill,
    SkillPackageError,
)

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


def test_a_skill_package_cannot_read_through_a_symlink(tmp_path):
    _write(tmp_path, "skills/demo/SKILL.md", SKILL_MD)
    outside = _write(tmp_path, "outside.txt", "secret\n")
    linked = tmp_path / "skills" / "demo" / "references" / "outside.txt"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(outside)

    with pytest.raises(SkillPackageError, match="may not be a symlink"):
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


@pytest.mark.parametrize(
    ("frontmatter", "body", "message"),
    [
        (
            f"description: {'x' * (MAX_DESCRIPTION_CHARS + 1)}\n",
            "# Demo\n",
            "description is",
        ),
        ("description: A demo skill.\n", "x" * (MAX_BODY_CHARS + 1), "body must"),
        (
            "description: A demo skill.\n"
            f"triggers: {[f't{i}' for i in range(MAX_TRIGGERS + 1)]}\n",
            "# Demo\n",
            "triggers must",
        ),
        (
            "description: A demo skill.\n"
            f"triggers: [{'x' * (MAX_TRIGGER_CHARS + 1)}]\n",
            "# Demo\n",
            "exceeds",
        ),
    ],
)
def test_a_skill_that_cannot_be_installed_is_refused(
    tmp_path, frontmatter: str, body: str, message: str
):
    skill_md = f"---\nname: demo\n{frontmatter}---\n\n{body}"
    _write(tmp_path, "skills/demo/SKILL.md", skill_md)

    with pytest.raises(ValueError, match=message):
        _load(tmp_path, ENTRY)


def test_attribution_accepts_top_level_fields_but_prefers_metadata():
    parsed = ParsedSkill(
        name="demo",
        description="Demo",
        body="# Demo\n",
        extra={"source": "top/repo", "source_url": "https://example.com/top"},
    )

    assert _attribution_value(parsed, {"source": "nested/repo"}, "source") == (
        "nested/repo"
    )
    assert _attribution_value(parsed, {}, "source_url") == "https://example.com/top"


def test_retired_starters_are_not_also_seeded():
    """A slug in both lists would be delisted right after being upserted."""
    seeded = {entry["slug"] for entry in STARTER_SKILLS}
    assert not seeded & set(RETIRED_STARTER_SLUGS), seeded & set(RETIRED_STARTER_SLUGS)


@pytest.mark.parametrize("entry", STARTER_SKILLS, ids=lambda entry: entry["slug"])
def test_every_checked_in_starter_skill_loads(entry):
    parsed, files = _load_starter(entry)

    assert parsed.name == entry["slug"]
    assert files == []


def test_a_starter_that_cannot_be_installed_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.api.features.store.skill_seed._CONTENT_DIR", tmp_path)
    _write(
        tmp_path,
        "demo.md",
        SKILL_MD.replace("A demo skill.", "x" * (MAX_DESCRIPTION_CHARS + 1)),
    )

    with pytest.raises(ValueError, match="description is"):
        _load_starter(ENTRY)


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


def _catalog_tarball() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        content = b"skills: []\n"
        member = tarfile.TarInfo("owner-repo-abc123/catalog.yml")
        member.size = len(content)
        tar.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


def _mock_github(mocker, status: int, content: bytes = b"") -> Mock:
    request = httpx.Request("GET", "https://api.github.com/repos/o/r/tarball/main")
    return mocker.patch(
        "backend.api.features.store.skill_seed.httpx.get",
        return_value=httpx.Response(status, content=content, request=request),
    )


@pytest.fixture
def no_catalog_token(monkeypatch):
    for name in ("SKILLS_CATALOG_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def test_the_public_catalog_downloads_without_a_token(
    mocker, no_catalog_token, tmp_path
):
    get = _mock_github(mocker, 200, _catalog_tarball())

    root = _download_catalog(tmp_path)

    assert (root / "catalog.yml").read_text() == "skills: []\n"
    assert "Authorization" not in get.call_args.kwargs["headers"]


def test_a_catalog_token_is_sent_when_set(
    mocker, no_catalog_token, monkeypatch, tmp_path
):
    monkeypatch.setenv("SKILLS_CATALOG_TOKEN", "ghp_test")
    get = _mock_github(mocker, 200, _catalog_tarball())

    _download_catalog(tmp_path)

    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer ghp_test"


def test_a_refused_anonymous_download_says_how_to_fix_it(
    mocker, no_catalog_token, tmp_path
):
    _mock_github(mocker, 404)

    with pytest.raises(RuntimeError, match="may be private.*SKILLS_CATALOG_TOKEN"):
        _download_catalog(tmp_path)


def test_archive_fallback_extracts_checked_files(mocker, monkeypatch, tmp_path):
    monkeypatch.delattr(tarfile, "data_filter")
    archive = mocker.Mock()
    member = tarfile.TarInfo("repo/catalog.yml")
    archive.getmembers.return_value = [member]

    _extract_catalog_archive(archive, tmp_path)

    archive.extractall.assert_called_once_with(tmp_path, members=[member])


def test_archive_fallback_rejects_paths_outside_the_target(
    mocker, monkeypatch, tmp_path
):
    monkeypatch.delattr(tarfile, "data_filter")
    archive = mocker.Mock()
    member = tarfile.TarInfo("../outside")
    archive.getmembers.return_value = [member]

    with pytest.raises(RuntimeError, match="unsafe catalog archive member"):
        _extract_catalog_archive(archive, tmp_path)

    archive.extractall.assert_not_called()


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
    entry = next(entry for entry in STARTER_SKILLS if entry["slug"] == slug)
    parsed, _ = _load_starter(entry)
    assert skill_title(parsed.name, parsed.body) == title
