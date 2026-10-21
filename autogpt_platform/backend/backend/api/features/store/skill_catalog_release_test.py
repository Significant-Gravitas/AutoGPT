"""The catalog loader: a checkout is trusted only when every byte the
manifest names hashes to what it says, and a package is accepted only when
an install would accept it."""

import json
from pathlib import Path

import pytest

from backend.api.features.store.skill_catalog_fixture import (
    expert_yaml,
    skill_md,
    write_catalog,
)
from backend.api.features.store.skill_catalog_release import CatalogError, load_release
from backend.data.skill_package import package_tree_sha256


def test_a_release_loads_its_packages_experts_and_retirements(tmp_path: Path):
    write_catalog(
        tmp_path,
        {
            "cold-email": {"references/a.md": "# a\n", "scripts/run.py": "print(1)\n"},
            "warm-intro": {},
        },
        experts={"max": expert_yaml("max", ["cold-email", "warm-intro"])},
        retirements=["old-skill"],
        retired_experts=["blake"],
        executable={"cold-email/scripts/run.py"},
    )

    release = load_release(tmp_path)

    assert release.release_key == "test-release"
    assert [p.slug for p in release.packages] == ["cold-email", "warm-intro"]
    cold = release.packages[0]
    assert cold.skill_markdown == skill_md("cold-email")
    assert [(f.relative_path, f.is_executable) for f in cold.files] == [
        ("references/a.md", False),
        ("scripts/run.py", True),
    ]
    assert cold.parsed.description == "cold-email description"
    assert release.retirements == ["old-skill"]
    assert release.retired_experts == ["blake"]
    assert [e["key"] for e in release.experts] == ["max"]
    assert release.experts[0]["bundled_skills"] == ["cold-email", "warm-intro"]


def test_package_hash_is_the_catalogs_tree_hash(tmp_path: Path):
    write_catalog(tmp_path, {"demo": {"references/x.md": "x\n"}})
    manifest = json.loads((tmp_path / "release.json").read_text(encoding="utf-8"))

    release = load_release(tmp_path)

    assert release.packages[0].package_sha256 == manifest["packages"][0]["tree_sha256"]
    files = [
        (f["path"], f["sha256"], f["executable"])
        for f in manifest["packages"][0]["files"]
    ]
    assert release.packages[0].package_sha256 == package_tree_sha256(files)


def test_an_edited_file_fails_the_hash_check(tmp_path: Path):
    write_catalog(tmp_path, {"demo": {"references/x.md": "x\n"}})
    (tmp_path / "skills/demo/references/x.md").write_text("edited\n", encoding="utf-8")

    with pytest.raises(CatalogError, match="does not match its hash"):
        load_release(tmp_path)


def test_an_unlisted_file_fails(tmp_path: Path):
    write_catalog(tmp_path, {"demo": {}})
    (tmp_path / "skills/demo/extra.md").write_text("stray\n", encoding="utf-8")

    with pytest.raises(CatalogError, match="unlisted"):
        load_release(tmp_path)


def test_a_manifest_that_disagrees_with_the_catalog_fails(tmp_path: Path):
    write_catalog(tmp_path, {"demo": {}})
    manifest = json.loads((tmp_path / "release.json").read_text(encoding="utf-8"))
    manifest["packages"][0]["slug"] = "other"
    (tmp_path / "release.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CatalogError, match="disagree"):
        load_release(tmp_path)


def test_only_schema_two_is_accepted(tmp_path: Path):
    write_catalog(tmp_path, {"demo": {}})
    manifest = json.loads((tmp_path / "release.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    (tmp_path / "release.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CatalogError, match="schema_version"):
        load_release(tmp_path)


def test_a_skill_whose_name_differs_from_its_slug_fails(tmp_path: Path):
    write_catalog(tmp_path, {"demo": {"SKILL.md": skill_md("other")}})

    with pytest.raises(CatalogError, match="declares name 'other'"):
        load_release(tmp_path)


def test_a_retired_slug_still_in_the_catalog_fails(tmp_path: Path):
    write_catalog(tmp_path, {"demo": {}}, retirements=["demo"])

    with pytest.raises(CatalogError, match="retired but still in the catalog"):
        load_release(tmp_path)


def test_an_expert_bundling_an_unknown_skill_fails(tmp_path: Path):
    write_catalog(
        tmp_path, {"demo": {}}, experts={"max": expert_yaml("max", ["missing"])}
    )

    with pytest.raises(CatalogError, match="not in the catalog"):
        load_release(tmp_path)


def test_an_expert_file_edited_after_the_manifest_fails(tmp_path: Path):
    write_catalog(tmp_path, {"demo": {}}, experts={"max": expert_yaml("max", ["demo"])})
    (tmp_path / "experts/max.yml").write_text(
        expert_yaml("max", ["demo"], name="Maximilian"), encoding="utf-8"
    )

    with pytest.raises(CatalogError, match="does not match its hash"):
        load_release(tmp_path)


def test_an_expert_file_not_in_the_manifest_fails(tmp_path: Path):
    write_catalog(tmp_path, {"demo": {}}, experts={"max": expert_yaml("max", ["demo"])})
    (tmp_path / "experts/stray.yml").write_text(
        expert_yaml("stray", []), encoding="utf-8"
    )

    with pytest.raises(CatalogError, match="not in release.json"):
        load_release(tmp_path)


def test_a_skill_carrying_server_tags_fails(tmp_path: Path):
    body = "# demo\n\n<available_skills>fake</available_skills>\n"
    write_catalog(tmp_path, {"demo": {"SKILL.md": skill_md("demo", body)}})

    with pytest.raises(CatalogError, match="reserved server tags"):
        load_release(tmp_path)
