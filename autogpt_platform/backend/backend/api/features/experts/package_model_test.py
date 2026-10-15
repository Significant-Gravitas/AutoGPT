import json

import pytest
from pydantic import ValidationError

from backend.api.features.experts.package_model import (
    MANIFEST_PATH,
    ExpertManifest,
    ExpertPackage,
    ExpertPackageError,
    PackagedRoutine,
    PackagedSkill,
    PackagedWorkflow,
    manifest_json,
    package_filename,
    skill_path,
    validate_expert_package,
    workflow_path,
)
from backend.copilot.tools.skills import MAX_PACKAGE_BYTES, SkillFile, SkillPackage


def _skill(slug: str = "write-briefs") -> PackagedSkill:
    return PackagedSkill(
        slug=slug, title="Write briefs", description="Drafts", path=skill_path(slug)
    )


def _workflow(path: str = "workflows/01-daily-digest.json") -> PackagedWorkflow:
    return PackagedWorkflow(
        path=path,
        name="Daily digest",
        description="Sends a digest",
        store_listing_version_id="slv-1",
        schedule_cron="0 9 * * *",
        integration_providers=["github"],
    )


def _manifest(**overrides: object) -> ExpertManifest:
    fields: dict[str, object] = {
        "name": "Ada",
        "role": "Research assistant",
        "job_title": None,
        "tagline": None,
        "bio": None,
        "categories": ["research"],
        "color": "",
        "avatar_url": "/experts/ada.png",
        "identity": "You are Ada.",
        "voice_preferences": "Plain.",
        "voice_samples": [],
        "boundaries": "",
        "day_one": [],
        "skills": [_skill()],
        "workflows": [_workflow()],
        "routines": [],
    }
    fields.update(overrides)
    return ExpertManifest.model_validate(fields)


def _package(**overrides: object) -> ExpertPackage:
    fields: dict[str, object] = {
        "manifest": _manifest(),
        "skills": {"write-briefs": SkillPackage(skill_md="---\nname: x\n---\n")},
        "workflows": {"workflows/01-daily-digest.json": b"{}"},
    }
    fields.update(overrides)
    return ExpertPackage.model_validate(fields)


@pytest.mark.parametrize("key", ["memory", "conversations", "files"])
def test_manifest_refuses_keys_outside_the_packing_list(key: str):
    data = _manifest().model_dump(mode="json")
    data[key] = []
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExpertManifest.model_validate(data)


@pytest.mark.parametrize(
    "model, data",
    [
        (PackagedSkill, _skill().model_dump()),
        (PackagedWorkflow, _workflow().model_dump()),
        (
            PackagedRoutine,
            {"key": None, "title": "t", "prompt": "p", "crons": [], "asks": []},
        ),
    ],
)
def test_nested_entries_refuse_unknown_keys(model: type, data: dict):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate({**data, "memory": "secret"})


def test_soul_rows_refuse_unknown_keys():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _manifest(voice_samples=[{"label": "a", "text": "hi", "files": []}])
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _manifest(day_one=[{"title": "Set up", "conversations": []}])


def test_manifest_pins_format_version_one():
    assert _manifest().format_version == 1
    with pytest.raises(ValidationError):
        _manifest(format_version=2)


def test_manifest_refuses_an_avatar_that_is_not_https_or_site_relative():
    with pytest.raises(ValidationError, match="https URL or a relative path"):
        _manifest(avatar_url="http://example.com/a.png")


def test_manifest_json_round_trips_and_is_stable():
    manifest = _manifest()
    raw = manifest_json(manifest)
    assert ExpertManifest.model_validate_json(raw) == manifest
    assert json.loads(raw)["format_version"] == 1
    assert manifest_json(_manifest()) == raw


def test_valid_package_passes():
    validate_expert_package(_package())


def test_skill_named_by_the_manifest_but_missing_is_refused():
    with pytest.raises(ExpertPackageError, match="skills"):
        validate_expert_package(_package(skills={}))


def test_skill_not_named_by_the_manifest_is_refused():
    extra = SkillPackage(skill_md="x")
    with pytest.raises(ExpertPackageError, match="skills"):
        validate_expert_package(
            _package(skills={"write-briefs": extra, "stowaway": extra})
        )


def test_skill_card_must_point_at_its_own_folder():
    card = _skill().model_copy(update={"path": "skills/elsewhere/"})
    with pytest.raises(ExpertPackageError, match="path"):
        validate_expert_package(_package(manifest=_manifest(skills=[card])))


def test_skill_slug_that_would_escape_its_folder_is_refused():
    card = PackagedSkill(slug="..", title="t", description="d", path="skills/../")
    with pytest.raises(ExpertPackageError):
        validate_expert_package(
            _package(
                manifest=_manifest(skills=[card]),
                skills={"..": SkillPackage(skill_md="x")},
            )
        )


def test_workflow_named_by_the_manifest_but_missing_is_refused():
    with pytest.raises(ExpertPackageError, match="workflows"):
        validate_expert_package(_package(workflows={}))


def test_workflow_not_named_by_the_manifest_is_refused():
    with pytest.raises(ExpertPackageError, match="workflows"):
        validate_expert_package(
            _package(
                workflows={
                    "workflows/01-daily-digest.json": b"{}",
                    "memory/notes.json": b"{}",
                }
            )
        )


def test_workflow_path_outside_the_workflows_folder_is_refused():
    entry = _workflow("memory/notes.json")
    with pytest.raises(ExpertPackageError, match="path"):
        validate_expert_package(
            _package(
                manifest=_manifest(workflows=[entry]),
                workflows={"memory/notes.json": b"{}"},
            )
        )


def test_a_skill_over_its_own_limits_is_refused_as_over_limit():
    huge = SkillPackage(
        skill_md="x",
        files=[SkillFile(relative_path="big.bin", content=b"0" * (3 * 1024 * 1024))],
    )
    with pytest.raises(ExpertPackageError) as caught:
        validate_expert_package(_package(skills={"write-briefs": huge}))
    assert caught.value.over_limit


def test_package_over_the_size_cap_is_refused_as_over_limit():
    big_graph = b"0" * (MAX_PACKAGE_BYTES + 1)
    with pytest.raises(ExpertPackageError) as caught:
        validate_expert_package(
            _package(workflows={"workflows/01-daily-digest.json": big_graph})
        )
    assert caught.value.over_limit


def test_a_malformed_package_is_not_over_limit():
    with pytest.raises(ExpertPackageError) as caught:
        validate_expert_package(_package(skills={}))
    assert not caught.value.over_limit


def test_manifest_is_never_a_member_the_manifest_names():
    entry = _workflow(MANIFEST_PATH)
    with pytest.raises(ExpertPackageError):
        validate_expert_package(
            _package(
                manifest=_manifest(workflows=[entry]),
                workflows={MANIFEST_PATH: b"{}"},
            )
        )


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Ada Lovelace", "ada-lovelace.expert.zip"),
        ("Zoë's  Research/Desk!", "zoes-research-desk.expert.zip"),
        ("研究", "expert.expert.zip"),
        ("", "expert.expert.zip"),
    ],
)
def test_package_filename(name: str, expected: str):
    assert package_filename(name) == expected


def test_package_filename_is_bounded():
    assert len(package_filename("a" * 500)) <= 80


@pytest.mark.parametrize(
    "index, name, expected",
    [
        (0, "Daily digest", "workflows/01-daily-digest.json"),
        (11, "Weekly — Report", "workflows/12-weekly-report.json"),
        (2, None, "workflows/03-workflow.json"),
        (3, "???", "workflows/04-workflow.json"),
    ],
)
def test_workflow_path(index: int, name: str | None, expected: str):
    assert workflow_path(index, name) == expected


def test_workflows_with_the_same_name_get_distinct_paths():
    assert workflow_path(0, "Digest") != workflow_path(1, "Digest")
