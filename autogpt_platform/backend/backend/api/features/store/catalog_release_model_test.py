import pytest

from backend.api.features.store.catalog_release_model import (
    Adoption,
    ReleaseManifest,
    ReleaseSnapshot,
    digest,
)
from backend.api.features.store.catalog_release_state import (
    DatabaseState,
    validate_boundary,
)


def test_release_refuses_assignment_to_retired_package():
    with pytest.raises(ValueError, match="retired"):
        ReleaseManifest.model_validate(
            {
                "schema_version": 1,
                "release_key": "test",
                "catalog_sha256": "a" * 64,
                "packages": [],
                "experts": [{"key": "max", "skills": ["demo"]}],
                "retirements": ["demo"],
            }
        )


def test_adoption_refuses_duplicate_record_ids():
    with pytest.raises(ValueError, match="duplicate"):
        Adoption.model_validate(
            {"skills": {"first": "same-id", "second": "same-id"}, "experts": {}}
        )


def manifest():
    files = [{"path": "SKILL.md", "sha256": "a" * 64, "executable": False}]
    return ReleaseManifest.model_validate(
        {
            "schema_version": 1,
            "release_key": "test",
            "catalog_sha256": "b" * 64,
            "packages": [
                {"slug": "demo", "tree_sha256": digest(files), "files": files}
            ],
            "experts": [{"key": "max", "skills": ["demo"]}],
        }
    )


def state():
    return DatabaseState.model_validate(
        {
            "database_target": "test-local/catalogue-test",
            "active_release_id": None,
            "generation": 0,
            "skills": {
                "demo": {
                    "id": "skill-id",
                    "slug": "demo",
                    "owning_user_id": None,
                    "owning_org_id": None,
                    "active_version_id": "version-id",
                    "active_version_listing_id": "skill-id",
                    "active_version_organization_id": None,
                    "is_deleted": False,
                    "has_approved_version": True,
                    "version_fingerprint": "v",
                    "file_fingerprint": "f",
                }
            },
            "experts": {
                "max": {
                    "id": "expert-id",
                    "owner_user_id": None,
                    "organization_id": None,
                    "team_id": None,
                    "is_template": True,
                    "is_archived": False,
                    "skills": ["skill-id"],
                }
            },
        }
    )


def adoption():
    return Adoption(skills={"demo": "skill-id"}, experts={"max": "expert-id"})


@pytest.mark.parametrize("field", ["owning_user_id", "owning_org_id"])
def test_user_and_org_skill_collisions_are_refused(field):
    data = state().model_dump()
    data["skills"]["demo"][field] = "owner"
    with pytest.raises(ValueError, match="user or organisation"):
        validate_boundary(DatabaseState.model_validate(data), adoption(), manifest())


@pytest.mark.parametrize(
    "field", ["owner_user_id", "organization_id", "team_id", "is_template"]
)
def test_owned_experts_and_hired_copies_are_refused(field):
    data = state().model_dump()
    data["experts"]["max"][field] = False if field == "is_template" else "owner"
    with pytest.raises(ValueError, match="unowned platform template"):
        validate_boundary(DatabaseState.model_validate(data), adoption(), manifest())


def test_same_name_is_not_authority_to_adopt_another_id():
    changed = adoption().model_copy(update={"skills": {"demo": "someone-else"}})
    with pytest.raises(ValueError, match="adopted record ID"):
        validate_boundary(state(), changed, manifest())


def test_expected_absence_detects_new_owned_or_platform_collision():
    changed = adoption().model_copy(update={"skills": {"demo": None}})
    with pytest.raises(ValueError, match="expected absent"):
        validate_boundary(state(), changed, manifest())


def test_missing_package_is_not_implicit_retirement():
    data = state().model_dump()
    data["previous"] = ReleaseSnapshot.model_validate(
        {
            "skills": {"demo": {"listing_id": "skill-id", "version_id": "version-id"}},
            "experts": {"max": {"expert_id": "expert-id", "skills": ["demo"]}},
        }
    ).model_dump()
    removed = manifest().model_copy(update={"packages": [], "experts": []})
    with pytest.raises(ValueError, match="without explicit retirement"):
        validate_boundary(
            DatabaseState.model_validate(data),
            Adoption(skills={"demo": "skill-id"}, experts={}),
            removed,
        )


def test_preview_fingerprint_changes_on_ownership_files_or_order_drift():
    original = state()
    for record, field, value in [
        ("skills", "file_fingerprint", "tampered"),
        ("skills", "owning_user_id", "owner"),
        ("experts", "skills", ["other", "skill-id"]),
    ]:
        data = original.model_dump()
        key = "demo" if record == "skills" else "max"
        data[record][key][field] = value
        assert (
            DatabaseState.model_validate(data).fingerprint() != original.fingerprint()
        )


def test_valid_owned_boundary_is_exactly_adopted_records():
    validate_boundary(state(), adoption(), manifest())


@pytest.mark.parametrize("path", ["../SKILL.md", "/SKILL.md", "a\\b", "a//b"])
def test_manifest_rejects_unsafe_paths(path):
    data = manifest().model_dump()
    files = sorted(
        data["packages"][0]["files"]
        + [{"path": path, "sha256": "c" * 64, "executable": False}],
        key=lambda f: f["path"],
    )
    data["packages"][0].update(files=files, tree_sha256=digest(files))
    with pytest.raises(ValueError, match="unsafe package path"):
        ReleaseManifest.model_validate(data)
