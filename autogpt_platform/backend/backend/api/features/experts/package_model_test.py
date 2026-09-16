"""Tests for the expert manifest: what it may say, and above all what it may
not — memory, conversations and workspace files have no field to land in."""

import json

import pytest
from pydantic import ValidationError

from backend.api.features.experts.package_model import (
    ExpertManifest,
    ExpertPackage,
    PackagedAvatar,
    PackagedIdentity,
    PackagedWorkflow,
    expert_slug,
    manifest_json,
)

IDENTITY = PackagedIdentity(name="Maria")


def _manifest_dict(**extra: object) -> dict[str, object]:
    return {"format_version": 1, "identity": {"name": "Maria"}, **extra}


@pytest.mark.parametrize("key", ["memory", "conversations", "workspace", "notes"])
def test_a_key_the_manifest_does_not_define_is_refused(key: str):
    """The whole portability promise: a field the format has no name for
    cannot be smuggled through as an extra."""
    with pytest.raises(ValidationError, match="Extra inputs"):
        ExpertManifest.model_validate(_manifest_dict(**{key: ["anything"]}))


def test_an_unknown_key_inside_a_nested_section_is_refused_too():
    with pytest.raises(ValidationError, match="Extra inputs"):
        ExpertManifest.model_validate(
            _manifest_dict(soul={"identity": "x", "memory": "y"})
        )


def test_day_one_sits_beside_the_soul_rather_than_inside_it():
    """A template's day-one list is its creator's copy, not something the
    expert is told about itself, and later PRs read manifest.day_one."""
    manifest = ExpertManifest.model_validate(
        _manifest_dict(day_one=[{"title": "Wire up the inbox"}])
    )
    assert manifest.day_one[0].title == "Wire up the inbox"
    with pytest.raises(ValidationError, match="Extra inputs"):
        ExpertManifest.model_validate(
            _manifest_dict(soul={"day_one": [{"title": "Wire up the inbox"}]})
        )


def test_a_tool_profile_rides_along_untouched():
    """Its shape belongs to the copilot; a schema here would refuse a package
    written by a newer one, so it is carried as opaque JSON."""
    profile = {"enabled": ["search"], "limits": {"search": 3}}
    manifest = ExpertManifest.model_validate(_manifest_dict(tool_profile=profile))
    assert manifest.tool_profile == profile
    assert json.loads(manifest_json(manifest))["tool_profile"] == profile


def test_a_future_format_version_is_refused():
    with pytest.raises(ValidationError):
        ExpertManifest.model_validate(_manifest_dict(format_version=2))


def test_a_workflow_with_neither_a_store_ref_nor_a_graph_is_refused():
    """An entry naming nothing importable would arrive as a workflow the new
    owner can never run."""
    with pytest.raises(ValidationError, match="store listing"):
        PackagedWorkflow(name="Digest")


def test_a_workflow_with_only_a_store_reference_is_accepted():
    workflow = PackagedWorkflow(name="Digest", store_listing_version_id="ver-1")
    assert workflow.graph is None


def test_duplicate_skill_slugs_are_refused():
    with pytest.raises(ValidationError, match="twice"):
        ExpertManifest.model_validate(
            _manifest_dict(
                skills=[
                    {"slug": "research", "name": "Research"},
                    {"slug": "research", "name": "Research again"},
                ]
            )
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "file"},
        {"kind": "url"},
        {"kind": "file", "url": "https://cdn.example/a.png"},
        {"kind": "url", "path": "avatar.png"},
        {"kind": "file", "path": "avatar.png", "url": "https://cdn.example/a.png"},
    ],
)
def test_an_avatar_must_carry_exactly_the_source_its_kind_names(payload: dict):
    with pytest.raises(ValidationError):
        PackagedAvatar.model_validate(payload)


def test_an_avatar_file_must_be_a_supported_image_name():
    with pytest.raises(ValidationError, match="avatar"):
        PackagedAvatar(kind="file", path="avatar.bmp")


def test_an_avatar_url_must_be_https_or_site_relative():
    with pytest.raises(ValidationError):
        PackagedAvatar(kind="url", url="javascript:alert(1)")
    assert PackagedAvatar(kind="url", url="/experts/maria.svg").url


def test_the_manifest_is_written_sorted_so_two_exports_match():
    one = manifest_json(ExpertManifest(identity=IDENTITY))
    other = manifest_json(ExpertManifest.model_validate_json(one))
    assert one == other
    assert one.index(b'"format_version"') < one.index(b'"identity"')


@pytest.mark.parametrize(
    "name, slug",
    [
        ("Maria Ops", "maria-ops"),
        ("  Ada  ", "ada"),
        ("!!!", "expert"),
        ("Zoë's #1 Analyst", "zo-s-1-analyst"),
    ],
)
def test_the_filename_slug_comes_from_the_name(name: str, slug: str):
    assert expert_slug(name) == slug


def test_a_packages_size_counts_its_manifest_skills_and_avatar():
    package = ExpertPackage(
        manifest=ExpertManifest(identity=IDENTITY), avatar_bytes=b"\x89PNG"
    )
    assert package.size_bytes == len(manifest_json(package.manifest)) + 4
    assert package.slug == "maria"
