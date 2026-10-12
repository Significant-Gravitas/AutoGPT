"""Tests for the display title a listing shows on the marketplace, and for
the package model that crosses the DB manager RPC."""

import pytest

from backend.api.features.store.skill_model import SkillVersionPackage, skill_title
from backend.copilot.tools.skills import SkillFile
from backend.util.json import dumps, loads


@pytest.mark.parametrize(
    ("name", "body", "expected"),
    [
        (
            "seo-content-brief",
            "# SEO content brief\n\nUse this.\n",
            "SEO content brief",
        ),
        ("on-page-seo-audit", "# On-page SEO audit\n", "On-page SEO audit"),
        # A closed ATX heading: the trailing run is a marker, not the title.
        ("seo-playbook", "# SEO Playbook #\n", "SEO Playbook"),
        ("seo-playbook", "# SEO Playbook ###\r\n", "SEO Playbook"),
        (
            "icp-and-positioning",
            "\n\n#   ICP and positioning  \n",
            "ICP and positioning",
        ),
    ],
)
def test_the_authors_heading_is_the_title_acronyms_and_all(name, body, expected):
    assert skill_title(name, body) == expected


@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("brand-voice-guide", "Start with the voice, not the words.\n\n# Later\n"),
        ("brand-voice-guide", "## Not the top heading\n"),
        ("brand-voice-guide", "#no-space-is-not-a-heading\n"),
        # `#` alone is an empty heading; the line under it is a paragraph.
        ("brand-voice-guide", "#\nBrand voice, not a heading\n"),
        ("brand_voice_guide", ""),
    ],
)
def test_a_body_that_does_not_open_with_a_heading_falls_back_to_the_slug(name, body):
    assert skill_title(name, body) == "Brand voice guide"


def test_a_nameless_listing_keeps_its_name_rather_than_rendering_empty():
    assert skill_title("", "") == ""


def test_a_package_with_binary_files_survives_the_rpc_encoding():
    """The copilot executor gets packages over the DB manager RPC as JSON,
    the way FastAPI encodes a response and the client validates it back; a
    font or an image in a package has to come back byte for byte."""
    png = b"\x89PNG\r\n\x1a\n\xff\x00"
    package = SkillVersionPackage(
        version_id="v1",
        listing_id="l1",
        slug="demo",
        version=1,
        package_sha256=None,
        skill_markdown="---\nname: demo\ndescription: d\n---\n",
        files=[
            SkillFile(relative_path="assets/logo.png", content=png),
            SkillFile(
                relative_path="scripts/run.sh",
                content=b"#!/bin/sh\n",
                is_executable=True,
            ),
        ],
    )

    back = SkillVersionPackage.model_validate(loads(dumps({"v1": package}))["v1"])

    assert [(f.relative_path, f.content, f.is_executable) for f in back.files] == [
        ("assets/logo.png", png, False),
        ("scripts/run.sh", b"#!/bin/sh\n", True),
    ]
    assert back.model_dump() == package.model_dump()
