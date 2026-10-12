"""The roster loader: one YAML per expert, validated the way the seed needs."""

import pathlib

import pytest

from backend.api.features.experts.roster import RosterError, load_roster, parse_expert

MINIMAL = """\
key: maria
name: Maria
role: SEO & Content
job_title: SEO Content Manager
tagline: Takes a keyword from brief to draft.
avatar_url: /avatars/maria.webp
categories: [marketing, content]
bio: I write briefs.
identity: |
  You are Maria.

  You write briefs.
voice_preferences: Clear and direct.
voice_samples:
  - label: Punchy
    text: Stop guessing.
boundaries: Never invent statistics.
day_one:
  - title: A brief before the draft
    description: Turns a keyword into a brief.
    timing: day 1
preloads:
  - slug: automated-blog-writer
    cron: null
routines:
  - key: content-pipeline-check
    title: What ships this week
    prompt: Read the calendar.
    crons: ["H 9 * * 1-5"]
    asks: ["Where is the calendar?"]
    session_mode: THREAD
skills: [brand-voice-guide, seo-content-brief]
"""


def test_parse_expert_maps_every_field_onto_the_roster_entry():
    entry = parse_expert(MINIMAL, expected_key="maria")

    assert entry["key"] == "maria"
    assert entry["name"] == "Maria"
    assert entry["bundled_skills"] == ["brand-voice-guide", "seo-content-brief"]
    assert entry["categories"] == ["marketing", "content"]
    assert entry["identity"] == "You are Maria.\n\nYou write briefs.\n"
    assert [s.label for s in entry["voice_samples"]] == ["Punchy"]
    assert entry["day_one"][0].title == "A brief before the draft"
    assert entry["preloads"] == [{"slug": "automated-blog-writer", "cron": None}]
    assert entry["routines"][0]["key"] == "content-pipeline-check"
    assert entry["routines"][0]["session_mode"] == "THREAD"


def test_a_key_that_disagrees_with_the_file_name_fails():
    with pytest.raises(RosterError, match="named for 'alex'"):
        parse_expert(MINIMAL, expected_key="alex")


@pytest.mark.parametrize(
    "bad, message",
    [
        ("categories: [nonsense]", "category"),
        ("skills: [a, a]", "bundled twice"),
        ("skills: [Not A Slug]", "not a valid skill slug"),
        ("extra_field: 1", "extra_field"),
        ("key: Maria", "key"),
    ],
)
def test_malformed_expert_files_fail(bad: str, message: str):
    text = "\n".join(
        line
        for line in MINIMAL.splitlines()
        if not line.startswith(bad.split(":")[0] + ":")
    )
    if bad.startswith("skills: [a, a]") or bad.startswith("skills: [Not"):
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("skills:")
        )
    with pytest.raises(RosterError, match=message):
        parse_expert(text + "\n" + bad + "\n")


def test_load_roster_reads_every_file_sorted_by_key(tmp_path: pathlib.Path):
    experts = tmp_path / "experts"
    experts.mkdir()
    (experts / "maria.yml").write_text(MINIMAL, encoding="utf-8")
    (experts / "alex.yml").write_text(
        MINIMAL.replace("key: maria", "key: alex").replace("name: Maria", "name: Alex"),
        encoding="utf-8",
    )

    roster = load_roster(tmp_path)

    assert [entry["key"] for entry in roster] == ["alex", "maria"]


def test_load_roster_refuses_two_experts_with_one_display_name(
    tmp_path: pathlib.Path,
):
    experts = tmp_path / "experts"
    experts.mkdir()
    (experts / "maria.yml").write_text(MINIMAL, encoding="utf-8")
    (experts / "maria-2.yml").write_text(
        MINIMAL.replace("key: maria", "key: maria-2"), encoding="utf-8"
    )

    with pytest.raises(RosterError, match="share a display name"):
        load_roster(tmp_path)


def test_load_roster_needs_the_experts_directory(tmp_path: pathlib.Path):
    with pytest.raises(RosterError, match="experts/ is missing"):
        load_roster(tmp_path)
