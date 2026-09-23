from backend.copilot.tools.skill_merge import (
    MergeConflict,
    SkillContent,
    merge_skill,
    merge_text,
)

BODY = "# Brief\n\nStep one: read the page.\nStep two: list keywords.\n\nStep three: draft.\nStep four: review.\n"


def _content(**overrides) -> SkillContent:
    fields = {
        "description": "Turn a keyword into a brief.",
        "body": BODY,
        "triggers": ["brief", "keyword"],
        "files": {"references/checklist.md": b"- intent\n"},
    }
    return SkillContent(**{**fields, **overrides})


def test_an_untouched_copy_takes_the_new_version_whole():
    base = _content()
    ours = _content(
        description="Turn a keyword into a ranked brief.",
        body=BODY.replace("draft.", "draft in the brand voice."),
        triggers=["brief", "keyword", "seo brief"],
        files={"references/checklist.md": b"- intent\n- competitors\n"},
    )

    merged = merge_skill(base, base, ours)

    assert merged.content == ours
    assert merged.conflicts == []


def test_an_untouched_field_is_updated_and_an_edited_one_kept():
    base = _content()
    theirs = _content(description="My own description.")
    ours = _content(
        description="Turn a keyword into a ranked brief.",
        triggers=["brief", "keyword", "seo brief"],
    )

    merged = merge_skill(base, theirs, ours)

    assert merged.content.description == "My own description."
    assert merged.content.triggers == ["brief", "keyword", "seo brief"]
    assert [c.field for c in merged.conflicts] == ["description"]


def test_an_edit_the_catalog_did_not_touch_is_kept_without_conflict():
    base = _content()
    theirs = _content(description="My own description.")

    merged = merge_skill(base, theirs, base)

    assert merged.content == theirs
    assert merged.conflicts == []


def test_non_overlapping_text_edits_both_land():
    theirs = BODY.replace("Step one: read the page.", "Step one: read the page twice.")
    ours = BODY.replace("Step four: review.", "Step four: review against the brief.")
    conflicts: list[MergeConflict] = []

    merged = merge_text(BODY, theirs, ours, conflicts)

    assert "Step one: read the page twice.\n" in merged
    assert "Step four: review against the brief.\n" in merged
    assert "Step two: list keywords.\n" in merged
    assert conflicts == []


def test_an_overlapping_edit_keeps_the_owners_lines_and_records_the_conflict():
    theirs = BODY.replace("Step three: draft.", "Step three: draft by hand.")
    ours = BODY.replace("Step three: draft.", "Step three: draft with the template.")
    conflicts: list[MergeConflict] = []

    merged = merge_text(BODY, theirs, ours, conflicts)

    assert merged == theirs
    assert conflicts == [
        MergeConflict(
            field="body",
            base="Step three: draft.\n",
            ours="Step three: draft with the template.\n",
            theirs="Step three: draft by hand.\n",
        )
    ]


def test_the_same_edit_on_both_sides_is_not_a_conflict():
    edited = BODY.replace("Step two", "Step 2")
    conflicts: list[MergeConflict] = []

    assert merge_text(BODY, edited, edited, conflicts) == edited
    assert conflicts == []


def test_lines_added_by_each_side_in_different_places_both_land():
    theirs = BODY + "My own closing note.\n"
    ours = "Read this first.\n" + BODY
    conflicts: list[MergeConflict] = []

    merged = merge_text(BODY, theirs, ours, conflicts)

    assert merged == "Read this first.\n" + BODY + "My own closing note.\n"
    assert conflicts == []


def test_triggers_keep_the_owners_changes_and_take_the_catalogs():
    base = _content(triggers=["brief", "keyword", "outline"])
    theirs = _content(triggers=["brief", "outline", "my trigger"])
    ours = _content(triggers=["brief", "keyword", "seo brief"])

    merged = merge_skill(base, theirs, ours)

    assert merged.content.triggers == ["brief", "my trigger", "seo brief"]


def test_files_follow_the_same_rule_per_path():
    base = _content(files={"a.md": b"a\n", "b.md": b"b\n", "gone.md": b"x\n"})
    theirs = _content(
        files={"a.md": b"mine\n", "b.md": b"b\n", "gone.md": b"x\n", "own.md": b"o\n"}
    )
    ours = _content(files={"a.md": b"new a\n", "b.md": b"new b\n", "added.md": b"n\n"})

    merged = merge_skill(base, theirs, ours)

    assert merged.content.files == {
        "a.md": b"mine\n",
        "b.md": b"new b\n",
        "added.md": b"n\n",
        "own.md": b"o\n",
    }
    assert [c.field for c in merged.conflicts] == ["files/a.md"]
