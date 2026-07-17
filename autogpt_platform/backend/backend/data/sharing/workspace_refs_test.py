"""Unit tests for the workspace-file reference extractor.

The chat-share PR added two alternation branches to ``_FILE_ID_RE``
(``file_id=<uuid>`` from ``[Attached files]`` blocks and
``"file_id":"<uuid>"`` from JSON tool output) on top of the existing
``workspace://<uuid>`` leaf scan.  ``shared_execution_file_test.py``
exercises the original ``workspace://`` path; this file pins the two
new patterns and the cross-pattern de-dup behaviour.
"""

from __future__ import annotations

from backend.data.sharing.workspace_refs import (
    cut_lands_inside_artifact_link,
    extract_artifact_links,
    extract_workspace_file_ids,
)

UUID_A = "11111111-2222-3333-4444-555555555555"
UUID_B = "66666666-7777-8888-9999-aaaaaaaaaaaa"


class TestFileIdEqualsPattern:
    """``file_id=<uuid>`` shape — used inside ``[Attached files]`` blocks
    the copilot appends to user messages."""

    def test_extracts_from_attached_files_block(self):
        content = f"[Attached files]\nfile_id={UUID_A} my-image.png"
        assert extract_workspace_file_ids(content) == {UUID_A}

    def test_extracts_multiple_attached_files(self):
        content = (
            "[Attached files]\n"
            f"file_id={UUID_A} image-1.png\n"
            f"file_id={UUID_B} image-2.jpg"
        )
        assert extract_workspace_file_ids(content) == {UUID_A, UUID_B}

    def test_rejects_non_uuid_after_file_id_equals(self):
        # The regex demands the exact UUID layout; "not-a-uuid" must not
        # be admitted to the allowlist.
        content = "[Attached files]\nfile_id=not-a-uuid"
        assert extract_workspace_file_ids(content) == set()

    def test_does_not_match_uuid_without_file_id_prefix(self):
        # Bare UUIDs in narrative prose must not be pulled out as
        # references — only the anchored ``file_id=`` form counts.
        content = f"Here is a uuid in the wild: {UUID_A}"
        assert extract_workspace_file_ids(content) == set()


class TestFileIdJsonPattern:
    """``"file_id":"<uuid>"`` shape — JSON-serialised tool output."""

    def test_extracts_from_json_blob(self):
        content = (
            '{"type":"workspace_file_written",'
            f'"file_id":"{UUID_A}",'
            '"path":"out.png"}'
        )
        assert extract_workspace_file_ids(content) == {UUID_A}

    def test_extracts_with_whitespace_around_colon(self):
        # The regex tolerates ``"file_id" : "..."`` formatting (per
        # ``\s*:\s*`` in the pattern).  Pin the tolerance.
        content = f'{{"file_id" : "{UUID_A}"}}'
        assert extract_workspace_file_ids(content) == {UUID_A}

    def test_rejects_single_quoted_file_id(self):
        # Python repr() / single-quoted forms are not JSON and should
        # not match the JSON-anchored pattern.
        content = f"{{'file_id': '{UUID_A}'}}"
        assert extract_workspace_file_ids(content) == set()


class TestMixedInput:
    """All three reference shapes coexist in one pass."""

    def test_collects_all_three_shapes_and_dedupes(self):
        # workspace:// leaf (in a list), file_id= substring (in narrative),
        # and "file_id":"" JSON token.  Same UUID in two of them must
        # dedupe; distinct UUIDs all surface.
        UUID_C = "cccccccc-dddd-eeee-ffff-000000000000"
        outputs = {
            "image": [f"workspace://{UUID_A}#image/png"],
            "notes": f"[Attached files]\nfile_id={UUID_B} note.txt",
            "tool_response": (
                '{"type":"workspace_file_written",' f'"file_id":"{UUID_C}"}}'
            ),
            # Same UUID via a second shape — must dedupe.
            "duplicate": f"file_id={UUID_A}",
        }
        assert extract_workspace_file_ids(outputs) == {UUID_A, UUID_B, UUID_C}

    def test_ignores_non_string_leaves(self):
        # Numbers, booleans, None shouldn't crash the scan.
        outputs = {
            "count": 3,
            "ok": True,
            "missing": None,
            "ref": [f"workspace://{UUID_A}"],
        }
        assert extract_workspace_file_ids(outputs) == {UUID_A}


class TestExtractArtifactLinks:
    """``[name](workspace://id#mime)`` markdown parsing used by chat-bot
    delivery to peel artifacts out of assistant text."""

    def test_no_artifacts_returns_text_unchanged(self):
        text, artifacts = extract_artifact_links("hello world")
        assert text == "hello world"
        assert artifacts == []

    def test_strips_artifact_markdown(self):
        text, artifacts = extract_artifact_links(
            "Here's your result: [chart.png](workspace://abc-123#image/png)"
        )
        assert text == "Here's your result:"
        assert len(artifacts) == 1
        assert artifacts[0].file_id == "abc-123"
        assert artifacts[0].display_name == "chart.png"
        assert artifacts[0].mime_hint == "image/png"

    def test_handles_artifact_with_no_mime_fragment(self):
        text, artifacts = extract_artifact_links("[doc.pdf](workspace://xyz)")
        assert text == ""
        assert artifacts[0].file_id == "xyz"
        assert artifacts[0].mime_hint is None

    def test_strips_image_embed_without_leaving_bang(self):
        # The prompt tells the LLM to emit images as `![name](workspace://...)`;
        # the leading `!` must be consumed so no marker is left behind.
        text, artifacts = extract_artifact_links(
            "Result: ![chart](workspace://abc-123#image/png)"
        )
        assert text == "Result:"
        assert "!" not in text
        assert artifacts[0].file_id == "abc-123"
        assert artifacts[0].display_name == "chart"
        assert artifacts[0].mime_hint == "image/png"

    def test_handles_multiple_artifacts(self):
        text, artifacts = extract_artifact_links(
            "First [a.png](workspace://1#image/png) "
            "then [b.csv](workspace://2#text/csv)"
        )
        assert "workspace://" not in text
        assert [a.file_id for a in artifacts] == ["1", "2"]

    def test_collapses_blank_lines_left_behind(self):
        text, _ = extract_artifact_links(
            "Above paragraph.\n\n[file.png](workspace://abc)\n\nBelow paragraph."
        )
        assert "\n\n\n" not in text
        assert "Above paragraph." in text
        assert "Below paragraph." in text


class TestCutLandsInsideArtifactLink:
    def test_cut_inside_link_pulls_back_to_start(self):
        text = "see [chart.png](workspace://abc-123#image/png) now"
        start = text.index("[chart.png]")
        mid = text.index("workspace://")  # a cut landing inside the link
        assert cut_lands_inside_artifact_link(text, mid) == start

    def test_cut_outside_link_is_unchanged(self):
        text = "see [chart.png](workspace://abc) now"
        end = len(text) - 1
        assert cut_lands_inside_artifact_link(text, end) == end

    def test_cut_at_link_boundaries_is_unchanged(self):
        text = "x [a.png](workspace://a) y"
        start = text.index("[a.png]")
        after = text.index(") y") + 1
        # Boundaries are exclusive — exactly at start/end keeps the link whole.
        assert cut_lands_inside_artifact_link(text, start) == start
        assert cut_lands_inside_artifact_link(text, after) == after

    def test_cut_inside_leading_link_returns_end_not_zero(self):
        # A link at index 0 can't pull the cut back to its start — that yields
        # an empty chunk and stalls the stream. Cut at the link's end instead so
        # the whole link is emitted and the buffer advances.
        link = "[report.txt](workspace://abc-123#text/plain)"
        text = link + " trailing text"
        cut = link.index("workspace://")  # inside the leading link
        assert cut_lands_inside_artifact_link(text, cut) == len(link)
