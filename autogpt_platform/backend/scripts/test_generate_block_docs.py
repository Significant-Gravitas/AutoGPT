#!/usr/bin/env python3
"""Tests for the block documentation generator."""

import pytest

from scripts.generate_block_docs import (
    BlockDoc,
    OrphanedManualContentError,
    class_name_to_display_name,
    collect_orphaned_manual_sections,
    extract_manual_content,
    file_path_to_title,
    find_block_manual_content,
    find_orphaned_manual_sections,
    generate_anchor,
    generate_overview_table,
    parse_rekey_args,
    type_to_readable,
    write_block_docs,
)


class TestFilePathToTitle:
    @pytest.mark.parametrize(
        ("file_path", "expected"),
        [
            ("dataforb2b/enrich.md", "DataForB2B Enrich"),
            ("allquiet/on_call.md", "All Quiet On Call"),
            ("stripe_link/mpp.md", "Stripe Link MPP"),
            ("stripe/triggers.md", "Stripe Triggers"),
        ],
    )
    def test_integration_title(self, file_path: str, expected: str):
        assert file_path_to_title(file_path) == expected


class TestClassNameToDisplayName:
    """Tests for class_name_to_display_name function."""

    def test_simple_block_name(self):
        assert class_name_to_display_name("PrintBlock") == "Print"

    def test_multi_word_block_name(self):
        assert class_name_to_display_name("GetWeatherBlock") == "Get Weather"

    def test_consecutive_capitals(self):
        assert class_name_to_display_name("HTTPRequestBlock") == "HTTP Request"

    def test_ai_prefix(self):
        assert class_name_to_display_name("AIConditionBlock") == "AI Condition"

    def test_no_block_suffix(self):
        assert class_name_to_display_name("SomeClass") == "Some Class"


class TestTypeToReadable:
    """Tests for type_to_readable function."""

    def test_string_type(self):
        assert type_to_readable({"type": "string"}) == "str"

    def test_integer_type(self):
        assert type_to_readable({"type": "integer"}) == "int"

    def test_number_type(self):
        assert type_to_readable({"type": "number"}) == "float"

    def test_boolean_type(self):
        assert type_to_readable({"type": "boolean"}) == "bool"

    def test_array_type(self):
        result = type_to_readable({"type": "array", "items": {"type": "string"}})
        assert result == "List[str]"

    def test_object_type(self):
        result = type_to_readable({"type": "object", "title": "MyModel"})
        assert result == "MyModel"

    def test_anyof_with_null(self):
        result = type_to_readable({"anyOf": [{"type": "string"}, {"type": "null"}]})
        assert result == "str"

    def test_anyof_multiple_types(self):
        result = type_to_readable({"anyOf": [{"type": "string"}, {"type": "integer"}]})
        assert result == "str | int"

    def test_enum_type(self):
        result = type_to_readable(
            {"type": "string", "enum": ["option1", "option2", "option3"]}
        )
        assert result == '"option1" | "option2" | "option3"'

    def test_none_input(self):
        assert type_to_readable(None) == "Any"

    def test_non_dict_input(self):
        assert type_to_readable("string") == "string"


class TestExtractManualContent:
    """Tests for extract_manual_content function."""

    def test_extract_how_it_works(self):
        content = """
### How it works
<!-- MANUAL: how_it_works -->
This is how it works.
<!-- END MANUAL -->
"""
        result = extract_manual_content(content)
        assert result == {"how_it_works": "This is how it works."}

    def test_extract_use_case(self):
        content = """
### Possible use case
<!-- MANUAL: use_case -->
Example use case here.
<!-- END MANUAL -->
"""
        result = extract_manual_content(content)
        assert result == {"use_case": "Example use case here."}

    def test_extract_multiple_sections(self):
        content = """
<!-- MANUAL: how_it_works -->
How it works content.
<!-- END MANUAL -->

<!-- MANUAL: use_case -->
Use case content.
<!-- END MANUAL -->
"""
        result = extract_manual_content(content)
        assert result == {
            "how_it_works": "How it works content.",
            "use_case": "Use case content.",
        }

    def test_empty_content(self):
        result = extract_manual_content("")
        assert result == {}

    def test_no_markers(self):
        result = extract_manual_content("Some content without markers")
        assert result == {}


class TestGenerateAnchor:
    """Tests for generate_anchor function."""

    def test_simple_name(self):
        assert generate_anchor("Print") == "print"

    def test_multi_word_name(self):
        assert generate_anchor("Get Weather") == "get-weather"

    def test_name_with_parentheses(self):
        assert generate_anchor("Something (Optional)") == "something-optional"

    def test_already_lowercase(self):
        assert generate_anchor("already lowercase") == "already-lowercase"


class TestIntegration:
    """Integration tests that require block loading."""

    def test_load_blocks(self):
        """Test that blocks can be loaded successfully."""
        import logging
        import sys
        from pathlib import Path

        logging.disable(logging.CRITICAL)
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from scripts.generate_block_docs import load_all_blocks_for_docs

        blocks = load_all_blocks_for_docs()
        assert len(blocks) > 0, "Should load at least one block"

    def test_block_doc_has_required_fields(self):
        """Test that extracted block docs have required fields."""
        import logging
        import sys
        from pathlib import Path

        logging.disable(logging.CRITICAL)
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from scripts.generate_block_docs import load_all_blocks_for_docs

        blocks = load_all_blocks_for_docs()
        block = blocks[0]

        assert hasattr(block, "id")
        assert hasattr(block, "name")
        assert hasattr(block, "description")
        assert hasattr(block, "categories")
        assert hasattr(block, "inputs")
        assert hasattr(block, "outputs")

    def test_file_mapping_is_deterministic(self):
        """Test that file mapping produces consistent results."""
        import logging
        import sys
        from pathlib import Path

        logging.disable(logging.CRITICAL)
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from scripts.generate_block_docs import (
            get_block_file_mapping,
            load_all_blocks_for_docs,
        )

        # Load blocks twice and compare mappings
        blocks1 = load_all_blocks_for_docs()
        blocks2 = load_all_blocks_for_docs()

        mapping1 = get_block_file_mapping(blocks1)
        mapping2 = get_block_file_mapping(blocks2)

        # Check same files are generated
        assert set(mapping1.keys()) == set(mapping2.keys())

        # Check same block counts per file
        for file_path in mapping1:
            assert len(mapping1[file_path]) == len(mapping2[file_path])


class TestOverviewGuideLinks:
    """The overview's 'Creating Your Own Blocks' guide links must point at the
    live docs host, not the dead docs.agpt.co one (OPEN-3209)."""

    def test_no_dead_docs_host(self):
        overview = generate_overview_table([])
        assert "docs.agpt.co" not in overview

    def test_links_use_agpt_docs_host(self):
        overview = generate_overview_table([])
        assert "(https://agpt.co/docs/platform/new-blocks)" in overview
        assert "(https://agpt.co/docs/platform/block-sdk-guide)" in overview


class TestOrphanedManualSections:
    """A block rename must not let the generator overwrite prose (OPEN-3458)."""

    def test_renamed_block_orphans_its_prose(self):
        orphans = find_orphaned_manual_sections(
            RENAMED_DOC, ["AllQuiet Create Incident"]
        )
        assert orphans == [("All Quiet Create Incident", ["how_it_works", "use_case"])]

    def test_matching_heading_is_not_orphaned(self):
        orphans = find_orphaned_manual_sections(
            RENAMED_DOC, ["All Quiet Create Incident"]
        )
        assert orphans == []

    def test_rekey_rescues_the_prose(self):
        orphans = find_orphaned_manual_sections(
            RENAMED_DOC,
            ["AllQuiet Create Incident"],
            {"All Quiet Create Incident": "AllQuiet Create Incident"},
        )
        assert orphans == []

    def test_placeholders_are_not_worth_saving(self):
        doc = RENAMED_DOC.replace(
            "Posts to the incident endpoint.", "_Add technical explanation here._"
        ).replace(
            "A triage agent pages the responder.",
            "_Add practical use case examples here._",
        )
        assert find_orphaned_manual_sections(doc, ["AllQuiet Create Incident"]) == []

    def test_file_level_sections_belong_to_no_block(self):
        doc = RENAMED_DOC + (
            "\n<!-- MANUAL: additional_content -->\nFile-level notes.\n<!-- END MANUAL -->\n"
        )
        orphans = find_orphaned_manual_sections(doc, ["AllQuiet Create Incident"])
        assert orphans == [("All Quiet Create Incident", ["how_it_works", "use_case"])]

    def test_collect_reports_the_file(self, tmp_path):
        doc_path = tmp_path / "allquiet" / "incidents.md"
        doc_path.parent.mkdir()
        doc_path.write_text(RENAMED_DOC)

        orphans = collect_orphaned_manual_sections(
            tmp_path,
            {"allquiet/incidents.md": [make_block("AllQuiet Create Incident")]},
        )
        assert orphans == {
            "allquiet/incidents.md": [
                ("All Quiet Create Incident", ["how_it_works", "use_case"])
            ]
        }


class TestFindBlockManualContent:
    def test_reads_prose_under_the_current_heading(self):
        manual = find_block_manual_content(RENAMED_DOC, "All Quiet Create Incident")
        assert manual["how_it_works"] == "Posts to the incident endpoint."

    def test_falls_back_to_a_rekeyed_heading(self):
        manual = find_block_manual_content(
            RENAMED_DOC,
            "AllQuiet Create Incident",
            {"All Quiet Create Incident": "AllQuiet Create Incident"},
        )
        assert manual["how_it_works"] == "Posts to the incident endpoint."

    def test_unknown_heading_yields_nothing(self):
        assert find_block_manual_content(RENAMED_DOC, "AllQuiet Create Incident") == {}


class TestWriteRefusesToDropProse:
    def test_raises_and_leaves_the_file_untouched(self, tmp_path):
        doc_path = tmp_path / "allquiet" / "incidents.md"
        doc_path.parent.mkdir()
        doc_path.write_text(RENAMED_DOC)

        with pytest.raises(OrphanedManualContentError) as excinfo:
            write_block_docs(tmp_path, [make_block("AllQuiet Create Incident")])

        assert "All Quiet Create Incident" in str(excinfo.value)
        assert "--rekey" in str(excinfo.value)
        assert doc_path.read_text() == RENAMED_DOC

    def test_rekey_carries_the_prose_over(self, tmp_path):
        doc_path = tmp_path / "allquiet" / "incidents.md"
        doc_path.parent.mkdir()
        doc_path.write_text(RENAMED_DOC)

        write_block_docs(
            tmp_path,
            [make_block("AllQuiet Create Incident")],
            rename_map={"All Quiet Create Incident": "AllQuiet Create Incident"},
        )

        written = doc_path.read_text()
        assert "## AllQuiet Create Incident" in written
        assert "Posts to the incident endpoint." in written
        assert "_Add technical explanation here._" not in written

    def test_allow_orphaned_manual_is_an_explicit_opt_in(self, tmp_path):
        doc_path = tmp_path / "allquiet" / "incidents.md"
        doc_path.parent.mkdir()
        doc_path.write_text(RENAMED_DOC)

        write_block_docs(
            tmp_path,
            [make_block("AllQuiet Create Incident")],
            allow_orphaned_manual=True,
        )

        assert "_Add technical explanation here._" in doc_path.read_text()


class TestParseRekeyArgs:
    def test_parses_pairs(self):
        assert parse_rekey_args(["All Quiet Foo=AllQuiet Foo"]) == {
            "All Quiet Foo": "AllQuiet Foo"
        }

    def test_strips_surrounding_whitespace(self):
        assert parse_rekey_args([" Old Name = New Name "]) == {"Old Name": "New Name"}

    @pytest.mark.parametrize("arg", ["no separator", "=New Name", "Old Name="])
    def test_rejects_malformed_pairs(self, arg: str):
        with pytest.raises(ValueError):
            parse_rekey_args([arg])


RENAMED_DOC = """# All Quiet Incidents
<!-- MANUAL: file_description -->
Blocks that create All Quiet incidents.
<!-- END MANUAL -->

## All Quiet Create Incident

### What it is
Creates an incident.

### How it works
<!-- MANUAL: how_it_works -->
Posts to the incident endpoint.
<!-- END MANUAL -->

### Possible use case
<!-- MANUAL: use_case -->
A triage agent pages the responder.
<!-- END MANUAL -->

---
"""


def make_block(name: str) -> BlockDoc:
    return BlockDoc(
        id="test-block-id",
        name=name,
        class_name="AllQuietCreateIncidentBlock",
        description="Creates an incident.",
        categories=["COMMUNICATION"],
        category_descriptions={},
        inputs=[],
        outputs=[],
        block_type="Standard",
        source_file="blocks/allquiet/incidents.py",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
