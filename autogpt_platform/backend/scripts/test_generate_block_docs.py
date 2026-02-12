#!/usr/bin/env python3
"""Tests for the block documentation generator."""
import pytest

from scripts.generate_block_docs import (
    class_name_to_display_name,
    extract_manual_content,
    generate_anchor,
    type_to_readable,
)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
