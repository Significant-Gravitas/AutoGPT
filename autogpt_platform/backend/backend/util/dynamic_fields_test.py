"""Tests for dynamic field utilities."""

import pytest

from backend.util.dynamic_fields import (
    extract_base_field_name,
    get_dynamic_field_description,
    group_fields_by_base_name,
    is_dynamic_field,
)


class TestExtractBaseFieldName:
    @pytest.mark.parametrize(
        "field, base",
        [
            ("values_#_name", "values"),
            ("data_#_key1_#_key2", "data"),
            ("config_#_database_#_host", "config"),
            ("items_$_0", "items"),
            ("results_$_5_$_10", "results"),
            ("nested_$_0_$_1_$_2", "nested"),
            ("user_@_name", "user"),
            ("response_@_data_@_items", "response"),
            ("obj_@_attr1_@_attr2", "obj"),
            ("data_$_0_#_key", "data"),
            ("items_#_user_@_name", "items"),
            ("complex_$_0_@_attr_#_key", "complex"),
            ("regular_field", "regular_field"),
            ("simple", "simple"),
            ("", ""),
            # A plain underscore is not a delimiter.
            ("field_name_here", "field_name_here"),
            ("my_field_#_key", "my_field"),
        ],
    )
    def test_extract_base_field_name(self, field, base):
        assert extract_base_field_name(field) == base


class TestIsDynamicField:
    @pytest.mark.parametrize(
        "field, dynamic",
        [
            ("values_#_name", True),
            ("data_#_key1_#_key2", True),
            ("items_$_0", True),
            ("results_$_5_$_10", True),
            ("user_@_name", True),
            ("response_@_data_@_items", True),
            ("data_$_0_#_key", True),
            ("items_#_user_@_name", True),
            ("regular_field", False),
            ("field_name_here", False),
            ("simple", False),
            ("", False),
        ],
    )
    def test_is_dynamic_field(self, field, dynamic):
        assert is_dynamic_field(field) is dynamic


class TestGetDynamicFieldDescription:
    @pytest.mark.parametrize(
        "base, field, description",
        [
            ("values", "values_#_name", "Dictionary value for values['name']"),
            ("config", "config_#_database", "Dictionary value for config['database']"),
            ("items", "items_$_0", "List item for items[0]"),
            ("results", "results_$_5", "List item for results[5]"),
            ("user", "user_@_name", "Object attribute for user.name"),
            ("response", "response_@_data", "Object attribute for response.data"),
            ("field", "field", "Dynamic value for field"),
        ],
    )
    def test_get_dynamic_field_description(self, base, field, description):
        assert get_dynamic_field_description(base, field) == description


class TestGroupFieldsByBaseName:
    """Test grouping fields by their base names."""

    def test_group_mixed_fields(self):
        """Test grouping a mix of dynamic and regular fields."""
        fields = [
            "values_#_name",
            "values_#_age",
            "items_$_0",
            "items_$_1",
            "user_@_email",
            "regular_field",
            "another_field",
        ]

        result = group_fields_by_base_name(fields)

        expected = {
            "values": ["values_#_name", "values_#_age"],
            "items": ["items_$_0", "items_$_1"],
            "user": ["user_@_email"],
            "regular_field": ["regular_field"],
            "another_field": ["another_field"],
        }

        assert result == expected

    def test_group_empty_list(self):
        """Test grouping an empty list."""
        result = group_fields_by_base_name([])
        assert result == {}

    def test_group_single_field(self):
        """Test grouping a single field."""
        result = group_fields_by_base_name(["values_#_name"])
        assert result == {"values": ["values_#_name"]}

    def test_group_complex_dynamic_fields(self):
        """Test grouping complex nested dynamic fields."""
        fields = [
            "data_$_0_#_key1",
            "data_$_0_#_key2",
            "data_$_1_#_key1",
            "other_@_attr",
        ]

        result = group_fields_by_base_name(fields)

        expected = {
            "data": ["data_$_0_#_key1", "data_$_0_#_key2", "data_$_1_#_key1"],
            "other": ["other_@_attr"],
        }

        assert result == expected

    def test_preserve_order(self):
        """Test that field order is preserved within groups."""
        fields = ["values_#_c", "values_#_a", "values_#_b"]
        result = group_fields_by_base_name(fields)

        # Should preserve the original order
        assert result["values"] == ["values_#_c", "values_#_a", "values_#_b"]
