"""Tests for dynamic field utilities."""

from backend.util.dynamic_fields import (
    extract_base_field_name,
    get_dynamic_field_description,
    group_fields_by_base_name,
    is_dynamic_field,
)


class TestExtractBaseFieldName:
    """Test extracting base field names from dynamic field names."""

    def test_extract_dict_field(self):
        """Test extracting base name from dictionary fields."""
        assert extract_base_field_name("values_#_name") == "values"
        assert extract_base_field_name("data_#_key1_#_key2") == "data"
        assert extract_base_field_name("config_#_database_#_host") == "config"

    def test_extract_list_field(self):
        """Test extracting base name from list fields."""
        assert extract_base_field_name("items_$_0") == "items"
        assert extract_base_field_name("results_$_5_$_10") == "results"
        assert extract_base_field_name("nested_$_0_$_1_$_2") == "nested"

    def test_extract_object_field(self):
        """Test extracting base name from object fields."""
        assert extract_base_field_name("user_@_name") == "user"
        assert extract_base_field_name("response_@_data_@_items") == "response"
        assert extract_base_field_name("obj_@_attr1_@_attr2") == "obj"

    def test_extract_mixed_fields(self):
        """Test extracting base name from mixed dynamic fields."""
        assert extract_base_field_name("data_$_0_#_key") == "data"
        assert extract_base_field_name("items_#_user_@_name") == "items"
        assert extract_base_field_name("complex_$_0_@_attr_#_key") == "complex"

    def test_extract_regular_field(self):
        """Test extracting base name from regular (non-dynamic) fields."""
        assert extract_base_field_name("regular_field") == "regular_field"
        assert extract_base_field_name("simple") == "simple"
        assert extract_base_field_name("") == ""

    def test_extract_field_with_underscores(self):
        """Test fields with regular underscores (not dynamic delimiters)."""
        assert extract_base_field_name("field_name_here") == "field_name_here"
        assert extract_base_field_name("my_field_#_key") == "my_field"


class TestIsDynamicField:
    """Test identifying dynamic fields."""

    def test_is_dynamic_dict_field(self):
        """Test identifying dictionary dynamic fields."""
        assert is_dynamic_field("values_#_name") is True
        assert is_dynamic_field("data_#_key1_#_key2") is True

    def test_is_dynamic_list_field(self):
        """Test identifying list dynamic fields."""
        assert is_dynamic_field("items_$_0") is True
        assert is_dynamic_field("results_$_5_$_10") is True

    def test_is_dynamic_object_field(self):
        """Test identifying object dynamic fields."""
        assert is_dynamic_field("user_@_name") is True
        assert is_dynamic_field("response_@_data_@_items") is True

    def test_is_dynamic_mixed_field(self):
        """Test identifying mixed dynamic fields."""
        assert is_dynamic_field("data_$_0_#_key") is True
        assert is_dynamic_field("items_#_user_@_name") is True

    def test_is_not_dynamic_field(self):
        """Test identifying non-dynamic fields."""
        assert is_dynamic_field("regular_field") is False
        assert is_dynamic_field("field_name_here") is False
        assert is_dynamic_field("simple") is False
        assert is_dynamic_field("") is False


class TestGetDynamicFieldDescription:
    """Test generating descriptions for dynamic fields."""

    def test_dict_field_description(self):
        """Test descriptions for dictionary fields."""
        desc = get_dynamic_field_description("values", "values_#_name")
        assert "Dictionary value for values['name']" == desc

        desc = get_dynamic_field_description("config", "config_#_database")
        assert "Dictionary value for config['database']" == desc

    def test_list_field_description(self):
        """Test descriptions for list fields."""
        desc = get_dynamic_field_description("items", "items_$_0")
        assert "List item for items[0]" == desc

        desc = get_dynamic_field_description("results", "results_$_5")
        assert "List item for results[5]" == desc

    def test_object_field_description(self):
        """Test descriptions for object fields."""
        desc = get_dynamic_field_description("user", "user_@_name")
        assert "Object attribute for user.name" == desc

        desc = get_dynamic_field_description("response", "response_@_data")
        assert "Object attribute for response.data" == desc

    def test_fallback_description(self):
        """Test fallback description for non-dynamic fields."""
        desc = get_dynamic_field_description("field", "field")
        assert "Dynamic value for field" == desc


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
