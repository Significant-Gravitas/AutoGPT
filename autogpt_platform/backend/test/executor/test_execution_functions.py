from backend.data.execution import merge_execution_input, parse_execution_output


def test_parse_execution_output():
    # Test case for list extraction
    output = ("result", [10, 20, 30])
    assert parse_execution_output(output, "result_$_1") == 20
    assert parse_execution_output(output, "result_$_3") is None

    # Test case for dictionary extraction
    output = ("config", {"key1": "value1", "key2": "value2"})
    assert parse_execution_output(output, "config_#_key1") == "value1"
    assert parse_execution_output(output, "config_#_key3") is None

    # Test case for object extraction
    class Sample:
        attr1 = "value1"
        attr2 = "value2"

    output = ("object", Sample())
    assert parse_execution_output(output, "object_@_attr1") == "value1"
    assert parse_execution_output(output, "object_@_attr3") is None

    # Test case for direct match
    output = ("direct", "match")
    assert parse_execution_output(output, "direct") == "match"
    assert parse_execution_output(output, "nomatch") is None


def test_merge_execution_input():
    # Test case for merging list inputs
    data = {"list_$_0": "a", "list_$_1": "b", "list_$_3": "d"}
    merged_data = merge_execution_input(data)
    assert merged_data["list"] == ["a", "b", "", "d"]

    # Test case for merging dictionary inputs
    data = {"dict_#_key1": "value1", "dict_#_key2": "value2"}
    merged_data = merge_execution_input(data)
    assert merged_data["dict"] == {"key1": "value1", "key2": "value2"}

    # Test case for merging object inputs
    data = {"object_@_attr1": "value1", "object_@_attr2": "value2"}
    merged_data = merge_execution_input(data)
    assert hasattr(merged_data["object"], "attr1")
    assert hasattr(merged_data["object"], "attr2")
    assert merged_data["object"].attr1 == "value1"
    assert merged_data["object"].attr2 == "value2"

    # Test case for mixed inputs
    data = {"list_$_0": "a", "dict_#_key1": "value1", "object_@_attr1": "value1"}
    merged_data = merge_execution_input(data)
    assert merged_data["list"] == ["a"]
    assert merged_data["dict"] == {"key1": "value1"}
    assert hasattr(merged_data["object"], "attr1")
    assert merged_data["object"].attr1 == "value1"
