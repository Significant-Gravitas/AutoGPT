from backend.copilot.capabilities.schema_trim import collapse_nested_objects


def _object(**properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties)}


def test_output_pins_and_their_fields_stay_whole():
    schema = _object(quote=_object(id={"type": "string"}, total={"type": "integer"}))
    assert collapse_nested_objects(schema) == schema


def test_objects_below_a_pin_s_fields_list_their_field_names():
    address = _object(city={"type": "string"}, zip={"type": "string"})
    schema = _object(person=_object(name={"type": "string"}, address=address))

    collapsed = collapse_nested_objects(schema)

    person = collapsed["properties"]["person"]
    assert person["properties"]["name"] == {"type": "string"}
    assert person["properties"]["address"] == {
        "type": "object",
        "field_names": ["city", "zip"],
    }


def test_an_array_pin_keeps_its_item_fields_and_collapses_below_them():
    line = _object(sku={"type": "string"}, qty={"type": "integer"})
    item = _object(id={"type": "string"}, lines={"type": "array", "items": line})
    schema = _object(items={"type": "array", "items": item})

    collapsed = collapse_nested_objects(schema)

    fields = collapsed["properties"]["items"]["items"]["properties"]
    assert fields["id"] == {"type": "string"}
    assert fields["lines"]["items"] == {"type": "object", "field_names": ["sku", "qty"]}


def test_nested_objects_inside_any_of_collapse_too():
    detail = _object(a={"type": "string"})
    schema = _object(
        pin=_object(maybe={"anyOf": [_object(detail=detail), {"type": "null"}]})
    )

    collapsed = collapse_nested_objects(schema)

    maybe = collapsed["properties"]["pin"]["properties"]["maybe"]
    assert maybe["anyOf"][0] == {"type": "object", "field_names": ["detail"]}


def test_a_field_named_properties_is_a_field_not_a_schema_keyword():
    schema = _object(pin=_object(properties={"type": "string"}))
    assert collapse_nested_objects(schema) == schema
