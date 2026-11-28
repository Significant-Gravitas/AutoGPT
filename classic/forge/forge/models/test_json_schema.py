import pytest

from forge.models.json_schema import JSONSchema


class TestToDict:
    def test_keeps_numeric_bounds(self):
        schema = JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="The execution timeout for the code.",
            minimum=1,
            maximum=600,
        )

        assert schema.to_dict() == {
            "type": "integer",
            "description": "The execution timeout for the code.",
            "minimum": 1,
            "maximum": 600,
        }

    def test_keeps_a_minimum_without_a_maximum(self):
        schema = JSONSchema(type=JSONSchema.Type.NUMBER, minimum=0)

        assert schema.to_dict()["minimum"] == 0

    def test_keeps_the_enum_of_a_text_parameter(self):
        schema = JSONSchema(type=JSONSchema.Type.STRING, enum=["low", "high"])

        assert schema.to_dict()["enum"] == ["low", "high"]


class TestValidateObject:
    def test_refuses_a_value_below_the_minimum(self):
        schema = JSONSchema(type=JSONSchema.Type.INTEGER, minimum=1, maximum=600)

        is_valid, errors = schema.validate_object(0)

        assert not is_valid
        assert len(errors) == 1

    def test_refuses_a_value_above_the_maximum(self):
        schema = JSONSchema(type=JSONSchema.Type.INTEGER, minimum=1, maximum=600)

        is_valid, errors = schema.validate_object(601)

        assert not is_valid
        assert len(errors) == 1

    @pytest.mark.parametrize("value", [1, 300, 600])
    def test_accepts_a_value_inside_the_bounds(self, value: int):
        schema = JSONSchema(type=JSONSchema.Type.INTEGER, minimum=1, maximum=600)

        is_valid, errors = schema.validate_object(value)

        assert is_valid
        assert errors == []
