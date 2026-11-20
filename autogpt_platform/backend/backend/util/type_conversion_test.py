import pytest
from pydantic import BaseModel

from backend.util.type import coerce_inputs_to_schema, convert


@pytest.mark.parametrize("value", ['["1", "2", "3"]', ["1", "2", "3"], ("1", "2", "3")])
def test_variadic_tuples_convert_every_element(value: object):
    assert convert(value, tuple[int, ...]) == (1, 2, 3)


@pytest.mark.parametrize("value", ['["5", 6]', ["5", 6], ("5", 6)])
def test_fixed_length_tuples_keep_their_per_element_types(value: object):
    assert convert(value, tuple[int, str]) == (5, "6")


class VariadicTupleInputs(BaseModel):
    values: tuple[int, ...]


def test_schema_coercion_keeps_every_element_of_a_variadic_tuple():
    values = {"values": '["1", "2", "3"]'}
    coerce_inputs_to_schema(values, VariadicTupleInputs)
    assert VariadicTupleInputs.model_validate(values).values == (1, 2, 3)
