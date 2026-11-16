"""Integers outside orjson's i64/u64 range must survive loads() (#16811)."""

import json

import orjson
import pytest

from backend.util.json import loads

_U64_MAX = 18446744073709551615
_I64_MIN = -9223372036854775808
_ORDER_ID = 123456789012345678901234
_LEDGER_SEQ = 18446744073709551616
_VALUE_WEI = 1234567890123456789012


def test_loads_keeps_integers_orjson_would_round():
    payload = {
        "order_id": _ORDER_ID,
        "ledger_seq": _LEDGER_SEQ,
        "value_wei": _VALUE_WEI,
    }
    parsed = loads(json.dumps(payload))
    assert parsed == payload
    assert all(type(value) is int for value in parsed.values())


def test_loads_keeps_the_json_decoder_wei_example():
    raw = '{"tx":"0xabc","value_wei":1234567890123456789012,"block":21000000}'
    parsed = loads(raw)
    assert parsed == {"tx": "0xabc", "value_wei": _VALUE_WEI, "block": 21000000}
    assert type(parsed["value_wei"]) is int
    assert type(parsed["block"]) is int


def test_loads_keeps_a_nested_list_of_big_integers():
    raw = '{"ids":[123456789012345678901234,1]}'
    parsed = loads(raw)
    assert parsed["ids"] == [_ORDER_ID, 1]
    assert type(parsed["ids"][0]) is int


def test_loads_keeps_u64_max_and_i64_min_exact():
    assert loads(str(_U64_MAX)) == _U64_MAX
    assert type(loads(str(_U64_MAX))) is int
    assert loads(str(_I64_MIN)) == _I64_MIN
    assert type(loads(str(_I64_MIN))) is int


def test_loads_keeps_an_integer_below_i64_min():
    value = _I64_MIN - 1
    parsed = loads(str(value))
    assert parsed == value
    assert type(parsed) is int


def test_loads_leaves_ordinary_json_on_the_orjson_result():
    raw = '{"n":1,"f":1.5,"s":"123456789012345678901234","b":true,"z":null}'
    assert loads(raw) == orjson.loads(raw)


def test_loads_big_integer_from_bytes():
    raw = b'{"n":123456789012345678901234}'
    assert loads(raw)["n"] == _ORDER_ID
    assert type(loads(raw)["n"]) is int


def test_loads_invalid_json_still_raises_orjson_error():
    with pytest.raises(orjson.JSONDecodeError):
        loads("{")


def test_loads_invalid_json_still_honours_fallback():
    assert loads("not json", fallback=None) is None
