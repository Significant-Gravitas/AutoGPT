import json
from enum import Enum
from pathlib import PurePosixPath

import pytest

from backend.util.json import SafeJson


class EncodedKey(Enum):
    SQLALCHEMY = "_sa_metadata"


class ContainerPayload(Enum):
    DICT = {"_sa_metadata": "omitted", "public": "yes\x00"}
    LIST = ["text\x07", {"_sa_metadata": "omitted", "public": "yes\x00"}]
    RAW_KEYS = {b"_sa_raw": "kept", "_sa_unsupported": object()}


def test_safejson_encodes_enum_bytes_before_returning_json_data():
    class Blob(Enum):
        DATA = b"ok"

    result = SafeJson({"blob": Blob.DATA})

    assert json.dumps(result.data) == '{"blob": "ok"}'
    assert result.data == {"blob": "ok"}


@pytest.mark.parametrize(
    "key", [b"_sa_metadata", EncodedKey.SQLALCHEMY, PurePosixPath("_sa_metadata")]
)
def test_safejson_filters_keys_encoded_to_sqlalchemy_names(key):
    result = SafeJson({"nested": [{key: "omitted", "public": "yes\x00"}]})

    assert result.data == {"nested": [{"public": "yes"}]}
    assert json.loads(json.dumps(result.data)) == result.data


@pytest.mark.parametrize(
    "payload, expected",
    [
        (ContainerPayload.DICT, {"public": "yes"}),
        (ContainerPayload.LIST, ["text", {"public": "yes"}]),
        (ContainerPayload.RAW_KEYS, {"_sa_raw": "kept"}),
    ],
)
def test_safejson_filters_sqlalchemy_keys_in_enum_containers(payload, expected):
    result = SafeJson({"payload": payload})

    assert result.data == {"payload": expected}
    assert json.loads(json.dumps(result.data)) == result.data


def test_safejson_filters_before_sanitizing_keys_and_resolving_collisions():
    result = SafeJson(
        {
            "_\x00sa_metadata": "first",
            "_s\x00a_metadata": "last",
            b"_sa_metadata": "omitted",
            "_SA_metadata": "uppercase",
            7: "integer",
            False: "boolean",
            None: "null",
        }
    )

    assert result.data == {
        "_sa_metadata": "last",
        "_SA_metadata": "uppercase",
        7: "integer",
        False: "boolean",
        None: "null",
    }
