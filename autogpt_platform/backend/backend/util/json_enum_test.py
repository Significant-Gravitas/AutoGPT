import json
from enum import Enum

from backend.util.json import SafeJson


def test_safejson_encodes_enum_bytes_before_returning_json_data():
    class Blob(Enum):
        DATA = b"ok"

    result = SafeJson({"blob": Blob.DATA})

    assert json.dumps(result.data) == '{"blob": "ok"}'
    assert result.data == {"blob": "ok"}
