import json

from fastapi.encoders import jsonable_encoder


def to_dict(data) -> dict:
    return jsonable_encoder(data)


def dumps(data) -> str:
    return json.dumps(jsonable_encoder(data))


loads = json.loads
