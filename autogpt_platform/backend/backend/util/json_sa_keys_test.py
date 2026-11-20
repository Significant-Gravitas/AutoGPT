"""Dict keys starting with "_sa" survive the JSON helpers.

FastAPI's ``jsonable_encoder`` drops every such key by default, at any depth: a
guard for SQLAlchemy ORM state, which nothing here encodes. A block output such
as ``{"_sales_total": 42}`` was stored, sent and shown without it.
"""

import pytest
from pydantic import BaseModel

from backend.util.json import SafeJson, dumps, loads, to_dict

REPORT = {"_sales_total": 42, "rows": [{"_sa_region": "EMEA"}]}


class Wrapper(BaseModel):
    report: dict


@pytest.mark.parametrize(
    "encode",
    [to_dict, lambda value: loads(dumps(value)), lambda value: SafeJson(value).data],
    ids=["to_dict", "dumps", "SafeJson"],
)
def test_keys_starting_with_sa_are_kept_at_every_depth(encode):
    assert encode(REPORT) == REPORT
    assert encode(Wrapper(report=REPORT)) == {"report": REPORT}


def test_safejson_still_cleans_keys_starting_with_sa_and_their_values():
    dirty = {"_sa\x00les_total": "4\x002", "rows": [{"_sa_region": "EM\x07EA\n"}]}

    assert SafeJson(dirty).data == {
        "_sales_total": "42",
        "rows": [{"_sa_region": "EMEA\n"}],
    }
