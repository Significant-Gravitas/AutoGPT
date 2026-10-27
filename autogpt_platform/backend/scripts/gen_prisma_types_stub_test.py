"""Tests for the Prisma types stub generator: which aliases keep their type
and which collapse to ``dict[str, Any]``."""

from pathlib import Path

from scripts.gen_prisma_types_stub import generate_stub

_TYPES = """
from typing import Union, List
from typing_extensions import Literal

SortOrder = _types.SortOrder
DreamPassScalarFieldKeys = Literal['id', 'status']
Serializable = Union[None, bool, str, List['Serializable']]
DreamPassOrderByInput = Union[
    '_DreamPass_id_OrderByInput',
    '_DreamPass_status_OrderByInput',
]


class _DreamPass_id_OrderByInput(dict):
    pass


class DreamPassWhereInput(dict):
    pass
"""


def test_a_union_of_private_types_collapses_and_the_rest_keep_their_type(
    tmp_path: Path,
) -> None:
    """A Union naming the private types the stub leaves out would read as
    Unknown (every ``*OrderByInput``, so every ``find_many``); it collapses
    like the classes do. Literals, public Unions and module references
    keep their type."""
    source = tmp_path / "types.py"
    source.write_text(_TYPES, encoding="utf-8")
    stub = tmp_path / "types.pyi"

    generate_stub(source, stub)

    lines = stub.read_text(encoding="utf-8").splitlines()
    assert "DreamPassOrderByInput = _PrismaDict" in lines
    assert "DreamPassWhereInput = _PrismaDict" in lines
    assert "SortOrder = _types.SortOrder" in lines
    assert "DreamPassScalarFieldKeys = Literal['id', 'status']" in lines
    assert any(line.startswith("Serializable = Union[") for line in lines)
    assert not any("_DreamPass_id_OrderByInput" in line for line in lines)
