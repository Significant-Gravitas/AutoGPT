from typing import TypedDict

from numpy import generic, signedinteger, unsignedinteger, floating, complexfloating

class _SCTypes(TypedDict):
    int: list[type[signedinteger]]
    uint: list[type[unsignedinteger]]
    float: list[type[floating]]
    complex: list[type[complexfloating]]
    others: list[type]

sctypeDict: dict[int | str, type[generic]]
sctypes: _SCTypes
