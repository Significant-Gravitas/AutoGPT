import copy
import re

from ..helpers import nativestr


def bulk_of_jsons(d):
    """Replace serialized JSON values with objects in a
    bulk array response (list).
    """

    def _f(b):
        for index, item in enumerate(b):
            if item is not None:
                b[index] = d(item)
        return b

    return _f


def decode_dict_keys(obj):
    """Decode the keys of the given dictionary with utf-8."""
    newobj = copy.copy(obj)
    for k in obj.keys():
        if isinstance(k, bytes):
            newobj[k.decode("utf-8")] = newobj[k]
            newobj.pop(k)
    return newobj


def unstring(obj):
    """
    Attempt to parse string to native integer formats.
    One can't simply call int/float in a try/catch because there is a
    semantic difference between (for example) 15.0 and 15.
    """
    floatreg = "^\\d+.\\d+$"
    match = re.findall(floatreg, obj)
    if match != []:
        return float(match[0])

    intreg = "^\\d+$"
    match = re.findall(intreg, obj)
    if match != []:
        return int(match[0])
    return obj


def decode_list(b):
    """
    Given a non-deserializable object, make a best effort to
    return a useful set of results.
    """
    if isinstance(b, list):
        return [nativestr(obj) for obj in b]
    elif isinstance(b, bytes):
        return unstring(nativestr(b))
    elif isinstance(b, str):
        return unstring(b)
    return b
