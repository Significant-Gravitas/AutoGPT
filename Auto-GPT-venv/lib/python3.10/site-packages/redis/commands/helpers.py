import copy
import random
import string
from typing import List, Tuple

from redis.typing import KeysT, KeyT


def list_or_args(keys: KeysT, args: Tuple[KeyT, ...]) -> List[KeyT]:
    # returns a single new list combining keys and args
    try:
        iter(keys)
        # a string or bytes instance can be iterated, but indicates
        # keys wasn't passed as a list
        if isinstance(keys, (bytes, str)):
            keys = [keys]
        else:
            keys = list(keys)
    except TypeError:
        keys = [keys]
    if args:
        keys.extend(args)
    return keys


def nativestr(x):
    """Return the decoded binary string, or a string, depending on type."""
    r = x.decode("utf-8", "replace") if isinstance(x, bytes) else x
    if r == "null":
        return
    return r


def delist(x):
    """Given a list of binaries, return the stringified version."""
    if x is None:
        return x
    return [nativestr(obj) for obj in x]


def parse_to_list(response):
    """Optimistically parse the response to a list."""
    res = []

    if response is None:
        return res

    for item in response:
        try:
            res.append(int(item))
        except ValueError:
            try:
                res.append(float(item))
            except ValueError:
                res.append(nativestr(item))
        except TypeError:
            res.append(None)
    return res


def parse_list_to_dict(response):
    res = {}
    for i in range(0, len(response), 2):
        if isinstance(response[i], list):
            res["Child iterators"].append(parse_list_to_dict(response[i]))
        elif isinstance(response[i + 1], list):
            res["Child iterators"] = [parse_list_to_dict(response[i + 1])]
        else:
            try:
                res[response[i]] = float(response[i + 1])
            except (TypeError, ValueError):
                res[response[i]] = response[i + 1]
    return res


def parse_to_dict(response):
    if response is None:
        return {}

    res = {}
    for det in response:
        if isinstance(det[1], list):
            res[det[0]] = parse_list_to_dict(det[1])
        else:
            try:  # try to set the attribute. may be provided without value
                try:  # try to convert the value to float
                    res[det[0]] = float(det[1])
                except (TypeError, ValueError):
                    res[det[0]] = det[1]
            except IndexError:
                pass
    return res


def random_string(length=10):
    """
    Returns a random N character long string.
    """
    return "".join(  # nosec
        random.choice(string.ascii_lowercase) for x in range(length)
    )


def quote_string(v):
    """
    RedisGraph strings must be quoted,
    quote_string wraps given v with quotes incase
    v is a string.
    """

    if isinstance(v, bytes):
        v = v.decode()
    elif not isinstance(v, str):
        return v
    if len(v) == 0:
        return '""'

    v = v.replace("\\", "\\\\")
    v = v.replace('"', '\\"')

    return f'"{v}"'


def decode_dict_keys(obj):
    """Decode the keys of the given dictionary with utf-8."""
    newobj = copy.copy(obj)
    for k in obj.keys():
        if isinstance(k, bytes):
            newobj[k.decode("utf-8")] = newobj[k]
            newobj.pop(k)
    return newobj


def stringify_param_value(value):
    """
    Turn a parameter value into a string suitable for the params header of
    a Cypher command.
    You may pass any value that would be accepted by `json.dumps()`.

    Ways in which output differs from that of `str()`:
        * Strings are quoted.
        * None --> "null".
        * In dictionaries, keys are _not_ quoted.

    :param value: The parameter value to be turned into a string.
    :return: string
    """

    if isinstance(value, str):
        return quote_string(value)
    elif value is None:
        return "null"
    elif isinstance(value, (list, tuple)):
        return f'[{",".join(map(stringify_param_value, value))}]'
    elif isinstance(value, dict):
        return f'{{{",".join(f"{k}:{stringify_param_value(v)}" for k, v in value.items())}}}'  # noqa
    else:
        return str(value)
