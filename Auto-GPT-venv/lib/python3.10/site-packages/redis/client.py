import copy
import datetime
import re
import threading
import time
import warnings
from itertools import chain
from typing import Optional

from redis.commands import (
    CoreCommands,
    RedisModuleCommands,
    SentinelCommands,
    list_or_args,
)
from redis.connection import ConnectionPool, SSLConnection, UnixDomainSocketConnection
from redis.credentials import CredentialProvider
from redis.exceptions import (
    ConnectionError,
    ExecAbortError,
    ModuleError,
    PubSubError,
    RedisError,
    ResponseError,
    TimeoutError,
    WatchError,
)
from redis.lock import Lock
from redis.retry import Retry
from redis.utils import safe_str, str_if_bytes

SYM_EMPTY = b""
EMPTY_RESPONSE = "EMPTY_RESPONSE"

# some responses (ie. dump) are binary, and just meant to never be decoded
NEVER_DECODE = "NEVER_DECODE"


def timestamp_to_datetime(response):
    "Converts a unix timestamp to a Python datetime object"
    if not response:
        return None
    try:
        response = int(response)
    except ValueError:
        return None
    return datetime.datetime.fromtimestamp(response)


def string_keys_to_dict(key_string, callback):
    return dict.fromkeys(key_string.split(), callback)


class CaseInsensitiveDict(dict):
    "Case insensitive dict implementation. Assumes string keys only."

    def __init__(self, data):
        for k, v in data.items():
            self[k.upper()] = v

    def __contains__(self, k):
        return super().__contains__(k.upper())

    def __delitem__(self, k):
        super().__delitem__(k.upper())

    def __getitem__(self, k):
        return super().__getitem__(k.upper())

    def get(self, k, default=None):
        return super().get(k.upper(), default)

    def __setitem__(self, k, v):
        super().__setitem__(k.upper(), v)

    def update(self, data):
        data = CaseInsensitiveDict(data)
        super().update(data)


def parse_debug_object(response):
    "Parse the results of Redis's DEBUG OBJECT command into a Python dict"
    # The 'type' of the object is the first item in the response, but isn't
    # prefixed with a name
    response = str_if_bytes(response)
    response = "type:" + response
    response = dict(kv.split(":") for kv in response.split())

    # parse some expected int values from the string response
    # note: this cmd isn't spec'd so these may not appear in all redis versions
    int_fields = ("refcount", "serializedlength", "lru", "lru_seconds_idle")
    for field in int_fields:
        if field in response:
            response[field] = int(response[field])

    return response


def parse_object(response, infotype):
    """Parse the results of an OBJECT command"""
    if infotype in ("idletime", "refcount"):
        return int_or_none(response)
    return response


def parse_info(response):
    """Parse the result of Redis's INFO command into a Python dict"""
    info = {}
    response = str_if_bytes(response)

    def get_value(value):
        if "," not in value or "=" not in value:
            try:
                if "." in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value
        else:
            sub_dict = {}
            for item in value.split(","):
                k, v = item.rsplit("=", 1)
                sub_dict[k] = get_value(v)
            return sub_dict

    for line in response.splitlines():
        if line and not line.startswith("#"):
            if line.find(":") != -1:
                # Split, the info fields keys and values.
                # Note that the value may contain ':'. but the 'host:'
                # pseudo-command is the only case where the key contains ':'
                key, value = line.split(":", 1)
                if key == "cmdstat_host":
                    key, value = line.rsplit(":", 1)

                if key == "module":
                    # Hardcode a list for key 'modules' since there could be
                    # multiple lines that started with 'module'
                    info.setdefault("modules", []).append(get_value(value))
                else:
                    info[key] = get_value(value)
            else:
                # if the line isn't splittable, append it to the "__raw__" key
                info.setdefault("__raw__", []).append(line)

    return info


def parse_memory_stats(response, **kwargs):
    """Parse the results of MEMORY STATS"""
    stats = pairs_to_dict(response, decode_keys=True, decode_string_values=True)
    for key, value in stats.items():
        if key.startswith("db."):
            stats[key] = pairs_to_dict(
                value, decode_keys=True, decode_string_values=True
            )
    return stats


SENTINEL_STATE_TYPES = {
    "can-failover-its-master": int,
    "config-epoch": int,
    "down-after-milliseconds": int,
    "failover-timeout": int,
    "info-refresh": int,
    "last-hello-message": int,
    "last-ok-ping-reply": int,
    "last-ping-reply": int,
    "last-ping-sent": int,
    "master-link-down-time": int,
    "master-port": int,
    "num-other-sentinels": int,
    "num-slaves": int,
    "o-down-time": int,
    "pending-commands": int,
    "parallel-syncs": int,
    "port": int,
    "quorum": int,
    "role-reported-time": int,
    "s-down-time": int,
    "slave-priority": int,
    "slave-repl-offset": int,
    "voted-leader-epoch": int,
}


def parse_sentinel_state(item):
    result = pairs_to_dict_typed(item, SENTINEL_STATE_TYPES)
    flags = set(result["flags"].split(","))
    for name, flag in (
        ("is_master", "master"),
        ("is_slave", "slave"),
        ("is_sdown", "s_down"),
        ("is_odown", "o_down"),
        ("is_sentinel", "sentinel"),
        ("is_disconnected", "disconnected"),
        ("is_master_down", "master_down"),
    ):
        result[name] = flag in flags
    return result


def parse_sentinel_master(response):
    return parse_sentinel_state(map(str_if_bytes, response))


def parse_sentinel_masters(response):
    result = {}
    for item in response:
        state = parse_sentinel_state(map(str_if_bytes, item))
        result[state["name"]] = state
    return result


def parse_sentinel_slaves_and_sentinels(response):
    return [parse_sentinel_state(map(str_if_bytes, item)) for item in response]


def parse_sentinel_get_master(response):
    return response and (response[0], int(response[1])) or None


def pairs_to_dict(response, decode_keys=False, decode_string_values=False):
    """Create a dict given a list of key/value pairs"""
    if response is None:
        return {}
    if decode_keys or decode_string_values:
        # the iter form is faster, but I don't know how to make that work
        # with a str_if_bytes() map
        keys = response[::2]
        if decode_keys:
            keys = map(str_if_bytes, keys)
        values = response[1::2]
        if decode_string_values:
            values = map(str_if_bytes, values)
        return dict(zip(keys, values))
    else:
        it = iter(response)
        return dict(zip(it, it))


def pairs_to_dict_typed(response, type_info):
    it = iter(response)
    result = {}
    for key, value in zip(it, it):
        if key in type_info:
            try:
                value = type_info[key](value)
            except Exception:
                # if for some reason the value can't be coerced, just use
                # the string value
                pass
        result[key] = value
    return result


def zset_score_pairs(response, **options):
    """
    If ``withscores`` is specified in the options, return the response as
    a list of (value, score) pairs
    """
    if not response or not options.get("withscores"):
        return response
    score_cast_func = options.get("score_cast_func", float)
    it = iter(response)
    return list(zip(it, map(score_cast_func, it)))


def sort_return_tuples(response, **options):
    """
    If ``groups`` is specified, return the response as a list of
    n-element tuples with n being the value found in options['groups']
    """
    if not response or not options.get("groups"):
        return response
    n = options["groups"]
    return list(zip(*[response[i::n] for i in range(n)]))


def int_or_none(response):
    if response is None:
        return None
    return int(response)


def parse_stream_list(response):
    if response is None:
        return None
    data = []
    for r in response:
        if r is not None:
            data.append((r[0], pairs_to_dict(r[1])))
        else:
            data.append((None, None))
    return data


def pairs_to_dict_with_str_keys(response):
    return pairs_to_dict(response, decode_keys=True)


def parse_list_of_dicts(response):
    return list(map(pairs_to_dict_with_str_keys, response))


def parse_xclaim(response, **options):
    if options.get("parse_justid", False):
        return response
    return parse_stream_list(response)


def parse_xautoclaim(response, **options):
    if options.get("parse_justid", False):
        return response[1]
    response[1] = parse_stream_list(response[1])
    return response


def parse_xinfo_stream(response, **options):
    data = pairs_to_dict(response, decode_keys=True)
    if not options.get("full", False):
        first = data["first-entry"]
        if first is not None:
            data["first-entry"] = (first[0], pairs_to_dict(first[1]))
        last = data["last-entry"]
        if last is not None:
            data["last-entry"] = (last[0], pairs_to_dict(last[1]))
    else:
        data["entries"] = {_id: pairs_to_dict(entry) for _id, entry in data["entries"]}
        data["groups"] = [
            pairs_to_dict(group, decode_keys=True) for group in data["groups"]
        ]
    return data


def parse_xread(response):
    if response is None:
        return []
    return [[r[0], parse_stream_list(r[1])] for r in response]


def parse_xpending(response, **options):
    if options.get("parse_detail", False):
        return parse_xpending_range(response)
    consumers = [{"name": n, "pending": int(p)} for n, p in response[3] or []]
    return {
        "pending": response[0],
        "min": response[1],
        "max": response[2],
        "consumers": consumers,
    }


def parse_xpending_range(response):
    k = ("message_id", "consumer", "time_since_delivered", "times_delivered")
    return [dict(zip(k, r)) for r in response]


def float_or_none(response):
    if response is None:
        return None
    return float(response)


def bool_ok(response):
    return str_if_bytes(response) == "OK"


def parse_zadd(response, **options):
    if response is None:
        return None
    if options.get("as_score"):
        return float(response)
    return int(response)


def parse_client_list(response, **options):
    clients = []
    for c in str_if_bytes(response).splitlines():
        # Values might contain '='
        clients.append(dict(pair.split("=", 1) for pair in c.split(" ")))
    return clients


def parse_config_get(response, **options):
    response = [str_if_bytes(i) if i is not None else None for i in response]
    return response and pairs_to_dict(response) or {}


def parse_scan(response, **options):
    cursor, r = response
    return int(cursor), r


def parse_hscan(response, **options):
    cursor, r = response
    return int(cursor), r and pairs_to_dict(r) or {}


def parse_zscan(response, **options):
    score_cast_func = options.get("score_cast_func", float)
    cursor, r = response
    it = iter(r)
    return int(cursor), list(zip(it, map(score_cast_func, it)))


def parse_zmscore(response, **options):
    # zmscore: list of scores (double precision floating point number) or nil
    return [float(score) if score is not None else None for score in response]


def parse_slowlog_get(response, **options):
    space = " " if options.get("decode_responses", False) else b" "

    def parse_item(item):
        result = {"id": item[0], "start_time": int(item[1]), "duration": int(item[2])}
        # Redis Enterprise injects another entry at index [3], which has
        # the complexity info (i.e. the value N in case the command has
        # an O(N) complexity) instead of the command.
        if isinstance(item[3], list):
            result["command"] = space.join(item[3])
        else:
            result["complexity"] = item[3]
            result["command"] = space.join(item[4])
        return result

    return [parse_item(item) for item in response]


def parse_stralgo(response, **options):
    """
    Parse the response from `STRALGO` command.
    Without modifiers the returned value is string.
    When LEN is given the command returns the length of the result
    (i.e integer).
    When IDX is given the command returns a dictionary with the LCS
    length and all the ranges in both the strings, start and end
    offset for each string, where there are matches.
    When WITHMATCHLEN is given, each array representing a match will
    also have the length of the match at the beginning of the array.
    """
    if options.get("len", False):
        return int(response)
    if options.get("idx", False):
        if options.get("withmatchlen", False):
            matches = [
                [(int(match[-1]))] + list(map(tuple, match[:-1]))
                for match in response[1]
            ]
        else:
            matches = [list(map(tuple, match)) for match in response[1]]
        return {
            str_if_bytes(response[0]): matches,
            str_if_bytes(response[2]): int(response[3]),
        }
    return str_if_bytes(response)


def parse_cluster_info(response, **options):
    response = str_if_bytes(response)
    return dict(line.split(":") for line in response.splitlines() if line)


def _parse_node_line(line):
    line_items = line.split(" ")
    node_id, addr, flags, master_id, ping, pong, epoch, connected = line.split(" ")[:8]
    addr = addr.split("@")[0]
    node_dict = {
        "node_id": node_id,
        "flags": flags,
        "master_id": master_id,
        "last_ping_sent": ping,
        "last_pong_rcvd": pong,
        "epoch": epoch,
        "slots": [],
        "migrations": [],
        "connected": True if connected == "connected" else False,
    }
    if len(line_items) >= 9:
        slots, migrations = _parse_slots(line_items[8:])
        node_dict["slots"], node_dict["migrations"] = slots, migrations
    return addr, node_dict


def _parse_slots(slot_ranges):
    slots, migrations = [], []
    for s_range in slot_ranges:
        if "->-" in s_range:
            slot_id, dst_node_id = s_range[1:-1].split("->-", 1)
            migrations.append(
                {"slot": slot_id, "node_id": dst_node_id, "state": "migrating"}
            )
        elif "-<-" in s_range:
            slot_id, src_node_id = s_range[1:-1].split("-<-", 1)
            migrations.append(
                {"slot": slot_id, "node_id": src_node_id, "state": "importing"}
            )
        else:
            s_range = [sl for sl in s_range.split("-")]
            slots.append(s_range)

    return slots, migrations


def parse_cluster_nodes(response, **options):
    """
    @see: https://redis.io/commands/cluster-nodes  # string / bytes
    @see: https://redis.io/commands/cluster-replicas # list of string / bytes
    """
    if isinstance(response, (str, bytes)):
        response = response.splitlines()
    return dict(_parse_node_line(str_if_bytes(node)) for node in response)


def parse_geosearch_generic(response, **options):
    """
    Parse the response of 'GEOSEARCH', GEORADIUS' and 'GEORADIUSBYMEMBER'
    commands according to 'withdist', 'withhash' and 'withcoord' labels.
    """
    if options["store"] or options["store_dist"]:
        # `store` and `store_dist` cant be combined
        # with other command arguments.
        # relevant to 'GEORADIUS' and 'GEORADIUSBYMEMBER'
        return response

    if type(response) != list:
        response_list = [response]
    else:
        response_list = response

    if not options["withdist"] and not options["withcoord"] and not options["withhash"]:
        # just a bunch of places
        return response_list

    cast = {
        "withdist": float,
        "withcoord": lambda ll: (float(ll[0]), float(ll[1])),
        "withhash": int,
    }

    # zip all output results with each casting function to get
    # the properly native Python value.
    f = [lambda x: x]
    f += [cast[o] for o in ["withdist", "withhash", "withcoord"] if options[o]]
    return [list(map(lambda fv: fv[0](fv[1]), zip(f, r))) for r in response_list]


def parse_command(response, **options):
    commands = {}
    for command in response:
        cmd_dict = {}
        cmd_name = str_if_bytes(command[0])
        cmd_dict["name"] = cmd_name
        cmd_dict["arity"] = int(command[1])
        cmd_dict["flags"] = [str_if_bytes(flag) for flag in command[2]]
        cmd_dict["first_key_pos"] = command[3]
        cmd_dict["last_key_pos"] = command[4]
        cmd_dict["step_count"] = command[5]
        if len(command) > 7:
            cmd_dict["tips"] = command[7]
            cmd_dict["key_specifications"] = command[8]
            cmd_dict["subcommands"] = command[9]
        commands[cmd_name] = cmd_dict
    return commands


def parse_pubsub_numsub(response, **options):
    return list(zip(response[0::2], response[1::2]))


def parse_client_kill(response, **options):
    if isinstance(response, int):
        return response
    return str_if_bytes(response) == "OK"


def parse_acl_getuser(response, **options):
    if response is None:
        return None
    data = pairs_to_dict(response, decode_keys=True)

    # convert everything but user-defined data in 'keys' to native strings
    data["flags"] = list(map(str_if_bytes, data["flags"]))
    data["passwords"] = list(map(str_if_bytes, data["passwords"]))
    data["commands"] = str_if_bytes(data["commands"])
    if isinstance(data["keys"], str) or isinstance(data["keys"], bytes):
        data["keys"] = list(str_if_bytes(data["keys"]).split(" "))
    if data["keys"] == [""]:
        data["keys"] = []
    if "channels" in data:
        if isinstance(data["channels"], str) or isinstance(data["channels"], bytes):
            data["channels"] = list(str_if_bytes(data["channels"]).split(" "))
        if data["channels"] == [""]:
            data["channels"] = []
    if "selectors" in data:
        data["selectors"] = [
            list(map(str_if_bytes, selector)) for selector in data["selectors"]
        ]

    # split 'commands' into separate 'categories' and 'commands' lists
    commands, categories = [], []
    for command in data["commands"].split(" "):
        if "@" in command:
            categories.append(command)
        else:
            commands.append(command)

    data["commands"] = commands
    data["categories"] = categories
    data["enabled"] = "on" in data["flags"]
    return data


def parse_acl_log(response, **options):
    if response is None:
        return None
    if isinstance(response, list):
        data = []
        for log in response:
            log_data = pairs_to_dict(log, True, True)
            client_info = log_data.get("client-info", "")
            log_data["client-info"] = parse_client_info(client_info)

            # float() is lossy comparing to the "double" in C
            log_data["age-seconds"] = float(log_data["age-seconds"])
            data.append(log_data)
    else:
        data = bool_ok(response)
    return data


def parse_client_info(value):
    """
    Parsing client-info in ACL Log in following format.
    "key1=value1 key2=value2 key3=value3"
    """
    client_info = {}
    infos = str_if_bytes(value).split(" ")
    for info in infos:
        key, value = info.split("=")
        client_info[key] = value

    # Those fields are defined as int in networking.c
    for int_key in {
        "id",
        "age",
        "idle",
        "db",
        "sub",
        "psub",
        "multi",
        "qbuf",
        "qbuf-free",
        "obl",
        "argv-mem",
        "oll",
        "omem",
        "tot-mem",
    }:
        client_info[int_key] = int(client_info[int_key])
    return client_info


def parse_module_result(response):
    if isinstance(response, ModuleError):
        raise response
    return True


def parse_set_result(response, **options):
    """
    Handle SET result since GET argument is available since Redis 6.2.
    Parsing SET result into:
    - BOOL
    - String when GET argument is used
    """
    if options.get("get"):
        # Redis will return a getCommand result.
        # See `setGenericCommand` in t_string.c
        return response
    return response and str_if_bytes(response) == "OK"


class AbstractRedis:
    RESPONSE_CALLBACKS = {
        **string_keys_to_dict(
            "AUTH COPY EXPIRE EXPIREAT PEXPIRE PEXPIREAT "
            "HEXISTS HMSET MOVE MSETNX PERSIST "
            "PSETEX RENAMENX SISMEMBER SMOVE SETEX SETNX",
            bool,
        ),
        **string_keys_to_dict(
            "BITCOUNT BITPOS DECRBY DEL EXISTS GEOADD GETBIT HDEL HLEN "
            "HSTRLEN INCRBY LINSERT LLEN LPUSHX PFADD PFCOUNT RPUSHX SADD "
            "SCARD SDIFFSTORE SETBIT SETRANGE SINTERSTORE SREM STRLEN "
            "SUNIONSTORE UNLINK XACK XDEL XLEN XTRIM ZCARD ZLEXCOUNT ZREM "
            "ZREMRANGEBYLEX ZREMRANGEBYRANK ZREMRANGEBYSCORE",
            int,
        ),
        **string_keys_to_dict("INCRBYFLOAT HINCRBYFLOAT", float),
        **string_keys_to_dict(
            # these return OK, or int if redis-server is >=1.3.4
            "LPUSH RPUSH",
            lambda r: isinstance(r, int) and r or str_if_bytes(r) == "OK",
        ),
        **string_keys_to_dict("SORT", sort_return_tuples),
        **string_keys_to_dict("ZSCORE ZINCRBY GEODIST", float_or_none),
        **string_keys_to_dict(
            "FLUSHALL FLUSHDB LSET LTRIM MSET PFMERGE ASKING READONLY READWRITE "
            "RENAME SAVE SELECT SHUTDOWN SLAVEOF SWAPDB WATCH UNWATCH ",
            bool_ok,
        ),
        **string_keys_to_dict("BLPOP BRPOP", lambda r: r and tuple(r) or None),
        **string_keys_to_dict(
            "SDIFF SINTER SMEMBERS SUNION", lambda r: r and set(r) or set()
        ),
        **string_keys_to_dict(
            "ZPOPMAX ZPOPMIN ZINTER ZDIFF ZUNION ZRANGE ZRANGEBYSCORE "
            "ZREVRANGE ZREVRANGEBYSCORE",
            zset_score_pairs,
        ),
        **string_keys_to_dict(
            "BZPOPMIN BZPOPMAX", lambda r: r and (r[0], r[1], float(r[2])) or None
        ),
        **string_keys_to_dict("ZRANK ZREVRANK", int_or_none),
        **string_keys_to_dict("XREVRANGE XRANGE", parse_stream_list),
        **string_keys_to_dict("XREAD XREADGROUP", parse_xread),
        **string_keys_to_dict("BGREWRITEAOF BGSAVE", lambda r: True),
        "ACL CAT": lambda r: list(map(str_if_bytes, r)),
        "ACL DELUSER": int,
        "ACL GENPASS": str_if_bytes,
        "ACL GETUSER": parse_acl_getuser,
        "ACL HELP": lambda r: list(map(str_if_bytes, r)),
        "ACL LIST": lambda r: list(map(str_if_bytes, r)),
        "ACL LOAD": bool_ok,
        "ACL LOG": parse_acl_log,
        "ACL SAVE": bool_ok,
        "ACL SETUSER": bool_ok,
        "ACL USERS": lambda r: list(map(str_if_bytes, r)),
        "ACL WHOAMI": str_if_bytes,
        "CLIENT GETNAME": str_if_bytes,
        "CLIENT ID": int,
        "CLIENT KILL": parse_client_kill,
        "CLIENT LIST": parse_client_list,
        "CLIENT INFO": parse_client_info,
        "CLIENT SETNAME": bool_ok,
        "CLIENT UNBLOCK": lambda r: r and int(r) == 1 or False,
        "CLIENT PAUSE": bool_ok,
        "CLIENT GETREDIR": int,
        "CLIENT TRACKINGINFO": lambda r: list(map(str_if_bytes, r)),
        "CLUSTER ADDSLOTS": bool_ok,
        "CLUSTER ADDSLOTSRANGE": bool_ok,
        "CLUSTER COUNT-FAILURE-REPORTS": lambda x: int(x),
        "CLUSTER COUNTKEYSINSLOT": lambda x: int(x),
        "CLUSTER DELSLOTS": bool_ok,
        "CLUSTER DELSLOTSRANGE": bool_ok,
        "CLUSTER FAILOVER": bool_ok,
        "CLUSTER FORGET": bool_ok,
        "CLUSTER GETKEYSINSLOT": lambda r: list(map(str_if_bytes, r)),
        "CLUSTER INFO": parse_cluster_info,
        "CLUSTER KEYSLOT": lambda x: int(x),
        "CLUSTER MEET": bool_ok,
        "CLUSTER NODES": parse_cluster_nodes,
        "CLUSTER REPLICAS": parse_cluster_nodes,
        "CLUSTER REPLICATE": bool_ok,
        "CLUSTER RESET": bool_ok,
        "CLUSTER SAVECONFIG": bool_ok,
        "CLUSTER SET-CONFIG-EPOCH": bool_ok,
        "CLUSTER SETSLOT": bool_ok,
        "CLUSTER SLAVES": parse_cluster_nodes,
        "COMMAND": parse_command,
        "COMMAND COUNT": int,
        "COMMAND GETKEYS": lambda r: list(map(str_if_bytes, r)),
        "CONFIG GET": parse_config_get,
        "CONFIG RESETSTAT": bool_ok,
        "CONFIG SET": bool_ok,
        "DEBUG OBJECT": parse_debug_object,
        "FUNCTION DELETE": bool_ok,
        "FUNCTION FLUSH": bool_ok,
        "FUNCTION RESTORE": bool_ok,
        "GEOHASH": lambda r: list(map(str_if_bytes, r)),
        "GEOPOS": lambda r: list(
            map(lambda ll: (float(ll[0]), float(ll[1])) if ll is not None else None, r)
        ),
        "GEOSEARCH": parse_geosearch_generic,
        "GEORADIUS": parse_geosearch_generic,
        "GEORADIUSBYMEMBER": parse_geosearch_generic,
        "HGETALL": lambda r: r and pairs_to_dict(r) or {},
        "HSCAN": parse_hscan,
        "INFO": parse_info,
        "LASTSAVE": timestamp_to_datetime,
        "MEMORY PURGE": bool_ok,
        "MEMORY STATS": parse_memory_stats,
        "MEMORY USAGE": int_or_none,
        "MODULE LOAD": parse_module_result,
        "MODULE UNLOAD": parse_module_result,
        "MODULE LIST": lambda r: [pairs_to_dict(m) for m in r],
        "OBJECT": parse_object,
        "PING": lambda r: str_if_bytes(r) == "PONG",
        "QUIT": bool_ok,
        "STRALGO": parse_stralgo,
        "PUBSUB NUMSUB": parse_pubsub_numsub,
        "RANDOMKEY": lambda r: r and r or None,
        "RESET": str_if_bytes,
        "SCAN": parse_scan,
        "SCRIPT EXISTS": lambda r: list(map(bool, r)),
        "SCRIPT FLUSH": bool_ok,
        "SCRIPT KILL": bool_ok,
        "SCRIPT LOAD": str_if_bytes,
        "SENTINEL CKQUORUM": bool_ok,
        "SENTINEL FAILOVER": bool_ok,
        "SENTINEL FLUSHCONFIG": bool_ok,
        "SENTINEL GET-MASTER-ADDR-BY-NAME": parse_sentinel_get_master,
        "SENTINEL MASTER": parse_sentinel_master,
        "SENTINEL MASTERS": parse_sentinel_masters,
        "SENTINEL MONITOR": bool_ok,
        "SENTINEL RESET": bool_ok,
        "SENTINEL REMOVE": bool_ok,
        "SENTINEL SENTINELS": parse_sentinel_slaves_and_sentinels,
        "SENTINEL SET": bool_ok,
        "SENTINEL SLAVES": parse_sentinel_slaves_and_sentinels,
        "SET": parse_set_result,
        "SLOWLOG GET": parse_slowlog_get,
        "SLOWLOG LEN": int,
        "SLOWLOG RESET": bool_ok,
        "SSCAN": parse_scan,
        "TIME": lambda x: (int(x[0]), int(x[1])),
        "XCLAIM": parse_xclaim,
        "XAUTOCLAIM": parse_xautoclaim,
        "XGROUP CREATE": bool_ok,
        "XGROUP DELCONSUMER": int,
        "XGROUP DESTROY": bool,
        "XGROUP SETID": bool_ok,
        "XINFO CONSUMERS": parse_list_of_dicts,
        "XINFO GROUPS": parse_list_of_dicts,
        "XINFO STREAM": parse_xinfo_stream,
        "XPENDING": parse_xpending,
        "ZADD": parse_zadd,
        "ZSCAN": parse_zscan,
        "ZMSCORE": parse_zmscore,
    }


class Redis(AbstractRedis, RedisModuleCommands, CoreCommands, SentinelCommands):
    """
    Implementation of the Redis protocol.

    This abstract class provides a Python interface to all Redis commands
    and an implementation of the Redis protocol.

    Pipelines derive from this, implementing how
    the commands are sent and received to the Redis server. Based on
    configuration, an instance will either use a ConnectionPool, or
    Connection object to talk to redis.

    It is not safe to pass PubSub or Pipeline objects between threads.
    """

    @classmethod
    def from_url(cls, url, **kwargs):
        """
        Return a Redis client object configured from the given URL

        For example::

            redis://[[username]:[password]]@localhost:6379/0
            rediss://[[username]:[password]]@localhost:6379/0
            unix://[username@]/path/to/socket.sock?db=0[&password=password]

        Three URL schemes are supported:

        - `redis://` creates a TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/redis>
        - `rediss://` creates a SSL wrapped TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/rediss>
        - ``unix://``: creates a Unix Domain Socket connection.

        The username, password, hostname, path and all querystring values
        are passed through urllib.parse.unquote in order to replace any
        percent-encoded values with their corresponding characters.

        There are several ways to specify a database number. The first value
        found will be used:

            1. A ``db`` querystring option, e.g. redis://localhost?db=0
            2. If using the redis:// or rediss:// schemes, the path argument
               of the url, e.g. redis://localhost/0
            3. A ``db`` keyword argument to this function.

        If none of these options are specified, the default db=0 is used.

        All querystring options are cast to their appropriate Python types.
        Boolean arguments can be specified with string values "True"/"False"
        or "Yes"/"No". Values that cannot be properly cast cause a
        ``ValueError`` to be raised. Once parsed, the querystring arguments
        and keyword arguments are passed to the ``ConnectionPool``'s
        class initializer. In the case of conflicting arguments, querystring
        arguments always win.

        """
        connection_pool = ConnectionPool.from_url(url, **kwargs)
        return cls(connection_pool=connection_pool)

    def __init__(
        self,
        host="localhost",
        port=6379,
        db=0,
        password=None,
        socket_timeout=None,
        socket_connect_timeout=None,
        socket_keepalive=None,
        socket_keepalive_options=None,
        connection_pool=None,
        unix_socket_path=None,
        encoding="utf-8",
        encoding_errors="strict",
        charset=None,
        errors=None,
        decode_responses=False,
        retry_on_timeout=False,
        retry_on_error=None,
        ssl=False,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_cert_reqs="required",
        ssl_ca_certs=None,
        ssl_ca_path=None,
        ssl_ca_data=None,
        ssl_check_hostname=False,
        ssl_password=None,
        ssl_validate_ocsp=False,
        ssl_validate_ocsp_stapled=False,
        ssl_ocsp_context=None,
        ssl_ocsp_expected_cert=None,
        max_connections=None,
        single_connection_client=False,
        health_check_interval=0,
        client_name=None,
        username=None,
        retry=None,
        redis_connect_func=None,
        credential_provider: Optional[CredentialProvider] = None,
    ):
        """
        Initialize a new Redis client.
        To specify a retry policy for specific errors, first set
        `retry_on_error` to a list of the error/s to retry on, then set
        `retry` to a valid `Retry` object.
        To retry on TimeoutError, `retry_on_timeout` can also be set to `True`.

        Args:

        single_connection_client:
            if `True`, connection pool is not used. In that case `Redis`
            instance use is not thread safe.
        """
        if not connection_pool:
            if charset is not None:
                warnings.warn(
                    DeprecationWarning(
                        '"charset" is deprecated. Use "encoding" instead'
                    )
                )
                encoding = charset
            if errors is not None:
                warnings.warn(
                    DeprecationWarning(
                        '"errors" is deprecated. Use "encoding_errors" instead'
                    )
                )
                encoding_errors = errors
            if not retry_on_error:
                retry_on_error = []
            if retry_on_timeout is True:
                retry_on_error.append(TimeoutError)
            kwargs = {
                "db": db,
                "username": username,
                "password": password,
                "socket_timeout": socket_timeout,
                "encoding": encoding,
                "encoding_errors": encoding_errors,
                "decode_responses": decode_responses,
                "retry_on_error": retry_on_error,
                "retry": copy.deepcopy(retry),
                "max_connections": max_connections,
                "health_check_interval": health_check_interval,
                "client_name": client_name,
                "redis_connect_func": redis_connect_func,
                "credential_provider": credential_provider,
            }
            # based on input, setup appropriate connection args
            if unix_socket_path is not None:
                kwargs.update(
                    {
                        "path": unix_socket_path,
                        "connection_class": UnixDomainSocketConnection,
                    }
                )
            else:
                # TCP specific options
                kwargs.update(
                    {
                        "host": host,
                        "port": port,
                        "socket_connect_timeout": socket_connect_timeout,
                        "socket_keepalive": socket_keepalive,
                        "socket_keepalive_options": socket_keepalive_options,
                    }
                )

                if ssl:
                    kwargs.update(
                        {
                            "connection_class": SSLConnection,
                            "ssl_keyfile": ssl_keyfile,
                            "ssl_certfile": ssl_certfile,
                            "ssl_cert_reqs": ssl_cert_reqs,
                            "ssl_ca_certs": ssl_ca_certs,
                            "ssl_ca_data": ssl_ca_data,
                            "ssl_check_hostname": ssl_check_hostname,
                            "ssl_password": ssl_password,
                            "ssl_ca_path": ssl_ca_path,
                            "ssl_validate_ocsp_stapled": ssl_validate_ocsp_stapled,
                            "ssl_validate_ocsp": ssl_validate_ocsp,
                            "ssl_ocsp_context": ssl_ocsp_context,
                            "ssl_ocsp_expected_cert": ssl_ocsp_expected_cert,
                        }
                    )
            connection_pool = ConnectionPool(**kwargs)
        self.connection_pool = connection_pool
        self.connection = None
        if single_connection_client:
            self.connection = self.connection_pool.get_connection("_")

        self.response_callbacks = CaseInsensitiveDict(self.__class__.RESPONSE_CALLBACKS)

    def __repr__(self):
        return f"{type(self).__name__}<{repr(self.connection_pool)}>"

    def get_encoder(self):
        """Get the connection pool's encoder"""
        return self.connection_pool.get_encoder()

    def get_connection_kwargs(self):
        """Get the connection's key-word arguments"""
        return self.connection_pool.connection_kwargs

    def get_retry(self) -> Optional["Retry"]:
        return self.get_connection_kwargs().get("retry")

    def set_retry(self, retry: "Retry") -> None:
        self.get_connection_kwargs().update({"retry": retry})
        self.connection_pool.set_retry(retry)

    def set_response_callback(self, command, callback):
        """Set a custom Response Callback"""
        self.response_callbacks[command] = callback

    def load_external_module(self, funcname, func):
        """
        This function can be used to add externally defined redis modules,
        and their namespaces to the redis client.

        funcname - A string containing the name of the function to create
        func - The function, being added to this class.

        ex: Assume that one has a custom redis module named foomod that
        creates command named 'foo.dothing' and 'foo.anotherthing' in redis.
        To load function functions into this namespace:

        from redis import Redis
        from foomodule import F
        r = Redis()
        r.load_external_module("foo", F)
        r.foo().dothing('your', 'arguments')

        For a concrete example see the reimport of the redisjson module in
        tests/test_connection.py::test_loading_external_modules
        """
        setattr(self, funcname, func)

    def pipeline(self, transaction=True, shard_hint=None):
        """
        Return a new pipeline object that can queue multiple commands for
        later execution. ``transaction`` indicates whether all commands
        should be executed atomically. Apart from making a group of operations
        atomic, pipelines are useful for reducing the back-and-forth overhead
        between the client and server.
        """
        return Pipeline(
            self.connection_pool, self.response_callbacks, transaction, shard_hint
        )

    def transaction(self, func, *watches, **kwargs):
        """
        Convenience method for executing the callable `func` as a transaction
        while watching all keys specified in `watches`. The 'func' callable
        should expect a single argument which is a Pipeline object.
        """
        shard_hint = kwargs.pop("shard_hint", None)
        value_from_callable = kwargs.pop("value_from_callable", False)
        watch_delay = kwargs.pop("watch_delay", None)
        with self.pipeline(True, shard_hint) as pipe:
            while True:
                try:
                    if watches:
                        pipe.watch(*watches)
                    func_value = func(pipe)
                    exec_value = pipe.execute()
                    return func_value if value_from_callable else exec_value
                except WatchError:
                    if watch_delay is not None and watch_delay > 0:
                        time.sleep(watch_delay)
                    continue

    def lock(
        self,
        name,
        timeout=None,
        sleep=0.1,
        blocking=True,
        blocking_timeout=None,
        lock_class=None,
        thread_local=True,
    ):
        """
        Return a new Lock object using key ``name`` that mimics
        the behavior of threading.Lock.

        If specified, ``timeout`` indicates a maximum life for the lock.
        By default, it will remain locked until release() is called.

        ``sleep`` indicates the amount of time to sleep per loop iteration
        when the lock is in blocking mode and another client is currently
        holding the lock.

        ``blocking`` indicates whether calling ``acquire`` should block until
        the lock has been acquired or to fail immediately, causing ``acquire``
        to return False and the lock not being acquired. Defaults to True.
        Note this value can be overridden by passing a ``blocking``
        argument to ``acquire``.

        ``blocking_timeout`` indicates the maximum amount of time in seconds to
        spend trying to acquire the lock. A value of ``None`` indicates
        continue trying forever. ``blocking_timeout`` can be specified as a
        float or integer, both representing the number of seconds to wait.

        ``lock_class`` forces the specified lock implementation. Note that as
        of redis-py 3.0, the only lock class we implement is ``Lock`` (which is
        a Lua-based lock). So, it's unlikely you'll need this parameter, unless
        you have created your own custom lock class.

        ``thread_local`` indicates whether the lock token is placed in
        thread-local storage. By default, the token is placed in thread local
        storage so that a thread only sees its token, not a token set by
        another thread. Consider the following timeline:

            time: 0, thread-1 acquires `my-lock`, with a timeout of 5 seconds.
                     thread-1 sets the token to "abc"
            time: 1, thread-2 blocks trying to acquire `my-lock` using the
                     Lock instance.
            time: 5, thread-1 has not yet completed. redis expires the lock
                     key.
            time: 5, thread-2 acquired `my-lock` now that it's available.
                     thread-2 sets the token to "xyz"
            time: 6, thread-1 finishes its work and calls release(). if the
                     token is *not* stored in thread local storage, then
                     thread-1 would see the token value as "xyz" and would be
                     able to successfully release the thread-2's lock.

        In some use cases it's necessary to disable thread local storage. For
        example, if you have code where one thread acquires a lock and passes
        that lock instance to a worker thread to release later. If thread
        local storage isn't disabled in this case, the worker thread won't see
        the token set by the thread that acquired the lock. Our assumption
        is that these cases aren't common and as such default to using
        thread local storage."""
        if lock_class is None:
            lock_class = Lock
        return lock_class(
            self,
            name,
            timeout=timeout,
            sleep=sleep,
            blocking=blocking,
            blocking_timeout=blocking_timeout,
            thread_local=thread_local,
        )

    def pubsub(self, **kwargs):
        """
        Return a Publish/Subscribe object. With this object, you can
        subscribe to channels and listen for messages that get published to
        them.
        """
        return PubSub(self.connection_pool, **kwargs)

    def monitor(self):
        return Monitor(self.connection_pool)

    def client(self):
        return self.__class__(
            connection_pool=self.connection_pool, single_connection_client=True
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        # In case a connection property does not yet exist
        # (due to a crash earlier in the Redis() constructor), return
        # immediately as there is nothing to clean-up.
        if not hasattr(self, "connection"):
            return

        conn = self.connection
        if conn:
            self.connection = None
            self.connection_pool.release(conn)

    def _send_command_parse_response(self, conn, command_name, *args, **options):
        """
        Send a command and parse the response
        """
        conn.send_command(*args)
        return self.parse_response(conn, command_name, **options)

    def _disconnect_raise(self, conn, error):
        """
        Close the connection and raise an exception
        if retry_on_error is not set or the error
        is not one of the specified error types
        """
        conn.disconnect()
        if (
            conn.retry_on_error is None
            or isinstance(error, tuple(conn.retry_on_error)) is False
        ):
            raise error

    # COMMAND EXECUTION AND PROTOCOL PARSING
    def execute_command(self, *args, **options):
        """Execute a command and return a parsed response"""
        pool = self.connection_pool
        command_name = args[0]
        conn = self.connection or pool.get_connection(command_name, **options)

        try:
            return conn.retry.call_with_retry(
                lambda: self._send_command_parse_response(
                    conn, command_name, *args, **options
                ),
                lambda error: self._disconnect_raise(conn, error),
            )
        finally:
            if not self.connection:
                pool.release(conn)

    def parse_response(self, connection, command_name, **options):
        """Parses a response from the Redis server"""
        try:
            if NEVER_DECODE in options:
                response = connection.read_response(disable_decoding=True)
                options.pop(NEVER_DECODE)
            else:
                response = connection.read_response()
        except ResponseError:
            if EMPTY_RESPONSE in options:
                return options[EMPTY_RESPONSE]
            raise

        if EMPTY_RESPONSE in options:
            options.pop(EMPTY_RESPONSE)

        if command_name in self.response_callbacks:
            return self.response_callbacks[command_name](response, **options)
        return response


StrictRedis = Redis


class Monitor:
    """
    Monitor is useful for handling the MONITOR command to the redis server.
    next_command() method returns one command from monitor
    listen() method yields commands from monitor.
    """

    monitor_re = re.compile(r"\[(\d+) (.*)\] (.*)")
    command_re = re.compile(r'"(.*?)(?<!\\)"')

    def __init__(self, connection_pool):
        self.connection_pool = connection_pool
        self.connection = self.connection_pool.get_connection("MONITOR")

    def __enter__(self):
        self.connection.send_command("MONITOR")
        # check that monitor returns 'OK', but don't return it to user
        response = self.connection.read_response()
        if not bool_ok(response):
            raise RedisError(f"MONITOR failed: {response}")
        return self

    def __exit__(self, *args):
        self.connection.disconnect()
        self.connection_pool.release(self.connection)

    def next_command(self):
        """Parse the response from a monitor command"""
        response = self.connection.read_response()
        if isinstance(response, bytes):
            response = self.connection.encoder.decode(response, force=True)
        command_time, command_data = response.split(" ", 1)
        m = self.monitor_re.match(command_data)
        db_id, client_info, command = m.groups()
        command = " ".join(self.command_re.findall(command))
        # Redis escapes double quotes because each piece of the command
        # string is surrounded by double quotes. We don't have that
        # requirement so remove the escaping and leave the quote.
        command = command.replace('\\"', '"')

        if client_info == "lua":
            client_address = "lua"
            client_port = ""
            client_type = "lua"
        elif client_info.startswith("unix"):
            client_address = "unix"
            client_port = client_info[5:]
            client_type = "unix"
        else:
            # use rsplit as ipv6 addresses contain colons
            client_address, client_port = client_info.rsplit(":", 1)
            client_type = "tcp"
        return {
            "time": float(command_time),
            "db": int(db_id),
            "client_address": client_address,
            "client_port": client_port,
            "client_type": client_type,
            "command": command,
        }

    def listen(self):
        """Listen for commands coming to the server."""
        while True:
            yield self.next_command()


class PubSub:
    """
    PubSub provides publish, subscribe and listen support to Redis channels.

    After subscribing to one or more channels, the listen() method will block
    until a message arrives on one of the subscribed channels. That message
    will be returned and it's safe to start listening again.
    """

    PUBLISH_MESSAGE_TYPES = ("message", "pmessage")
    UNSUBSCRIBE_MESSAGE_TYPES = ("unsubscribe", "punsubscribe")
    HEALTH_CHECK_MESSAGE = "redis-py-health-check"

    def __init__(
        self,
        connection_pool,
        shard_hint=None,
        ignore_subscribe_messages=False,
        encoder=None,
    ):
        self.connection_pool = connection_pool
        self.shard_hint = shard_hint
        self.ignore_subscribe_messages = ignore_subscribe_messages
        self.connection = None
        self.subscribed_event = threading.Event()
        # we need to know the encoding options for this connection in order
        # to lookup channel and pattern names for callback handlers.
        self.encoder = encoder
        if self.encoder is None:
            self.encoder = self.connection_pool.get_encoder()
        self.health_check_response_b = self.encoder.encode(self.HEALTH_CHECK_MESSAGE)
        if self.encoder.decode_responses:
            self.health_check_response = ["pong", self.HEALTH_CHECK_MESSAGE]
        else:
            self.health_check_response = [b"pong", self.health_check_response_b]
        self.reset()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reset()

    def __del__(self):
        try:
            # if this object went out of scope prior to shutting down
            # subscriptions, close the connection manually before
            # returning it to the connection pool
            self.reset()
        except Exception:
            pass

    def reset(self):
        if self.connection:
            self.connection.disconnect()
            self.connection.clear_connect_callbacks()
            self.connection_pool.release(self.connection)
            self.connection = None
        self.channels = {}
        self.health_check_response_counter = 0
        self.pending_unsubscribe_channels = set()
        self.patterns = {}
        self.pending_unsubscribe_patterns = set()
        self.subscribed_event.clear()

    def close(self):
        self.reset()

    def on_connect(self, connection):
        "Re-subscribe to any channels and patterns previously subscribed to"
        # NOTE: for python3, we can't pass bytestrings as keyword arguments
        # so we need to decode channel/pattern names back to unicode strings
        # before passing them to [p]subscribe.
        self.pending_unsubscribe_channels.clear()
        self.pending_unsubscribe_patterns.clear()
        if self.channels:
            channels = {}
            for k, v in self.channels.items():
                channels[self.encoder.decode(k, force=True)] = v
            self.subscribe(**channels)
        if self.patterns:
            patterns = {}
            for k, v in self.patterns.items():
                patterns[self.encoder.decode(k, force=True)] = v
            self.psubscribe(**patterns)

    @property
    def subscribed(self):
        """Indicates if there are subscriptions to any channels or patterns"""
        return self.subscribed_event.is_set()

    def execute_command(self, *args):
        """Execute a publish/subscribe command"""

        # NOTE: don't parse the response in this function -- it could pull a
        # legitimate message off the stack if the connection is already
        # subscribed to one or more channels

        if self.connection is None:
            self.connection = self.connection_pool.get_connection(
                "pubsub", self.shard_hint
            )
            # register a callback that re-subscribes to any channels we
            # were listening to when we were disconnected
            self.connection.register_connect_callback(self.on_connect)
        connection = self.connection
        kwargs = {"check_health": not self.subscribed}
        if not self.subscribed:
            self.clean_health_check_responses()
        self._execute(connection, connection.send_command, *args, **kwargs)

    def clean_health_check_responses(self):
        """
        If any health check responses are present, clean them
        """
        ttl = 10
        conn = self.connection
        while self.health_check_response_counter > 0 and ttl > 0:
            if self._execute(conn, conn.can_read, timeout=conn.socket_timeout):
                response = self._execute(conn, conn.read_response)
                if self.is_health_check_response(response):
                    self.health_check_response_counter -= 1
                else:
                    raise PubSubError(
                        "A non health check response was cleaned by "
                        "execute_command: {0}".format(response)
                    )
            ttl -= 1

    def _disconnect_raise_connect(self, conn, error):
        """
        Close the connection and raise an exception
        if retry_on_timeout is not set or the error
        is not a TimeoutError. Otherwise, try to reconnect
        """
        conn.disconnect()
        if not (conn.retry_on_timeout and isinstance(error, TimeoutError)):
            raise error
        conn.connect()

    def _execute(self, conn, command, *args, **kwargs):
        """
        Connect manually upon disconnection. If the Redis server is down,
        this will fail and raise a ConnectionError as desired.
        After reconnection, the ``on_connect`` callback should have been
        called by the # connection to resubscribe us to any channels and
        patterns we were previously listening to
        """
        return conn.retry.call_with_retry(
            lambda: command(*args, **kwargs),
            lambda error: self._disconnect_raise_connect(conn, error),
        )

    def parse_response(self, block=True, timeout=0):
        """Parse the response from a publish/subscribe command"""
        conn = self.connection
        if conn is None:
            raise RuntimeError(
                "pubsub connection not set: "
                "did you forget to call subscribe() or psubscribe()?"
            )

        self.check_health()

        def try_read():
            if not block:
                if not conn.can_read(timeout=timeout):
                    return None
            else:
                conn.connect()
            return conn.read_response()

        response = self._execute(conn, try_read)

        if self.is_health_check_response(response):
            # ignore the health check message as user might not expect it
            self.health_check_response_counter -= 1
            return None
        return response

    def is_health_check_response(self, response):
        """
        Check if the response is a health check response.
        If there are no subscriptions redis responds to PING command with a
        bulk response, instead of a multi-bulk with "pong" and the response.
        """
        return response in [
            self.health_check_response,  # If there was a subscription
            self.health_check_response_b,  # If there wasn't
        ]

    def check_health(self):
        conn = self.connection
        if conn is None:
            raise RuntimeError(
                "pubsub connection not set: "
                "did you forget to call subscribe() or psubscribe()?"
            )

        if conn.health_check_interval and time.time() > conn.next_health_check:
            conn.send_command("PING", self.HEALTH_CHECK_MESSAGE, check_health=False)
            self.health_check_response_counter += 1

    def _normalize_keys(self, data):
        """
        normalize channel/pattern names to be either bytes or strings
        based on whether responses are automatically decoded. this saves us
        from coercing the value for each message coming in.
        """
        encode = self.encoder.encode
        decode = self.encoder.decode
        return {decode(encode(k)): v for k, v in data.items()}

    def psubscribe(self, *args, **kwargs):
        """
        Subscribe to channel patterns. Patterns supplied as keyword arguments
        expect a pattern name as the key and a callable as the value. A
        pattern's callable will be invoked automatically when a message is
        received on that pattern rather than producing a message via
        ``listen()``.
        """
        if args:
            args = list_or_args(args[0], args[1:])
        new_patterns = dict.fromkeys(args)
        new_patterns.update(kwargs)
        ret_val = self.execute_command("PSUBSCRIBE", *new_patterns.keys())
        # update the patterns dict AFTER we send the command. we don't want to
        # subscribe twice to these patterns, once for the command and again
        # for the reconnection.
        new_patterns = self._normalize_keys(new_patterns)
        self.patterns.update(new_patterns)
        if not self.subscribed:
            # Set the subscribed_event flag to True
            self.subscribed_event.set()
            # Clear the health check counter
            self.health_check_response_counter = 0
        self.pending_unsubscribe_patterns.difference_update(new_patterns)
        return ret_val

    def punsubscribe(self, *args):
        """
        Unsubscribe from the supplied patterns. If empty, unsubscribe from
        all patterns.
        """
        if args:
            args = list_or_args(args[0], args[1:])
            patterns = self._normalize_keys(dict.fromkeys(args))
        else:
            patterns = self.patterns
        self.pending_unsubscribe_patterns.update(patterns)
        return self.execute_command("PUNSUBSCRIBE", *args)

    def subscribe(self, *args, **kwargs):
        """
        Subscribe to channels. Channels supplied as keyword arguments expect
        a channel name as the key and a callable as the value. A channel's
        callable will be invoked automatically when a message is received on
        that channel rather than producing a message via ``listen()`` or
        ``get_message()``.
        """
        if args:
            args = list_or_args(args[0], args[1:])
        new_channels = dict.fromkeys(args)
        new_channels.update(kwargs)
        ret_val = self.execute_command("SUBSCRIBE", *new_channels.keys())
        # update the channels dict AFTER we send the command. we don't want to
        # subscribe twice to these channels, once for the command and again
        # for the reconnection.
        new_channels = self._normalize_keys(new_channels)
        self.channels.update(new_channels)
        if not self.subscribed:
            # Set the subscribed_event flag to True
            self.subscribed_event.set()
            # Clear the health check counter
            self.health_check_response_counter = 0
        self.pending_unsubscribe_channels.difference_update(new_channels)
        return ret_val

    def unsubscribe(self, *args):
        """
        Unsubscribe from the supplied channels. If empty, unsubscribe from
        all channels
        """
        if args:
            args = list_or_args(args[0], args[1:])
            channels = self._normalize_keys(dict.fromkeys(args))
        else:
            channels = self.channels
        self.pending_unsubscribe_channels.update(channels)
        return self.execute_command("UNSUBSCRIBE", *args)

    def listen(self):
        "Listen for messages on channels this client has been subscribed to"
        while self.subscribed:
            response = self.handle_message(self.parse_response(block=True))
            if response is not None:
                yield response

    def get_message(self, ignore_subscribe_messages=False, timeout=0.0):
        """
        Get the next message if one is available, otherwise None.

        If timeout is specified, the system will wait for `timeout` seconds
        before returning. Timeout should be specified as a floating point
        number, or None, to wait indefinitely.
        """
        if not self.subscribed:
            # Wait for subscription
            start_time = time.time()
            if self.subscribed_event.wait(timeout) is True:
                # The connection was subscribed during the timeout time frame.
                # The timeout should be adjusted based on the time spent
                # waiting for the subscription
                time_spent = time.time() - start_time
                timeout = max(0.0, timeout - time_spent)
            else:
                # The connection isn't subscribed to any channels or patterns,
                # so no messages are available
                return None

        response = self.parse_response(block=(timeout is None), timeout=timeout)
        if response:
            return self.handle_message(response, ignore_subscribe_messages)
        return None

    def ping(self, message=None):
        """
        Ping the Redis server
        """
        message = "" if message is None else message
        return self.execute_command("PING", message)

    def handle_message(self, response, ignore_subscribe_messages=False):
        """
        Parses a pub/sub message. If the channel or pattern was subscribed to
        with a message handler, the handler is invoked instead of a parsed
        message being returned.
        """
        if response is None:
            return None
        message_type = str_if_bytes(response[0])
        if message_type == "pmessage":
            message = {
                "type": message_type,
                "pattern": response[1],
                "channel": response[2],
                "data": response[3],
            }
        elif message_type == "pong":
            message = {
                "type": message_type,
                "pattern": None,
                "channel": None,
                "data": response[1],
            }
        else:
            message = {
                "type": message_type,
                "pattern": None,
                "channel": response[1],
                "data": response[2],
            }

        # if this is an unsubscribe message, remove it from memory
        if message_type in self.UNSUBSCRIBE_MESSAGE_TYPES:
            if message_type == "punsubscribe":
                pattern = response[1]
                if pattern in self.pending_unsubscribe_patterns:
                    self.pending_unsubscribe_patterns.remove(pattern)
                    self.patterns.pop(pattern, None)
            else:
                channel = response[1]
                if channel in self.pending_unsubscribe_channels:
                    self.pending_unsubscribe_channels.remove(channel)
                    self.channels.pop(channel, None)
            if not self.channels and not self.patterns:
                # There are no subscriptions anymore, set subscribed_event flag
                # to false
                self.subscribed_event.clear()

        if message_type in self.PUBLISH_MESSAGE_TYPES:
            # if there's a message handler, invoke it
            if message_type == "pmessage":
                handler = self.patterns.get(message["pattern"], None)
            else:
                handler = self.channels.get(message["channel"], None)
            if handler:
                handler(message)
                return None
        elif message_type != "pong":
            # this is a subscribe/unsubscribe message. ignore if we don't
            # want them
            if ignore_subscribe_messages or self.ignore_subscribe_messages:
                return None

        return message

    def run_in_thread(self, sleep_time=0, daemon=False, exception_handler=None):
        for channel, handler in self.channels.items():
            if handler is None:
                raise PubSubError(f"Channel: '{channel}' has no handler registered")
        for pattern, handler in self.patterns.items():
            if handler is None:
                raise PubSubError(f"Pattern: '{pattern}' has no handler registered")

        thread = PubSubWorkerThread(
            self, sleep_time, daemon=daemon, exception_handler=exception_handler
        )
        thread.start()
        return thread


class PubSubWorkerThread(threading.Thread):
    def __init__(self, pubsub, sleep_time, daemon=False, exception_handler=None):
        super().__init__()
        self.daemon = daemon
        self.pubsub = pubsub
        self.sleep_time = sleep_time
        self.exception_handler = exception_handler
        self._running = threading.Event()

    def run(self):
        if self._running.is_set():
            return
        self._running.set()
        pubsub = self.pubsub
        sleep_time = self.sleep_time
        while self._running.is_set():
            try:
                pubsub.get_message(ignore_subscribe_messages=True, timeout=sleep_time)
            except BaseException as e:
                if self.exception_handler is None:
                    raise
                self.exception_handler(e, pubsub, self)
        pubsub.close()

    def stop(self):
        # trip the flag so the run loop exits. the run loop will
        # close the pubsub connection, which disconnects the socket
        # and returns the connection to the pool.
        self._running.clear()


class Pipeline(Redis):
    """
    Pipelines provide a way to transmit multiple commands to the Redis server
    in one transmission.  This is convenient for batch processing, such as
    saving all the values in a list to Redis.

    All commands executed within a pipeline are wrapped with MULTI and EXEC
    calls. This guarantees all commands executed in the pipeline will be
    executed atomically.

    Any command raising an exception does *not* halt the execution of
    subsequent commands in the pipeline. Instead, the exception is caught
    and its instance is placed into the response list returned by execute().
    Code iterating over the response list should be able to deal with an
    instance of an exception as a potential value. In general, these will be
    ResponseError exceptions, such as those raised when issuing a command
    on a key of a different datatype.
    """

    UNWATCH_COMMANDS = {"DISCARD", "EXEC", "UNWATCH"}

    def __init__(self, connection_pool, response_callbacks, transaction, shard_hint):
        self.connection_pool = connection_pool
        self.connection = None
        self.response_callbacks = response_callbacks
        self.transaction = transaction
        self.shard_hint = shard_hint

        self.watching = False
        self.reset()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reset()

    def __del__(self):
        try:
            self.reset()
        except Exception:
            pass

    def __len__(self):
        return len(self.command_stack)

    def __bool__(self):
        """Pipeline instances should always evaluate to True"""
        return True

    def reset(self):
        self.command_stack = []
        self.scripts = set()
        # make sure to reset the connection state in the event that we were
        # watching something
        if self.watching and self.connection:
            try:
                # call this manually since our unwatch or
                # immediate_execute_command methods can call reset()
                self.connection.send_command("UNWATCH")
                self.connection.read_response()
            except ConnectionError:
                # disconnect will also remove any previous WATCHes
                self.connection.disconnect()
        # clean up the other instance attributes
        self.watching = False
        self.explicit_transaction = False
        # we can safely return the connection to the pool here since we're
        # sure we're no longer WATCHing anything
        if self.connection:
            self.connection_pool.release(self.connection)
            self.connection = None

    def multi(self):
        """
        Start a transactional block of the pipeline after WATCH commands
        are issued. End the transactional block with `execute`.
        """
        if self.explicit_transaction:
            raise RedisError("Cannot issue nested calls to MULTI")
        if self.command_stack:
            raise RedisError(
                "Commands without an initial WATCH have already been issued"
            )
        self.explicit_transaction = True

    def execute_command(self, *args, **kwargs):
        if (self.watching or args[0] == "WATCH") and not self.explicit_transaction:
            return self.immediate_execute_command(*args, **kwargs)
        return self.pipeline_execute_command(*args, **kwargs)

    def _disconnect_reset_raise(self, conn, error):
        """
        Close the connection, reset watching state and
        raise an exception if we were watching,
        retry_on_timeout is not set,
        or the error is not a TimeoutError
        """
        conn.disconnect()
        # if we were already watching a variable, the watch is no longer
        # valid since this connection has died. raise a WatchError, which
        # indicates the user should retry this transaction.
        if self.watching:
            self.reset()
            raise WatchError(
                "A ConnectionError occurred on while watching one or more keys"
            )
        # if retry_on_timeout is not set, or the error is not
        # a TimeoutError, raise it
        if not (conn.retry_on_timeout and isinstance(error, TimeoutError)):
            self.reset()
            raise

    def immediate_execute_command(self, *args, **options):
        """
        Execute a command immediately, but don't auto-retry on a
        ConnectionError if we're already WATCHing a variable. Used when
        issuing WATCH or subsequent commands retrieving their values but before
        MULTI is called.
        """
        command_name = args[0]
        conn = self.connection
        # if this is the first call, we need a connection
        if not conn:
            conn = self.connection_pool.get_connection(command_name, self.shard_hint)
            self.connection = conn

        return conn.retry.call_with_retry(
            lambda: self._send_command_parse_response(
                conn, command_name, *args, **options
            ),
            lambda error: self._disconnect_reset_raise(conn, error),
        )

    def pipeline_execute_command(self, *args, **options):
        """
        Stage a command to be executed when execute() is next called

        Returns the current Pipeline object back so commands can be
        chained together, such as:

        pipe = pipe.set('foo', 'bar').incr('baz').decr('bang')

        At some other point, you can then run: pipe.execute(),
        which will execute all commands queued in the pipe.
        """
        self.command_stack.append((args, options))
        return self

    def _execute_transaction(self, connection, commands, raise_on_error):
        cmds = chain([(("MULTI",), {})], commands, [(("EXEC",), {})])
        all_cmds = connection.pack_commands(
            [args for args, options in cmds if EMPTY_RESPONSE not in options]
        )
        connection.send_packed_command(all_cmds)
        errors = []

        # parse off the response for MULTI
        # NOTE: we need to handle ResponseErrors here and continue
        # so that we read all the additional command messages from
        # the socket
        try:
            self.parse_response(connection, "_")
        except ResponseError as e:
            errors.append((0, e))

        # and all the other commands
        for i, command in enumerate(commands):
            if EMPTY_RESPONSE in command[1]:
                errors.append((i, command[1][EMPTY_RESPONSE]))
            else:
                try:
                    self.parse_response(connection, "_")
                except ResponseError as e:
                    self.annotate_exception(e, i + 1, command[0])
                    errors.append((i, e))

        # parse the EXEC.
        try:
            response = self.parse_response(connection, "_")
        except ExecAbortError:
            if errors:
                raise errors[0][1]
            raise

        # EXEC clears any watched keys
        self.watching = False

        if response is None:
            raise WatchError("Watched variable changed.")

        # put any parse errors into the response
        for i, e in errors:
            response.insert(i, e)

        if len(response) != len(commands):
            self.connection.disconnect()
            raise ResponseError(
                "Wrong number of response items from pipeline execution"
            )

        # find any errors in the response and raise if necessary
        if raise_on_error:
            self.raise_first_error(commands, response)

        # We have to run response callbacks manually
        data = []
        for r, cmd in zip(response, commands):
            if not isinstance(r, Exception):
                args, options = cmd
                command_name = args[0]
                if command_name in self.response_callbacks:
                    r = self.response_callbacks[command_name](r, **options)
            data.append(r)
        return data

    def _execute_pipeline(self, connection, commands, raise_on_error):
        # build up all commands into a single request to increase network perf
        all_cmds = connection.pack_commands([args for args, _ in commands])
        connection.send_packed_command(all_cmds)

        response = []
        for args, options in commands:
            try:
                response.append(self.parse_response(connection, args[0], **options))
            except ResponseError as e:
                response.append(e)

        if raise_on_error:
            self.raise_first_error(commands, response)
        return response

    def raise_first_error(self, commands, response):
        for i, r in enumerate(response):
            if isinstance(r, ResponseError):
                self.annotate_exception(r, i + 1, commands[i][0])
                raise r

    def annotate_exception(self, exception, number, command):
        cmd = " ".join(map(safe_str, command))
        msg = (
            f"Command # {number} ({cmd}) of pipeline "
            f"caused error: {exception.args[0]}"
        )
        exception.args = (msg,) + exception.args[1:]

    def parse_response(self, connection, command_name, **options):
        result = Redis.parse_response(self, connection, command_name, **options)
        if command_name in self.UNWATCH_COMMANDS:
            self.watching = False
        elif command_name == "WATCH":
            self.watching = True
        return result

    def load_scripts(self):
        # make sure all scripts that are about to be run on this pipeline exist
        scripts = list(self.scripts)
        immediate = self.immediate_execute_command
        shas = [s.sha for s in scripts]
        # we can't use the normal script_* methods because they would just
        # get buffered in the pipeline.
        exists = immediate("SCRIPT EXISTS", *shas)
        if not all(exists):
            for s, exist in zip(scripts, exists):
                if not exist:
                    s.sha = immediate("SCRIPT LOAD", s.script)

    def _disconnect_raise_reset(self, conn, error):
        """
        Close the connection, raise an exception if we were watching,
        and raise an exception if retry_on_timeout is not set,
        or the error is not a TimeoutError
        """
        conn.disconnect()
        # if we were watching a variable, the watch is no longer valid
        # since this connection has died. raise a WatchError, which
        # indicates the user should retry this transaction.
        if self.watching:
            raise WatchError(
                "A ConnectionError occurred on while watching one or more keys"
            )
        # if retry_on_timeout is not set, or the error is not
        # a TimeoutError, raise it
        if not (conn.retry_on_timeout and isinstance(error, TimeoutError)):
            self.reset()
            raise

    def execute(self, raise_on_error=True):
        """Execute all the commands in the current pipeline"""
        stack = self.command_stack
        if not stack and not self.watching:
            return []
        if self.scripts:
            self.load_scripts()
        if self.transaction or self.explicit_transaction:
            execute = self._execute_transaction
        else:
            execute = self._execute_pipeline

        conn = self.connection
        if not conn:
            conn = self.connection_pool.get_connection("MULTI", self.shard_hint)
            # assign to self.connection so reset() releases the connection
            # back to the pool after we're done
            self.connection = conn

        try:
            return conn.retry.call_with_retry(
                lambda: execute(conn, stack, raise_on_error),
                lambda error: self._disconnect_raise_reset(conn, error),
            )
        finally:
            self.reset()

    def discard(self):
        """
        Flushes all previously queued commands
        See: https://redis.io/commands/DISCARD
        """
        self.execute_command("DISCARD")

    def watch(self, *names):
        """Watches the values at keys ``names``"""
        if self.explicit_transaction:
            raise RedisError("Cannot issue a WATCH after a MULTI")
        return self.execute_command("WATCH", *names)

    def unwatch(self):
        """Unwatches all previously specified keys"""
        return self.watching and self.execute_command("UNWATCH") or True
