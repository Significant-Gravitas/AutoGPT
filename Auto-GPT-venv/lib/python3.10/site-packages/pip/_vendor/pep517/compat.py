"""Python 2/3 compatibility"""
import io
import json
import sys


# Handle reading and writing JSON in UTF-8, on Python 3 and 2.

if sys.version_info[0] >= 3:
    # Python 3
    def write_json(obj, path, **kwargs):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, **kwargs)

    def read_json(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

else:
    # Python 2
    def write_json(obj, path, **kwargs):
        with open(path, 'wb') as f:
            json.dump(obj, f, encoding='utf-8', **kwargs)

    def read_json(path):
        with open(path, 'rb') as f:
            return json.load(f)


# FileNotFoundError

try:
    FileNotFoundError = FileNotFoundError
except NameError:
    FileNotFoundError = IOError


if sys.version_info < (3, 6):
    from toml import load as _toml_load  # noqa: F401

    def toml_load(f):
        w = io.TextIOWrapper(f, encoding="utf8", newline="")
        try:
            return _toml_load(w)
        finally:
            w.detach()

    from toml import TomlDecodeError as TOMLDecodeError  # noqa: F401
else:
    from pip._vendor.tomli import load as toml_load  # noqa: F401
    from pip._vendor.tomli import TOMLDecodeError  # noqa: F401
