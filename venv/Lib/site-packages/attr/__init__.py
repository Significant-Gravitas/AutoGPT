# SPDX-License-Identifier: MIT

import sys
import warnings

from functools import partial

from . import converters, exceptions, filters, setters, validators
from ._cmp import cmp_using
from ._config import get_run_validators, set_run_validators
from ._funcs import asdict, assoc, astuple, evolve, has, resolve_types
from ._make import (
    NOTHING,
    Attribute,
    Factory,
    attrib,
    attrs,
    fields,
    fields_dict,
    make_class,
    validate,
)
from ._next_gen import define, field, frozen, mutable
from ._version_info import VersionInfo


if sys.version_info < (3, 7):  # pragma: no cover
    warnings.warn(
        "Running attrs on Python 3.6 is deprecated & we intend to drop "
        "support soon. If that's a problem for you, please let us know why & "
        "we MAY re-evaluate: <https://github.com/python-attrs/attrs/pull/993>",
        DeprecationWarning,
    )

__version__ = "22.2.0"
__version_info__ = VersionInfo._from_version_string(__version__)

__title__ = "attrs"
__description__ = "Classes Without Boilerplate"
__url__ = "https://www.attrs.org/"
__uri__ = __url__
__doc__ = __description__ + " <" + __uri__ + ">"

__author__ = "Hynek Schlawack"
__email__ = "hs@ox.cx"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2015 Hynek Schlawack"


s = attributes = attrs
ib = attr = attrib
dataclass = partial(attrs, auto_attribs=True)  # happy Easter ;)


class AttrsInstance:
    pass


__all__ = [
    "Attribute",
    "AttrsInstance",
    "Factory",
    "NOTHING",
    "asdict",
    "assoc",
    "astuple",
    "attr",
    "attrib",
    "attributes",
    "attrs",
    "cmp_using",
    "converters",
    "define",
    "evolve",
    "exceptions",
    "field",
    "fields",
    "fields_dict",
    "filters",
    "frozen",
    "get_run_validators",
    "has",
    "ib",
    "make_class",
    "mutable",
    "resolve_types",
    "s",
    "set_run_validators",
    "setters",
    "validate",
    "validators",
]
