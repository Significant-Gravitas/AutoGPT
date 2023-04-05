from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

# Because we need to type our own stuff, we have to make everything from
# attr explicitly public too.
from attr import __author__ as __author__
from attr import __copyright__ as __copyright__
from attr import __description__ as __description__
from attr import __email__ as __email__
from attr import __license__ as __license__
from attr import __title__ as __title__
from attr import __url__ as __url__
from attr import __version__ as __version__
from attr import __version_info__ as __version_info__
from attr import _FilterType
from attr import assoc as assoc
from attr import Attribute as Attribute
from attr import AttrsInstance as AttrsInstance
from attr import cmp_using as cmp_using
from attr import converters as converters
from attr import define as define
from attr import evolve as evolve
from attr import exceptions as exceptions
from attr import Factory as Factory
from attr import field as field
from attr import fields as fields
from attr import fields_dict as fields_dict
from attr import filters as filters
from attr import frozen as frozen
from attr import has as has
from attr import make_class as make_class
from attr import mutable as mutable
from attr import NOTHING as NOTHING
from attr import resolve_types as resolve_types
from attr import setters as setters
from attr import validate as validate
from attr import validators as validators

# TODO: see definition of attr.asdict/astuple
def asdict(
    inst: Any,
    recurse: bool = ...,
    filter: Optional[_FilterType[Any]] = ...,
    dict_factory: Type[Mapping[Any, Any]] = ...,
    retain_collection_types: bool = ...,
    value_serializer: Optional[
        Callable[[type, Attribute[Any], Any], Any]
    ] = ...,
    tuple_keys: bool = ...,
) -> Dict[str, Any]: ...

# TODO: add support for returning NamedTuple from the mypy plugin
def astuple(
    inst: Any,
    recurse: bool = ...,
    filter: Optional[_FilterType[Any]] = ...,
    tuple_factory: Type[Sequence[Any]] = ...,
    retain_collection_types: bool = ...,
) -> Tuple[Any, ...]: ...
