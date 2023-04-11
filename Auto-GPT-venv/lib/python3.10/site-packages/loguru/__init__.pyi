"""
.. |str| replace:: :class:`str`
.. |namedtuple| replace:: :func:`namedtuple<collections.namedtuple>`
.. |dict| replace:: :class:`dict`

.. |Logger| replace:: :class:`~loguru._logger.Logger`
.. |catch| replace:: :meth:`~loguru._logger.Logger.catch()`
.. |contextualize| replace:: :meth:`~loguru._logger.Logger.contextualize()`
.. |complete| replace:: :meth:`~loguru._logger.Logger.complete()`
.. |bind| replace:: :meth:`~loguru._logger.Logger.bind()`
.. |patch| replace:: :meth:`~loguru._logger.Logger.patch()`
.. |opt| replace:: :meth:`~loguru._logger.Logger.opt()`
.. |level| replace:: :meth:`~loguru._logger.Logger.level()`

.. _stub file: https://www.python.org/dev/peps/pep-0484/#stub-files
.. _string literals: https://www.python.org/dev/peps/pep-0484/#forward-references
.. _postponed evaluation of annotations: https://www.python.org/dev/peps/pep-0563/
.. |future| replace:: ``__future__``
.. _future: https://www.python.org/dev/peps/pep-0563/#enabling-the-future-behavior-in-python-3-7
.. |loguru-mypy| replace:: ``loguru-mypy``
.. _loguru-mypy: https://github.com/kornicameister/loguru-mypy
.. |documentation of loguru-mypy| replace:: documentation of ``loguru-mypy``
.. _documentation of loguru-mypy:
    https://github.com/kornicameister/loguru-mypy/blob/master/README.md
.. _@kornicameister: https://github.com/kornicameister

Loguru relies on a `stub file`_ to document its types. This implies that these types are not
accessible during execution of your program, however they can be used by type checkers and IDE.
Also, this means that your Python interpreter has to support `postponed evaluation of annotations`_
to prevent error at runtime. This is achieved with a |future|_ import in Python 3.7+ or by using
`string literals`_ for earlier versions.

A basic usage example could look like this:

.. code-block:: python

    from __future__ import annotations

    import loguru
    from loguru import logger

    def good_sink(message: loguru.Message):
        print("My name is", message.record["name"])

    def bad_filter(record: loguru.Record):
        return record["invalid"]

    logger.add(good_sink, filter=bad_filter)


.. code-block:: bash

    $ mypy test.py
    test.py:8: error: TypedDict "Record" has no key 'invalid'
    Found 1 error in 1 file (checked 1 source file)

There are several internal types to which you can be exposed using Loguru's public API, they are
listed here and might be useful to type hint your code:

- ``Logger``: the usual |logger| object (also returned by |opt|, |bind| and |patch|).
- ``Message``: the formatted logging message sent to the sinks (a |str| with ``record``
  attribute).
- ``Record``: the |dict| containing all contextual information of the logged message.
- ``Level``: the |namedtuple| returned by |level| (with ``name``, ``no``, ``color`` and ``icon``
  attributes).
- ``Catcher``: the context decorator returned by |catch|.
- ``Contextualizer``: the context decorator returned by |contextualize|.
- ``AwaitableCompleter``: the awaitable object returned by |complete|.
- ``RecordFile``: the ``record["file"]`` with ``name`` and ``path`` attributes.
- ``RecordLevel``: the ``record["level"]`` with ``name``, ``no`` and ``icon`` attributes.
- ``RecordThread``: the ``record["thread"]`` with ``id`` and ``name`` attributes.
- ``RecordProcess``: the ``record["process"]`` with ``id`` and ``name`` attributes.
- ``RecordException``: the ``record["exception"]`` with ``type``, ``value`` and ``traceback``
  attributes.

If that is not enough, one can also use the |loguru-mypy|_ library developed by `@kornicameister`_.
Plugin can be installed separately using::

    pip install loguru-mypy

It helps to catch several possible runtime errors by performing additional checks like:

- ``opt(lazy=True)`` loggers accepting only ``typing.Callable[[], typing.Any]`` arguments
- ``opt(record=True)`` loggers wrongly calling log handler like so ``logger.info(..., record={})``
- and even more...

For more details, go to official |documentation of loguru-mypy|_.
"""

import sys
from asyncio import AbstractEventLoop
from datetime import datetime, time, timedelta
from logging import Handler
from types import TracebackType
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    NamedTuple,
    NewType,
    Optional,
    Pattern,
    Sequence,
    TextIO,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

if sys.version_info >= (3, 5, 3):
    from typing import Awaitable
else:
    from typing_extensions import Awaitable

if sys.version_info >= (3, 6):
    from os import PathLike
    from typing import ContextManager

    PathLikeStr = PathLike[str]
else:
    from pathlib import PurePath as PathLikeStr

    from typing_extensions import ContextManager

if sys.version_info >= (3, 8):
    from typing import Protocol, TypedDict
else:
    from typing_extensions import Protocol, TypedDict

_T = TypeVar("_T")
_F = TypeVar("_F", bound=Callable[..., Any])
ExcInfo = Tuple[Optional[Type[BaseException]], Optional[BaseException], Optional[TracebackType]]

class _GeneratorContextManager(ContextManager[_T], Generic[_T]):
    def __call__(self, func: _F) -> _F: ...
    def __exit__(
        self,
        typ: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]: ...

Catcher = NewType("Catcher", _GeneratorContextManager[None])
Contextualizer = NewType("Contextualizer", _GeneratorContextManager[None])
AwaitableCompleter = Awaitable[None]

class Level(NamedTuple):
    name: str
    no: int
    color: str
    icon: str

class _RecordAttribute:
    def __repr__(self) -> str: ...
    def __format__(self, spec: str) -> str: ...

class RecordFile(_RecordAttribute):
    name: str
    path: str

class RecordLevel(_RecordAttribute):
    name: str
    no: int
    icon: str

class RecordThread(_RecordAttribute):
    id: int
    name: str

class RecordProcess(_RecordAttribute):
    id: int
    name: str

class RecordException(NamedTuple):
    type: Optional[Type[BaseException]]
    value: Optional[BaseException]
    traceback: Optional[TracebackType]

class Record(TypedDict):
    elapsed: timedelta
    exception: Optional[RecordException]
    extra: Dict[Any, Any]
    file: RecordFile
    function: str
    level: RecordLevel
    line: int
    message: str
    module: str
    name: Union[str, None]
    process: RecordProcess
    thread: RecordThread
    time: datetime

class Message(str):
    record: Record

class Writable(Protocol):
    def write(self, message: Message) -> None: ...

FilterDict = Dict[Union[str, None], Union[str, int, bool]]
FilterFunction = Callable[[Record], bool]
FormatFunction = Callable[[Record], str]
PatcherFunction = Callable[[Record], None]
RotationFunction = Callable[[Message, TextIO], bool]
RetentionFunction = Callable[[List[str]], None]
CompressionFunction = Callable[[str], None]

# Actually unusable because TypedDict can't allow extra keys: python/mypy#4617
class _HandlerConfig(TypedDict, total=False):
    sink: Union[str, PathLikeStr, TextIO, Writable, Callable[[Message], None], Handler]
    level: Union[str, int]
    format: Union[str, FormatFunction]
    filter: Optional[Union[str, FilterFunction, FilterDict]]
    colorize: Optional[bool]
    serialize: bool
    backtrace: bool
    diagnose: bool
    enqueue: bool
    catch: bool

class LevelConfig(TypedDict, total=False):
    name: str
    no: int
    color: str
    icon: str

ActivationConfig = Tuple[Union[str, None], bool]

class Logger:
    @overload
    def add(
        self,
        sink: Union[TextIO, Writable, Callable[[Message], None], Handler],
        *,
        level: Union[str, int] = ...,
        format: Union[str, FormatFunction] = ...,
        filter: Optional[Union[str, FilterFunction, FilterDict]] = ...,
        colorize: Optional[bool] = ...,
        serialize: bool = ...,
        backtrace: bool = ...,
        diagnose: bool = ...,
        enqueue: bool = ...,
        catch: bool = ...
    ) -> int: ...
    @overload
    def add(
        self,
        sink: Callable[[Message], Awaitable[None]],
        *,
        level: Union[str, int] = ...,
        format: Union[str, FormatFunction] = ...,
        filter: Optional[Union[str, FilterFunction, FilterDict]] = ...,
        colorize: Optional[bool] = ...,
        serialize: bool = ...,
        backtrace: bool = ...,
        diagnose: bool = ...,
        enqueue: bool = ...,
        catch: bool = ...,
        loop: Optional[AbstractEventLoop] = ...
    ) -> int: ...
    @overload
    def add(
        self,
        sink: Union[str, PathLikeStr],
        *,
        level: Union[str, int] = ...,
        format: Union[str, FormatFunction] = ...,
        filter: Optional[Union[str, FilterFunction, FilterDict]] = ...,
        colorize: Optional[bool] = ...,
        serialize: bool = ...,
        backtrace: bool = ...,
        diagnose: bool = ...,
        enqueue: bool = ...,
        catch: bool = ...,
        rotation: Optional[Union[str, int, time, timedelta, RotationFunction]] = ...,
        retention: Optional[Union[str, int, timedelta, RetentionFunction]] = ...,
        compression: Optional[Union[str, CompressionFunction]] = ...,
        delay: bool = ...,
        watch: bool = ...,
        mode: str = ...,
        buffering: int = ...,
        encoding: str = ...,
        **kwargs: Any
    ) -> int: ...
    def remove(self, handler_id: Optional[int] = ...) -> None: ...
    def complete(self) -> AwaitableCompleter: ...
    @overload
    def catch(  # type: ignore[misc]
        self,
        exception: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = ...,
        *,
        level: Union[str, int] = ...,
        reraise: bool = ...,
        onerror: Optional[Callable[[BaseException], None]] = ...,
        exclude: Optional[Union[Type[BaseException], Tuple[Type[BaseException], ...]]] = ...,
        default: Any = ...,
        message: str = ...
    ) -> Catcher: ...
    @overload
    def catch(self, exception: _F) -> _F: ...
    def opt(
        self,
        *,
        exception: Optional[Union[bool, ExcInfo, BaseException]] = ...,
        record: bool = ...,
        lazy: bool = ...,
        colors: bool = ...,
        raw: bool = ...,
        capture: bool = ...,
        depth: int = ...,
        ansi: bool = ...
    ) -> Logger: ...
    def bind(__self, **kwargs: Any) -> Logger: ...
    def contextualize(__self, **kwargs: Any) -> Contextualizer: ...
    def patch(self, patcher: PatcherFunction) -> Logger: ...
    @overload
    def level(self, name: str) -> Level: ...
    @overload
    def level(
        self, name: str, no: int = ..., color: Optional[str] = ..., icon: Optional[str] = ...
    ) -> Level: ...
    @overload
    def level(
        self,
        name: str,
        no: Optional[int] = ...,
        color: Optional[str] = ...,
        icon: Optional[str] = ...,
    ) -> Level: ...
    def disable(self, name: Union[str, None]) -> None: ...
    def enable(self, name: Union[str, None]) -> None: ...
    def configure(
        self,
        *,
        handlers: Sequence[Dict[str, Any]] = ...,
        levels: Optional[Sequence[LevelConfig]] = ...,
        extra: Optional[Dict[Any, Any]] = ...,
        patcher: Optional[PatcherFunction] = ...,
        activation: Optional[Sequence[ActivationConfig]] = ...
    ) -> List[int]: ...
    # @staticmethod cannot be used with @overload in mypy (python/mypy#7781).
    # However Logger is not exposed and logger is an instance of Logger
    # so for type checkers it is all the same whether it is defined here
    # as a static method or an instance method.
    @overload
    def parse(
        self,
        file: Union[str, PathLikeStr, TextIO],
        pattern: Union[str, Pattern[str]],
        *,
        cast: Union[Dict[str, Callable[[str], Any]], Callable[[Dict[str, str]], None]] = ...,
        chunk: int = ...
    ) -> Generator[Dict[str, Any], None, None]: ...
    @overload
    def parse(
        self,
        file: BinaryIO,
        pattern: Union[bytes, Pattern[bytes]],
        *,
        cast: Union[Dict[str, Callable[[bytes], Any]], Callable[[Dict[str, bytes]], None]] = ...,
        chunk: int = ...
    ) -> Generator[Dict[str, Any], None, None]: ...
    @overload
    def trace(__self, __message: str, *args: Any, **kwargs: Any) -> None: ...
    @overload
    def trace(__self, __message: Any) -> None: ...
    @overload
    def debug(__self, __message: str, *args: Any, **kwargs: Any) -> None: ...
    @overload
    def debug(__self, __message: Any) -> None: ...
    @overload
    def info(__self, __message: str, *args: Any, **kwargs: Any) -> None: ...
    @overload
    def info(__self, __message: Any) -> None: ...
    @overload
    def success(__self, __message: str, *args: Any, **kwargs: Any) -> None: ...
    @overload
    def success(__self, __message: Any) -> None: ...
    @overload
    def warning(__self, __message: str, *args: Any, **kwargs: Any) -> None: ...
    @overload
    def warning(__self, __message: Any) -> None: ...
    @overload
    def error(__self, __message: str, *args: Any, **kwargs: Any) -> None: ...
    @overload
    def error(__self, __message: Any) -> None: ...
    @overload
    def critical(__self, __message: str, *args: Any, **kwargs: Any) -> None: ...
    @overload
    def critical(__self, __message: Any) -> None: ...
    @overload
    def exception(__self, __message: str, *args: Any, **kwargs: Any) -> None: ...
    @overload
    def exception(__self, __message: Any) -> None: ...
    @overload
    def log(
        __self, __level: Union[int, str], __message: str, *args: Any, **kwargs: Any
    ) -> None: ...
    @overload
    def log(__self, __level: Union[int, str], __message: Any) -> None: ...
    def start(self, *args: Any, **kwargs: Any) -> int: ...
    def stop(self, *args: Any, **kwargs: Any) -> None: ...

logger: Logger
