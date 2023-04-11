"""
.. References and links rendered by Sphinx are kept here as "module documentation" so that they can
   be used in the ``Logger`` docstrings but do not pollute ``help(logger)`` output.

.. |Logger| replace:: :class:`~Logger`
.. |add| replace:: :meth:`~Logger.add()`
.. |remove| replace:: :meth:`~Logger.remove()`
.. |complete| replace:: :meth:`~Logger.complete()`
.. |catch| replace:: :meth:`~Logger.catch()`
.. |bind| replace:: :meth:`~Logger.bind()`
.. |contextualize| replace:: :meth:`~Logger.contextualize()`
.. |patch| replace:: :meth:`~Logger.patch()`
.. |opt| replace:: :meth:`~Logger.opt()`
.. |log| replace:: :meth:`~Logger.log()`
.. |level| replace:: :meth:`~Logger.level()`
.. |enable| replace:: :meth:`~Logger.enable()`
.. |disable| replace:: :meth:`~Logger.disable()`

.. |Any| replace:: :obj:`~typing.Any`
.. |str| replace:: :class:`str`
.. |int| replace:: :class:`int`
.. |bool| replace:: :class:`bool`
.. |tuple| replace:: :class:`tuple`
.. |namedtuple| replace:: :func:`namedtuple<collections.namedtuple>`
.. |list| replace:: :class:`list`
.. |dict| replace:: :class:`dict`
.. |str.format| replace:: :meth:`str.format()`
.. |Path| replace:: :class:`pathlib.Path`
.. |match.groupdict| replace:: :meth:`re.Match.groupdict()`
.. |Handler| replace:: :class:`logging.Handler`
.. |sys.stderr| replace:: :data:`sys.stderr`
.. |sys.exc_info| replace:: :func:`sys.exc_info()`
.. |time| replace:: :class:`datetime.time`
.. |datetime| replace:: :class:`datetime.datetime`
.. |timedelta| replace:: :class:`datetime.timedelta`
.. |open| replace:: :func:`open()`
.. |logging| replace:: :mod:`logging`
.. |signal| replace:: :mod:`signal`
.. |contextvars| replace:: :mod:`contextvars`
.. |Thread.run| replace:: :meth:`Thread.run()<threading.Thread.run()>`
.. |Exception| replace:: :class:`Exception`
.. |AbstractEventLoop| replace:: :class:`AbstractEventLoop<asyncio.AbstractEventLoop>`
.. |asyncio.get_running_loop| replace:: :func:`asyncio.get_running_loop()`
.. |asyncio.run| replace:: :func:`asyncio.run()`
.. |loop.run_until_complete| replace::
    :meth:`loop.run_until_complete()<asyncio.loop.run_until_complete()>`
.. |loop.create_task| replace:: :meth:`loop.create_task()<asyncio.loop.create_task()>`

.. |logger.trace| replace:: :meth:`logger.trace()<Logger.trace()>`
.. |logger.debug| replace:: :meth:`logger.debug()<Logger.debug()>`
.. |logger.info| replace:: :meth:`logger.info()<Logger.info()>`
.. |logger.success| replace:: :meth:`logger.success()<Logger.success()>`
.. |logger.warning| replace:: :meth:`logger.warning()<Logger.warning()>`
.. |logger.error| replace:: :meth:`logger.error()<Logger.error()>`
.. |logger.critical| replace:: :meth:`logger.critical()<Logger.critical()>`

.. |file-like object| replace:: ``file-like object``
.. _file-like object: https://docs.python.org/3/glossary.html#term-file-object
.. |callable| replace:: ``callable``
.. _callable: https://docs.python.org/3/library/functions.html#callable
.. |coroutine function| replace:: ``coroutine function``
.. _coroutine function: https://docs.python.org/3/glossary.html#term-coroutine-function
.. |re.Pattern| replace:: ``re.Pattern``
.. _re.Pattern: https://docs.python.org/3/library/re.html#re-objects

.. |better_exceptions| replace:: ``better_exceptions``
.. _better_exceptions: https://github.com/Qix-/better-exceptions

.. _Pendulum: https://pendulum.eustace.io/docs/#tokens
.. _@sdispater: https://github.com/sdispater
.. _@Qix-: https://github.com/Qix-
.. _formatting directives: https://docs.python.org/3/library/string.html#format-string-syntax
.. _reentrant: https://en.wikipedia.org/wiki/Reentrancy_(computing)
"""
import builtins
import contextlib
import functools
import itertools
import logging
import re
import sys
import warnings
from collections import namedtuple
from inspect import isclass, iscoroutinefunction, isgeneratorfunction
from multiprocessing import current_process
from os.path import basename, splitext
from threading import current_thread

from . import _asyncio_loop, _colorama, _defaults, _filters
from ._better_exceptions import ExceptionFormatter
from ._colorizer import Colorizer
from ._contextvars import ContextVar
from ._datetime import aware_now
from ._error_interceptor import ErrorInterceptor
from ._file_sink import FileSink
from ._get_frame import get_frame
from ._handler import Handler
from ._locks_machinery import create_logger_lock
from ._recattrs import RecordException, RecordFile, RecordLevel, RecordProcess, RecordThread
from ._simple_sinks import AsyncSink, CallableSink, StandardSink, StreamSink

if sys.version_info >= (3, 6):
    from os import PathLike
else:
    from pathlib import PurePath as PathLike


Level = namedtuple("Level", ["name", "no", "color", "icon"])

start_time = aware_now()

context = ContextVar("loguru_context", default={})


class Core:
    def __init__(self):
        levels = [
            Level(
                "TRACE",
                _defaults.LOGURU_TRACE_NO,
                _defaults.LOGURU_TRACE_COLOR,
                _defaults.LOGURU_TRACE_ICON,
            ),
            Level(
                "DEBUG",
                _defaults.LOGURU_DEBUG_NO,
                _defaults.LOGURU_DEBUG_COLOR,
                _defaults.LOGURU_DEBUG_ICON,
            ),
            Level(
                "INFO",
                _defaults.LOGURU_INFO_NO,
                _defaults.LOGURU_INFO_COLOR,
                _defaults.LOGURU_INFO_ICON,
            ),
            Level(
                "SUCCESS",
                _defaults.LOGURU_SUCCESS_NO,
                _defaults.LOGURU_SUCCESS_COLOR,
                _defaults.LOGURU_SUCCESS_ICON,
            ),
            Level(
                "WARNING",
                _defaults.LOGURU_WARNING_NO,
                _defaults.LOGURU_WARNING_COLOR,
                _defaults.LOGURU_WARNING_ICON,
            ),
            Level(
                "ERROR",
                _defaults.LOGURU_ERROR_NO,
                _defaults.LOGURU_ERROR_COLOR,
                _defaults.LOGURU_ERROR_ICON,
            ),
            Level(
                "CRITICAL",
                _defaults.LOGURU_CRITICAL_NO,
                _defaults.LOGURU_CRITICAL_COLOR,
                _defaults.LOGURU_CRITICAL_ICON,
            ),
        ]
        self.levels = {level.name: level for level in levels}
        self.levels_ansi_codes = {
            **{name: Colorizer.ansify(level.color) for name, level in self.levels.items()},
            None: "",
        }

        # Cache used internally to quickly access level attributes based on their name or severity.
        # It can also contain integers as keys, it serves to avoid calling "isinstance()" repeatedly
        # when "logger.log()" is used.
        self.levels_lookup = {
            name: (name, name, level.no, level.icon) for name, level in self.levels.items()
        }

        self.handlers_count = itertools.count()
        self.handlers = {}

        self.extra = {}
        self.patcher = None

        self.min_level = float("inf")
        self.enabled = {}
        self.activation_list = []
        self.activation_none = True

        self.lock = create_logger_lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = create_logger_lock()


class Logger:
    """An object to dispatch logging messages to configured handlers.

    The |Logger| is the core object of ``loguru``, every logging configuration and usage pass
    through a call to one of its methods. There is only one logger, so there is no need to retrieve
    one before usage.

    Once the ``logger`` is imported, it can be used to write messages about events happening in your
    code. By reading the output logs of your application, you gain a better understanding of the
    flow of your program and you more easily track and debug unexpected behaviors.

    Handlers to which the logger sends log messages are added using the |add| method. Note that you
    can use the |Logger| right after import as it comes pre-configured (logs are emitted to
    |sys.stderr| by default). Messages can be logged with different severity levels and they can be
    formatted using curly braces (it uses |str.format| under the hood).

    When a message is logged, a "record" is associated with it. This record is a dict which contains
    information about the logging context: time, function, file, line, thread, level... It also
    contains the ``__name__`` of the module, this is why you don't need named loggers.

    You should not instantiate a |Logger| by yourself, use ``from loguru import logger`` instead.
    """

    def __init__(self, core, exception, depth, record, lazy, colors, raw, capture, patchers, extra):
        self._core = core
        self._options = (exception, depth, record, lazy, colors, raw, capture, patchers, extra)

    def __repr__(self):
        return "<loguru.logger handlers=%r>" % list(self._core.handlers.values())

    def add(
        self,
        sink,
        *,
        level=_defaults.LOGURU_LEVEL,
        format=_defaults.LOGURU_FORMAT,
        filter=_defaults.LOGURU_FILTER,
        colorize=_defaults.LOGURU_COLORIZE,
        serialize=_defaults.LOGURU_SERIALIZE,
        backtrace=_defaults.LOGURU_BACKTRACE,
        diagnose=_defaults.LOGURU_DIAGNOSE,
        enqueue=_defaults.LOGURU_ENQUEUE,
        catch=_defaults.LOGURU_CATCH,
        **kwargs
    ):
        r"""Add a handler sending log messages to a sink adequately configured.

        Parameters
        ----------
        sink : |file-like object|_, |str|, |Path|, |callable|_, |coroutine function|_ or |Handler|
            An object in charge of receiving formatted logging messages and propagating them to an
            appropriate endpoint.
        level : |int| or |str|, optional
            The minimum severity level from which logged messages should be sent to the sink.
        format : |str| or |callable|_, optional
            The template used to format logged messages before being sent to the sink.
        filter : |callable|_, |str| or |dict|, optional
            A directive optionally used to decide for each logged message whether it should be sent
            to the sink or not.
        colorize : |bool|, optional
            Whether the color markups contained in the formatted message should be converted to ansi
            codes for terminal coloration, or stripped otherwise. If ``None``, the choice is
            automatically made based on the sink being a tty or not.
        serialize : |bool|, optional
            Whether the logged message and its records should be first converted to a JSON string
            before being sent to the sink.
        backtrace : |bool|, optional
            Whether the exception trace formatted should be extended upward, beyond the catching
            point, to show the full stacktrace which generated the error.
        diagnose : |bool|, optional
            Whether the exception trace should display the variables values to eases the debugging.
            This should be set to ``False`` in production to avoid leaking sensitive data.
        enqueue : |bool|, optional
            Whether the messages to be logged should first pass through a multiprocessing-safe queue
            before reaching the sink. This is useful while logging to a file through multiple
            processes. This also has the advantage of making logging calls non-blocking.
        catch : |bool|, optional
            Whether errors occurring while sink handles logs messages should be automatically
            caught. If ``True``, an exception message is displayed on |sys.stderr| but the exception
            is not propagated to the caller, preventing your app to crash.
        **kwargs
            Additional parameters that are only valid to configure a coroutine or file sink (see
            below).


        If and only if the sink is a coroutine function, the following parameter applies:

        Parameters
        ----------
        loop : |AbstractEventLoop|, optional
            The event loop in which the asynchronous logging task will be scheduled and executed. If
            ``None``, the loop used is the one returned by |asyncio.get_running_loop| at the time of
            the logging call (task is discarded if there is no loop currently running).


        If and only if the sink is a file path, the following parameters apply:

        Parameters
        ----------
        rotation : |str|, |int|, |time|, |timedelta| or |callable|_, optional
            A condition indicating whenever the current logged file should be closed and a new one
            started.
        retention : |str|, |int|, |timedelta| or |callable|_, optional
            A directive filtering old files that should be removed during rotation or end of
            program.
        compression : |str| or |callable|_, optional
            A compression or archive format to which log files should be converted at closure.
        delay : |bool|, optional
            Whether the file should be created as soon as the sink is configured, or delayed until
            first logged message. It defaults to ``False``.
        watch : |bool|, optional
            Whether or not the file should be watched and re-opened when deleted or changed (based
            on its device and inode properties) by an external program. It defaults to ``False``.
        mode : |str|, optional
            The opening mode as for built-in |open| function. It defaults to ``"a"`` (open the
            file in appending mode).
        buffering : |int|, optional
            The buffering policy as for built-in |open| function. It defaults to ``1`` (line
            buffered file).
        encoding : |str|, optional
            The file encoding as for built-in |open| function. It defaults to ``"utf8"``.
        **kwargs
            Others parameters are passed to the built-in |open| function.

        Returns
        -------
        :class:`int`
            An identifier associated with the added sink and which should be used to
            |remove| it.

        Raises
        ------
        ValueError
            If any of the arguments passed to configure the sink is invalid.

        Notes
        -----
        Extended summary follows.

        .. _sink:

        .. rubric:: The sink parameter

        The ``sink`` handles incoming log messages and proceed to their writing somewhere and
        somehow. A sink can take many forms:

        - A |file-like object|_ like ``sys.stderr`` or ``open("somefile.log", "w")``. Anything with
          a ``.write()`` method is considered as a file-like object. Custom handlers may also
          implement ``flush()`` (called after each logged message), ``stop()`` (called at sink
          termination) and ``complete()`` (awaited by the eponymous method).
        - A file path as |str| or |Path|. It can be parametrized with some additional parameters,
          see below.
        - A |callable|_ (such as a simple function) like ``lambda msg: print(msg)``. This
          allows for logging procedure entirely defined by user preferences and needs.
        - A asynchronous |coroutine function|_ defined with the ``async def`` statement. The
          coroutine object returned by such function will be added to the event loop using
          |loop.create_task|. The tasks should be awaited before ending the loop by using
          |complete|.
        - A built-in |Handler| like ``logging.StreamHandler``. In such a case, the `Loguru` records
          are automatically converted to the structure expected by the |logging| module.

        Note that the logging functions are not `reentrant`_. This means you should avoid using
        the ``logger`` inside any of your sinks or from within |signal| handlers. Otherwise, you
        may face deadlock if the module's sink was not explicitly disabled.

        .. _message:

        .. rubric:: The logged message

        The logged message passed to all added sinks is nothing more than a string of the
        formatted log, to which a special attribute is associated: the ``.record`` which is a dict
        containing all contextual information possibly needed (see below).

        Logged messages are formatted according to the ``format`` of the added sink. This format
        is usually a string containing braces fields to display attributes from the record dict.

        If fine-grained control is needed, the ``format`` can also be a function which takes the
        record as parameter and return the format template string. However, note that in such a
        case, you should take care of appending the line ending and exception field to the returned
        format, while ``"\n{exception}"`` is automatically appended for convenience if ``format`` is
        a string.

        The ``filter`` attribute can be used to control which messages are effectively passed to the
        sink and which one are ignored. A function can be used, accepting the record as an
        argument, and returning ``True`` if the message should be logged, ``False`` otherwise. If
        a string is used, only the records with the same ``name`` and its children will be allowed.
        One can also pass a ``dict`` mapping module names to minimum required level. In such case,
        each log record will search for it's closest parent in the ``dict`` and use the associated
        level as the filter. The ``dict`` values can be ``int`` severity, ``str`` level name or
        ``True`` and ``False`` to respectively authorize and discard all module logs
        unconditionally. In order to set a default level, the ``""`` module name should be used as
        it is the parent of all modules (it does not suppress global ``level`` threshold, though).

        Note that while calling a logging method, the keyword arguments (if any) are automatically
        added to the ``extra`` dict for convenient contextualization (in addition to being used for
        formatting).

        .. _levels:

        .. rubric:: The severity levels

        Each logged message is associated with a severity level. These levels make it possible to
        prioritize messages and to choose the verbosity of the logs according to usages. For
        example, it allows to display some debugging information to a developer, while hiding it to
        the end user running the application.

        The ``level`` attribute of every added sink controls the minimum threshold from which log
        messages are allowed to be emitted. While using the ``logger``, you are in charge of
        configuring the appropriate granularity of your logs. It is possible to add even more custom
        levels by using the |level| method.

        Here are the standard levels with their default severity value, each one is associated with
        a logging method of the same name:

        +----------------------+------------------------+------------------------+
        | Level name           | Severity value         | Logger method          |
        +======================+========================+========================+
        | ``TRACE``            | 5                      | |logger.trace|         |
        +----------------------+------------------------+------------------------+
        | ``DEBUG``            | 10                     | |logger.debug|         |
        +----------------------+------------------------+------------------------+
        | ``INFO``             | 20                     | |logger.info|          |
        +----------------------+------------------------+------------------------+
        | ``SUCCESS``          | 25                     | |logger.success|       |
        +----------------------+------------------------+------------------------+
        | ``WARNING``          | 30                     | |logger.warning|       |
        +----------------------+------------------------+------------------------+
        | ``ERROR``            | 40                     | |logger.error|         |
        +----------------------+------------------------+------------------------+
        | ``CRITICAL``         | 50                     | |logger.critical|      |
        +----------------------+------------------------+------------------------+

        .. _record:

        .. rubric:: The record dict

        The record is just a Python dict, accessible from sinks by ``message.record``. It contains
        all contextual information of the logging call (time, function, file, line, level, etc.).

        Each of the record keys can be used in the handler's ``format`` so the corresponding value
        is properly displayed in the logged message (e.g. ``"{level}"`` will return ``"INFO"``).
        Some records' values are objects with two or more attributes. These can be formatted with
        ``"{key.attr}"`` (``"{key}"`` would display one by default).

        Note that you can use any `formatting directives`_ available in Python's ``str.format()``
        method (e.g. ``"{key: >3}"`` will right-align and pad to a width of 3 characters). This is
        particularly useful for time formatting (see below).

        +------------+---------------------------------+----------------------------+
        | Key        | Description                     | Attributes                 |
        +============+=================================+============================+
        | elapsed    | The time elapsed since the      | See |timedelta|            |
        |            | start of the program            |                            |
        +------------+---------------------------------+----------------------------+
        | exception  | The formatted exception if any, | ``type``, ``value``,       |
        |            | ``None`` otherwise              | ``traceback``              |
        +------------+---------------------------------+----------------------------+
        | extra      | The dict of attributes          | None                       |
        |            | bound by the user (see |bind|)  |                            |
        +------------+---------------------------------+----------------------------+
        | file       | The file where the logging call | ``name`` (default),        |
        |            | was made                        | ``path``                   |
        +------------+---------------------------------+----------------------------+
        | function   | The function from which the     | None                       |
        |            | logging call was made           |                            |
        +------------+---------------------------------+----------------------------+
        | level      | The severity used to log the    | ``name`` (default),        |
        |            | message                         | ``no``, ``icon``           |
        +------------+---------------------------------+----------------------------+
        | line       | The line number in the source   | None                       |
        |            | code                            |                            |
        +------------+---------------------------------+----------------------------+
        | message    | The logged message (not yet     | None                       |
        |            | formatted)                      |                            |
        +------------+---------------------------------+----------------------------+
        | module     | The module where the logging    | None                       |
        |            | call was made                   |                            |
        +------------+---------------------------------+----------------------------+
        | name       | The ``__name__`` where the      | None                       |
        |            | logging call was made           |                            |
        +------------+---------------------------------+----------------------------+
        | process    | The process in which the        | ``name``, ``id`` (default) |
        |            | logging call was made           |                            |
        +------------+---------------------------------+----------------------------+
        | thread     | The thread in which the         | ``name``, ``id`` (default) |
        |            | logging call was made           |                            |
        +------------+---------------------------------+----------------------------+
        | time       | The aware local time when the   | See |datetime|             |
        |            | logging call was made           |                            |
        +------------+---------------------------------+----------------------------+

        .. _time:

        .. rubric:: The time formatting

        To use your favorite time representation, you can set it directly in the time formatter
        specifier of your handler format, like for example ``format="{time:HH:mm:ss} {message}"``.
        Note that this datetime represents your local time, and it is also made timezone-aware,
        so you can display the UTC offset to avoid ambiguities.

        The time field can be formatted using more human-friendly tokens. These constitute a subset
        of the one used by the `Pendulum`_ library of `@sdispater`_. To escape a token, just add
        square brackets around it, for example ``"[YY]"`` would display literally ``"YY"``.

        If you prefer to display UTC rather than local time, you can add ``"!UTC"`` at the very end
        of the time format, like ``{time:HH:mm:ss!UTC}``. Doing so will convert the ``datetime``
        to UTC before formatting.

        If no time formatter specifier is used, like for example if ``format="{time} {message}"``,
        the default one will use ISO 8601.

        +------------------------+---------+----------------------------------------+
        |                        | Token   | Output                                 |
        +========================+=========+========================================+
        | Year                   | YYYY    | 2000, 2001, 2002 ... 2012, 2013        |
        |                        +---------+----------------------------------------+
        |                        | YY      | 00, 01, 02 ... 12, 13                  |
        +------------------------+---------+----------------------------------------+
        | Quarter                | Q       | 1 2 3 4                                |
        +------------------------+---------+----------------------------------------+
        | Month                  | MMMM    | January, February, March ...           |
        |                        +---------+----------------------------------------+
        |                        | MMM     | Jan, Feb, Mar ...                      |
        |                        +---------+----------------------------------------+
        |                        | MM      | 01, 02, 03 ... 11, 12                  |
        |                        +---------+----------------------------------------+
        |                        | M       | 1, 2, 3 ... 11, 12                     |
        +------------------------+---------+----------------------------------------+
        | Day of Year            | DDDD    | 001, 002, 003 ... 364, 365             |
        |                        +---------+----------------------------------------+
        |                        | DDD     | 1, 2, 3 ... 364, 365                   |
        +------------------------+---------+----------------------------------------+
        | Day of Month           | DD      | 01, 02, 03 ... 30, 31                  |
        |                        +---------+----------------------------------------+
        |                        | D       | 1, 2, 3 ... 30, 31                     |
        +------------------------+---------+----------------------------------------+
        | Day of Week            | dddd    | Monday, Tuesday, Wednesday ...         |
        |                        +---------+----------------------------------------+
        |                        | ddd     | Mon, Tue, Wed ...                      |
        |                        +---------+----------------------------------------+
        |                        | d       | 0, 1, 2 ... 6                          |
        +------------------------+---------+----------------------------------------+
        | Days of ISO Week       | E       | 1, 2, 3 ... 7                          |
        +------------------------+---------+----------------------------------------+
        | Hour                   | HH      | 00, 01, 02 ... 23, 24                  |
        |                        +---------+----------------------------------------+
        |                        | H       | 0, 1, 2 ... 23, 24                     |
        |                        +---------+----------------------------------------+
        |                        | hh      | 01, 02, 03 ... 11, 12                  |
        |                        +---------+----------------------------------------+
        |                        | h       | 1, 2, 3 ... 11, 12                     |
        +------------------------+---------+----------------------------------------+
        | Minute                 | mm      | 00, 01, 02 ... 58, 59                  |
        |                        +---------+----------------------------------------+
        |                        | m       | 0, 1, 2 ... 58, 59                     |
        +------------------------+---------+----------------------------------------+
        | Second                 | ss      | 00, 01, 02 ... 58, 59                  |
        |                        +---------+----------------------------------------+
        |                        | s       | 0, 1, 2 ... 58, 59                     |
        +------------------------+---------+----------------------------------------+
        | Fractional Second      | S       | 0 1 ... 8 9                            |
        |                        +---------+----------------------------------------+
        |                        | SS      | 00, 01, 02 ... 98, 99                  |
        |                        +---------+----------------------------------------+
        |                        | SSS     | 000 001 ... 998 999                    |
        |                        +---------+----------------------------------------+
        |                        | SSSS... | 000[0..] 001[0..] ... 998[0..] 999[0..]|
        |                        +---------+----------------------------------------+
        |                        | SSSSSS  | 000000 000001 ... 999998 999999        |
        +------------------------+---------+----------------------------------------+
        | AM / PM                | A       | AM, PM                                 |
        +------------------------+---------+----------------------------------------+
        | Timezone               | Z       | -07:00, -06:00 ... +06:00, +07:00      |
        |                        +---------+----------------------------------------+
        |                        | ZZ      | -0700, -0600 ... +0600, +0700          |
        |                        +---------+----------------------------------------+
        |                        | zz      | EST CST ... MST PST                    |
        +------------------------+---------+----------------------------------------+
        | Seconds timestamp      | X       | 1381685817, 1234567890.123             |
        +------------------------+---------+----------------------------------------+
        | Microseconds timestamp | x       | 1234567890123                          |
        +------------------------+---------+----------------------------------------+

        .. _file:

        .. rubric:: The file sinks

        If the sink is a |str| or a |Path|, the corresponding file will be opened for writing logs.
        The path can also contain a special ``"{time}"`` field that will be formatted with the
        current date at file creation.

        The ``rotation`` check is made before logging each message. If there is already an existing
        file with the same name that the file to be created, then the existing file is renamed by
        appending the date to its basename to prevent file overwriting. This parameter accepts:

        - an |int| which corresponds to the maximum file size in bytes before that the current
          logged file is closed and a new one started over.
        - a |timedelta| which indicates the frequency of each new rotation.
        - a |time| which specifies the hour when the daily rotation should occur.
        - a |str| for human-friendly parametrization of one of the previously enumerated types.
          Examples: ``"100 MB"``, ``"0.5 GB"``, ``"1 month 2 weeks"``, ``"4 days"``, ``"10h"``,
          ``"monthly"``, ``"18:00"``, ``"sunday"``, ``"w0"``, ``"monday at 12:00"``, ...
        - a |callable|_ which will be invoked before logging. It should accept two arguments: the
          logged message and the file object, and it should return ``True`` if the rotation should
          happen now, ``False`` otherwise.

        The ``retention`` occurs at rotation or at sink stop if rotation is ``None``. Files are
        selected if they match the pattern ``"basename(.*).ext(.*)"`` (possible time fields are
        beforehand replaced with ``.*``) based on the sink file. This parameter accepts:

        - an |int| which indicates the number of log files to keep, while older files are removed.
        - a |timedelta| which specifies the maximum age of files to keep.
        - a |str| for human-friendly parametrization of the maximum age of files to keep.
          Examples: ``"1 week, 3 days"``, ``"2 months"``, ...
        - a |callable|_ which will be invoked before the retention process. It should accept the
          list of log files as argument and process to whatever it wants (moving files, removing
          them, etc.).

        The ``compression`` happens at rotation or at sink stop if rotation is ``None``. This
        parameter accepts:

        - a |str| which corresponds to the compressed or archived file extension. This can be one
          of: ``"gz"``, ``"bz2"``, ``"xz"``, ``"lzma"``, ``"tar"``, ``"tar.gz"``, ``"tar.bz2"``,
          ``"tar.xz"``, ``"zip"``.
        - a |callable|_ which will be invoked before file termination. It should accept the path of
          the log file as argument and process to whatever it wants (custom compression, network
          sending, removing it, etc.).

        Either way, if you use a custom function designed according to your preferences, you must be
        very careful not to use the ``logger`` within your function. Otherwise, there is a risk that
        your program hang because of a deadlock.

        .. _color:

        .. rubric:: The color markups

        To add colors to your logs, you just have to enclose your format string with the appropriate
        tags (e.g. ``<red>some message</red>``). These tags are automatically removed if the sink
        doesn't support ansi codes. For convenience, you can use ``</>`` to close the last opening
        tag without repeating its name (e.g. ``<red>another message</>``).

        The special tag ``<level>`` (abbreviated with ``<lvl>``) is transformed according to
        the configured color of the logged message level.

        Tags which are not recognized will raise an exception during parsing, to inform you about
        possible misuse. If you wish to display a markup tag literally, you can escape it by
        prepending a ``\`` like for example ``\<blue>``. If, for some reason, you need to escape a
        string programmatically, note that the regex used internally to parse markup tags is
        ``r"\\?</?((?:[fb]g\s)?[^<>\s]*)>"``.

        Note that when logging a message with ``opt(colors=True)``, color tags present in the
        formatting arguments (``args`` and ``kwargs``) are completely ignored. This is important if
        you need to log strings containing markups that might interfere with the color tags (in this
        case, do not use f-string).

        Here are the available tags (note that compatibility may vary depending on terminal):

        +------------------------------------+--------------------------------------+
        | Color (abbr)                       | Styles (abbr)                        |
        +====================================+======================================+
        | Black (k)                          | Bold (b)                             |
        +------------------------------------+--------------------------------------+
        | Blue (e)                           | Dim (d)                              |
        +------------------------------------+--------------------------------------+
        | Cyan (c)                           | Normal (n)                           |
        +------------------------------------+--------------------------------------+
        | Green (g)                          | Italic (i)                           |
        +------------------------------------+--------------------------------------+
        | Magenta (m)                        | Underline (u)                        |
        +------------------------------------+--------------------------------------+
        | Red (r)                            | Strike (s)                           |
        +------------------------------------+--------------------------------------+
        | White (w)                          | Reverse (v)                          |
        +------------------------------------+--------------------------------------+
        | Yellow (y)                         | Blink (l)                            |
        +------------------------------------+--------------------------------------+
        |                                    | Hide (h)                             |
        +------------------------------------+--------------------------------------+

        Usage:

        +-----------------+-------------------------------------------------------------------+
        | Description     | Examples                                                          |
        |                 +---------------------------------+---------------------------------+
        |                 | Foreground                      | Background                      |
        +=================+=================================+=================================+
        | Basic colors    | ``<red>``, ``<r>``              | ``<GREEN>``, ``<G>``            |
        +-----------------+---------------------------------+---------------------------------+
        | Light colors    | ``<light-blue>``, ``<le>``      | ``<LIGHT-CYAN>``, ``<LC>``      |
        +-----------------+---------------------------------+---------------------------------+
        | 8-bit colors    | ``<fg 86>``, ``<fg 255>``       | ``<bg 42>``, ``<bg 9>``         |
        +-----------------+---------------------------------+---------------------------------+
        | Hex colors      | ``<fg #00005f>``, ``<fg #EE1>`` | ``<bg #AF5FD7>``, ``<bg #fff>`` |
        +-----------------+---------------------------------+---------------------------------+
        | RGB colors      | ``<fg 0,95,0>``                 | ``<bg 72,119,65>``              |
        +-----------------+---------------------------------+---------------------------------+
        | Stylizing       | ``<bold>``, ``<b>``,  ``<underline>``, ``<u>``                    |
        +-----------------+-------------------------------------------------------------------+

        .. _env:

        .. rubric:: The environment variables

        The default values of sink parameters can be entirely customized. This is particularly
        useful if you don't like the log format of the pre-configured sink.

        Each of the |add| default parameter can be modified by setting the ``LOGURU_[PARAM]``
        environment variable. For example on Linux: ``export LOGURU_FORMAT="{time} - {message}"``
        or ``export LOGURU_DIAGNOSE=NO``.

        The default levels' attributes can also be modified by setting the ``LOGURU_[LEVEL]_[ATTR]``
        environment variable. For example, on Windows: ``setx LOGURU_DEBUG_COLOR "<blue>"``
        or ``setx LOGURU_TRACE_ICON "🚀"``. If you use the ``set`` command, do not include quotes
        but escape special symbol as needed, e.g. ``set LOGURU_DEBUG_COLOR=^<blue^>``.

        If you want to disable the pre-configured sink, you can set the ``LOGURU_AUTOINIT``
        variable to ``False``.

        On Linux, you will probably need to edit the ``~/.profile`` file to make this persistent. On
        Windows, don't forget to restart your terminal for the change to be taken into account.

        Examples
        --------
        >>> logger.add(sys.stdout, format="{time} - {level} - {message}", filter="sub.module")

        >>> logger.add("file_{time}.log", level="TRACE", rotation="100 MB")

        >>> def debug_only(record):
        ...     return record["level"].name == "DEBUG"
        ...
        >>> logger.add("debug.log", filter=debug_only)  # Other levels are filtered out

        >>> def my_sink(message):
        ...     record = message.record
        ...     update_db(message, time=record["time"], level=record["level"])
        ...
        >>> logger.add(my_sink)

        >>> level_per_module = {
        ...     "": "DEBUG",
        ...     "third.lib": "WARNING",
        ...     "anotherlib": False
        ... }
        >>> logger.add(lambda m: print(m, end=""), filter=level_per_module, level=0)

        >>> async def publish(message):
        ...     await api.post(message)
        ...
        >>> logger.add(publish, serialize=True)

        >>> from logging import StreamHandler
        >>> logger.add(StreamHandler(sys.stderr), format="{message}")

        >>> class RandomStream:
        ...     def __init__(self, seed, threshold):
        ...         self.threshold = threshold
        ...         random.seed(seed)
        ...     def write(self, message):
        ...         if random.random() > self.threshold:
        ...             print(message)
        ...
        >>> stream_object = RandomStream(seed=12345, threshold=0.25)
        >>> logger.add(stream_object, level="INFO")
        """
        with self._core.lock:
            handler_id = next(self._core.handlers_count)

        error_interceptor = ErrorInterceptor(catch, handler_id)

        if colorize is None and serialize:
            colorize = False

        if isinstance(sink, (str, PathLike)):
            path = sink
            name = "'%s'" % path

            if colorize is None:
                colorize = False

            wrapped_sink = FileSink(path, **kwargs)
            kwargs = {}
            encoding = wrapped_sink.encoding
            terminator = "\n"
            exception_prefix = ""
        elif hasattr(sink, "write") and callable(sink.write):
            name = getattr(sink, "name", None) or repr(sink)

            if colorize is None:
                colorize = _colorama.should_colorize(sink)

            if colorize is True and _colorama.should_wrap(sink):
                stream = _colorama.wrap(sink)
            else:
                stream = sink

            wrapped_sink = StreamSink(stream)
            encoding = getattr(sink, "encoding", None)
            terminator = "\n"
            exception_prefix = ""
        elif isinstance(sink, logging.Handler):
            name = repr(sink)

            if colorize is None:
                colorize = False

            wrapped_sink = StandardSink(sink)
            encoding = getattr(sink, "encoding", None)
            terminator = ""
            exception_prefix = "\n"
        elif iscoroutinefunction(sink) or iscoroutinefunction(
            getattr(sink, "__call__", None)  # noqa: B004
        ):
            name = getattr(sink, "__name__", None) or repr(sink)

            if colorize is None:
                colorize = False

            loop = kwargs.pop("loop", None)

            # The worker thread needs an event loop, it can't create a new one internally because it
            # has to be accessible by the user while calling "complete()", instead we use the global
            # one when the sink is added. If "enqueue=False" the event loop is dynamically retrieved
            # at each logging call, which is much more convenient. However, coroutine can't access
            # running loop in Python 3.5.2 and earlier versions, see python/asyncio#452.
            if enqueue and loop is None:
                try:
                    loop = _asyncio_loop.get_running_loop()
                except RuntimeError as e:
                    raise ValueError(
                        "An event loop is required to add a coroutine sink with `enqueue=True`, "
                        "but none has been passed as argument and none is currently running."
                    ) from e

            coro = sink if iscoroutinefunction(sink) else sink.__call__
            wrapped_sink = AsyncSink(coro, loop, error_interceptor)
            encoding = "utf8"
            terminator = "\n"
            exception_prefix = ""
        elif callable(sink):
            name = getattr(sink, "__name__", None) or repr(sink)

            if colorize is None:
                colorize = False

            wrapped_sink = CallableSink(sink)
            encoding = "utf8"
            terminator = "\n"
            exception_prefix = ""
        else:
            raise TypeError("Cannot log to objects of type '%s'" % type(sink).__name__)

        if kwargs:
            raise TypeError("add() got an unexpected keyword argument '%s'" % next(iter(kwargs)))

        if filter is None:
            filter_func = None
        elif filter == "":
            filter_func = _filters.filter_none
        elif isinstance(filter, str):
            parent = filter + "."
            length = len(parent)
            filter_func = functools.partial(_filters.filter_by_name, parent=parent, length=length)
        elif isinstance(filter, dict):
            level_per_module = {}
            for module, level_ in filter.items():
                if module is not None and not isinstance(module, str):
                    raise TypeError(
                        "The filter dict contains an invalid module, "
                        "it should be a string (or None), not: '%s'" % type(module).__name__
                    )
                if level_ is False:
                    levelno_ = False
                elif level_ is True:
                    levelno_ = 0
                elif isinstance(level_, str):
                    try:
                        levelno_ = self.level(level_).no
                    except ValueError:
                        raise ValueError(
                            "The filter dict contains a module '%s' associated to a level name "
                            "which does not exist: '%s'" % (module, level_)
                        )
                elif isinstance(level_, int):
                    levelno_ = level_
                else:
                    raise TypeError(
                        "The filter dict contains a module '%s' associated to an invalid level, "
                        "it should be an integer, a string or a boolean, not: '%s'"
                        % (module, type(level_).__name__)
                    )
                if levelno_ < 0:
                    raise ValueError(
                        "The filter dict contains a module '%s' associated to an invalid level, "
                        "it should be a positive integer, not: '%d'" % (module, levelno_)
                    )
                level_per_module[module] = levelno_
            filter_func = functools.partial(
                _filters.filter_by_level, level_per_module=level_per_module
            )
        elif callable(filter):
            if filter == builtins.filter:
                raise ValueError(
                    "The built-in 'filter()' function cannot be used as a 'filter' parameter, "
                    "this is most likely a mistake (please double-check the arguments passed "
                    "to 'logger.add()')."
                )
            filter_func = filter
        else:
            raise TypeError(
                "Invalid filter, it should be a function, a string or a dict, not: '%s'"
                % type(filter).__name__
            )

        if isinstance(level, str):
            levelno = self.level(level).no
        elif isinstance(level, int):
            levelno = level
        else:
            raise TypeError(
                "Invalid level, it should be an integer or a string, not: '%s'"
                % type(level).__name__
            )

        if levelno < 0:
            raise ValueError(
                "Invalid level value, it should be a positive integer, not: %d" % levelno
            )

        if isinstance(format, str):
            try:
                formatter = Colorizer.prepare_format(format + terminator + "{exception}")
            except ValueError as e:
                raise ValueError(
                    "Invalid format, color markups could not be parsed correctly"
                ) from e
            is_formatter_dynamic = False
        elif callable(format):
            if format == builtins.format:
                raise ValueError(
                    "The built-in 'format()' function cannot be used as a 'format' parameter, "
                    "this is most likely a mistake (please double-check the arguments passed "
                    "to 'logger.add()')."
                )
            formatter = format
            is_formatter_dynamic = True
        else:
            raise TypeError(
                "Invalid format, it should be a string or a function, not: '%s'"
                % type(format).__name__
            )

        if not isinstance(encoding, str):
            encoding = "ascii"

        with self._core.lock:
            exception_formatter = ExceptionFormatter(
                colorize=colorize,
                encoding=encoding,
                diagnose=diagnose,
                backtrace=backtrace,
                hidden_frames_filename=self.catch.__code__.co_filename,
                prefix=exception_prefix,
            )

            handler = Handler(
                name=name,
                sink=wrapped_sink,
                levelno=levelno,
                formatter=formatter,
                is_formatter_dynamic=is_formatter_dynamic,
                filter_=filter_func,
                colorize=colorize,
                serialize=serialize,
                enqueue=enqueue,
                id_=handler_id,
                error_interceptor=error_interceptor,
                exception_formatter=exception_formatter,
                levels_ansi_codes=self._core.levels_ansi_codes,
            )

            handlers = self._core.handlers.copy()
            handlers[handler_id] = handler

            self._core.min_level = min(self._core.min_level, levelno)
            self._core.handlers = handlers

        return handler_id

    def remove(self, handler_id=None):
        """Remove a previously added handler and stop sending logs to its sink.

        Parameters
        ----------
        handler_id : |int| or ``None``
            The id of the sink to remove, as it was returned by the |add| method. If ``None``, all
            handlers are removed. The pre-configured handler is guaranteed to have the index ``0``.

        Raises
        ------
        ValueError
            If ``handler_id`` is not ``None`` but there is no active handler with such id.

        Examples
        --------
        >>> i = logger.add(sys.stderr, format="{message}")
        >>> logger.info("Logging")
        Logging
        >>> logger.remove(i)
        >>> logger.info("No longer logging")
        """
        if not (handler_id is None or isinstance(handler_id, int)):
            raise TypeError(
                "Invalid handler id, it should be an integer as returned "
                "by the 'add()' method (or None), not: '%s'" % type(handler_id).__name__
            )

        with self._core.lock:
            handlers = self._core.handlers.copy()

            if handler_id is not None and handler_id not in handlers:
                raise ValueError("There is no existing handler with id %d" % handler_id) from None

            if handler_id is None:
                handler_ids = list(handlers.keys())
            else:
                handler_ids = [handler_id]

            for handler_id in handler_ids:
                handler = handlers.pop(handler_id)

                # This needs to be done first in case "stop()" raises an exception
                levelnos = (h.levelno for h in handlers.values())
                self._core.min_level = min(levelnos, default=float("inf"))
                self._core.handlers = handlers

                handler.stop()

    def complete(self):
        """Wait for the end of enqueued messages and asynchronous tasks scheduled by handlers.

        This method proceeds in two steps: first it waits for all logging messages added to handlers
        with ``enqueue=True`` to be processed, then it returns an object that can be awaited to
        finalize all logging tasks added to the event loop by coroutine sinks.

        It can be called from non-asynchronous code. This is especially recommended when the
        ``logger`` is utilized with ``multiprocessing`` to ensure messages put to the internal
        queue have been properly transmitted before leaving a child process.

        The returned object should be awaited before the end of a coroutine executed by
        |asyncio.run| or |loop.run_until_complete| to ensure all asynchronous logging messages are
        processed. The function |asyncio.get_running_loop| is called beforehand, only tasks
        scheduled in the same loop that the current one will be awaited by the method.

        Returns
        -------
        :term:`awaitable`
            An awaitable object which ensures all asynchronous logging calls are completed when
            awaited.

        Examples
        --------
        >>> async def sink(message):
        ...     await asyncio.sleep(0.1)  # IO processing...
        ...     print(message, end="")
        ...
        >>> async def work():
        ...     logger.info("Start")
        ...     logger.info("End")
        ...     await logger.complete()
        ...
        >>> logger.add(sink)
        1
        >>> asyncio.run(work())
        Start
        End

        >>> def process():
        ...     logger.info("Message sent from the child")
        ...     logger.complete()
        ...
        >>> logger.add(sys.stderr, enqueue=True)
        1
        >>> process = multiprocessing.Process(target=process)
        >>> process.start()
        >>> process.join()
        Message sent from the child
        """

        with self._core.lock:
            handlers = self._core.handlers.copy()
            for handler in handlers.values():
                handler.complete_queue()

        logger = self

        class AwaitableCompleter:
            def __await__(self):
                with logger._core.lock:
                    handlers = logger._core.handlers.copy()
                    for handler in handlers.values():
                        yield from handler.complete_async().__await__()

        return AwaitableCompleter()

    def catch(
        self,
        exception=Exception,
        *,
        level="ERROR",
        reraise=False,
        onerror=None,
        exclude=None,
        default=None,
        message="An error has been caught in function '{record[function]}', "
        "process '{record[process].name}' ({record[process].id}), "
        "thread '{record[thread].name}' ({record[thread].id}):"
    ):
        """Return a decorator to automatically log possibly caught error in wrapped function.

        This is useful to ensure unexpected exceptions are logged, the entire program can be
        wrapped by this method. This is also very useful to decorate |Thread.run| methods while
        using threads to propagate errors to the main logger thread.

        Note that the visibility of variables values (which uses the great |better_exceptions|_
        library from `@Qix-`_) depends on the ``diagnose`` option of each configured sink.

        The returned object can also be used as a context manager.

        Parameters
        ----------
        exception : |Exception|, optional
            The type of exception to intercept. If several types should be caught, a tuple of
            exceptions can be used too.
        level : |str| or |int|, optional
            The level name or severity with which the message should be logged.
        reraise : |bool|, optional
            Whether the exception should be raised again and hence propagated to the caller.
        onerror : |callable|_, optional
            A function that will be called if an error occurs, once the message has been logged.
            It should accept the exception instance as it sole argument.
        exclude : |Exception|, optional
            A type of exception (or a tuple of types) that will be purposely ignored and hence
            propagated to the caller without being logged.
        default : |Any|, optional
            The value to be returned by the decorated function if an error occurred without being
            re-raised.
        message : |str|, optional
            The message that will be automatically logged if an exception occurs. Note that it will
            be formatted with the ``record`` attribute.

        Returns
        -------
        :term:`decorator` / :term:`context manager`
            An object that can be used to decorate a function or as a context manager to log
            exceptions possibly caught.

        Examples
        --------
        >>> @logger.catch
        ... def f(x):
        ...     100 / x
        ...
        >>> def g():
        ...     f(10)
        ...     f(0)
        ...
        >>> g()
        ERROR - An error has been caught in function 'g', process 'Main' (367), thread 'ch1' (1398):
        Traceback (most recent call last):
          File "program.py", line 12, in <module>
            g()
            └ <function g at 0x7f225fe2bc80>
        > File "program.py", line 10, in g
            f(0)
            └ <function f at 0x7f225fe2b9d8>
          File "program.py", line 6, in f
            100 / x
                  └ 0
        ZeroDivisionError: division by zero

        >>> with logger.catch(message="Because we never know..."):
        ...    main()  # No exception, no logs

        >>> # Use 'onerror' to prevent the program exit code to be 0 (if 'reraise=False') while
        >>> # also avoiding the stacktrace to be duplicated on stderr (if 'reraise=True').
        >>> @logger.catch(onerror=lambda _: sys.exit(1))
        ... def main():
        ...     1 / 0
        """
        if callable(exception) and (
            not isclass(exception) or not issubclass(exception, BaseException)
        ):
            return self.catch()(exception)

        logger = self

        class Catcher:
            def __init__(self, from_decorator):
                self._from_decorator = from_decorator

            def __enter__(self):
                return None

            def __exit__(self, type_, value, traceback_):
                if type_ is None:
                    return

                if not issubclass(type_, exception):
                    return False

                if exclude is not None and issubclass(type_, exclude):
                    return False

                from_decorator = self._from_decorator
                _, depth, _, *options = logger._options

                if from_decorator:
                    depth += 1

                catch_options = [(type_, value, traceback_), depth, True] + options
                logger._log(level, from_decorator, catch_options, message, (), {})

                if onerror is not None:
                    onerror(value)

                return not reraise

            def __call__(self, function):
                if isclass(function):
                    raise TypeError(
                        "Invalid object decorated with 'catch()', it must be a function, "
                        "not a class (tried to wrap '%s')" % function.__name__
                    )

                catcher = Catcher(True)

                if iscoroutinefunction(function):

                    async def catch_wrapper(*args, **kwargs):
                        with catcher:
                            return await function(*args, **kwargs)
                        return default

                elif isgeneratorfunction(function):

                    def catch_wrapper(*args, **kwargs):
                        with catcher:
                            return (yield from function(*args, **kwargs))
                        return default

                else:

                    def catch_wrapper(*args, **kwargs):
                        with catcher:
                            return function(*args, **kwargs)
                        return default

                functools.update_wrapper(catch_wrapper, function)
                return catch_wrapper

        return Catcher(False)

    def opt(
        self,
        *,
        exception=None,
        record=False,
        lazy=False,
        colors=False,
        raw=False,
        capture=True,
        depth=0,
        ansi=False
    ):
        r"""Parametrize a logging call to slightly change generated log message.

        Note that it's not possible to chain |opt| calls, the last one takes precedence over the
        others as it will "reset" the options to their default values.

        Parameters
        ----------
        exception : |bool|, |tuple| or |Exception|, optional
            If it does not evaluate as ``False``, the passed exception is formatted and added to the
            log message. It could be an |Exception| object or a ``(type, value, traceback)`` tuple,
            otherwise the exception information is retrieved from |sys.exc_info|.
        record : |bool|, optional
            If ``True``, the record dict contextualizing the logging call can be used to format the
            message by using ``{record[key]}`` in the log message.
        lazy : |bool|, optional
            If ``True``, the logging call attribute to format the message should be functions which
            will be called only if the level is high enough. This can be used to avoid expensive
            functions if not necessary.
        colors : |bool|, optional
            If ``True``, logged message will be colorized according to the markups it possibly
            contains.
        raw : |bool|, optional
            If ``True``, the formatting of each sink will be bypassed and the message will be sent
            as is.
        capture : |bool|, optional
            If ``False``, the ``**kwargs`` of logged message will not automatically populate
            the ``extra`` dict (although they are still used for formatting).
        depth : |int|, optional
            Specify which stacktrace should be used to contextualize the logged message. This is
            useful while using the logger from inside a wrapped function to retrieve worthwhile
            information.
        ansi : |bool|, optional
            Deprecated since version 0.4.1: the ``ansi`` parameter will be removed in Loguru 1.0.0,
            it is replaced by ``colors`` which is a more appropriate name.

        Returns
        -------
        :class:`~Logger`
            A logger wrapping the core logger, but transforming logged message adequately before
            sending.

        Examples
        --------
        >>> try:
        ...     1 / 0
        ... except ZeroDivisionError:
        ...    logger.opt(exception=True).debug("Exception logged with debug level:")
        ...
        [18:10:02] DEBUG in '<module>' - Exception logged with debug level:
        Traceback (most recent call last, catch point marked):
        > File "<stdin>", line 2, in <module>
        ZeroDivisionError: division by zero

        >>> logger.opt(record=True).info("Current line is: {record[line]}")
        [18:10:33] INFO in '<module>' - Current line is: 1

        >>> logger.opt(lazy=True).debug("If sink <= DEBUG: {x}", x=lambda: math.factorial(2**5))
        [18:11:19] DEBUG in '<module>' - If sink <= DEBUG: 263130836933693530167218012160000000

        >>> logger.opt(colors=True).warning("We got a <red>BIG</red> problem")
        [18:11:30] WARNING in '<module>' - We got a BIG problem

        >>> logger.opt(raw=True).debug("No formatting\n")
        No formatting

        >>> logger.opt(capture=False).info("Displayed but not captured: {value}", value=123)
        [18:11:41] Displayed but not captured: 123

        >>> def wrapped():
        ...     logger.opt(depth=1).info("Get parent context")
        ...
        >>> def func():
        ...     wrapped()
        ...
        >>> func()
        [18:11:54] DEBUG in 'func' - Get parent context
        """
        if ansi:
            colors = True
            warnings.warn(
                "The 'ansi' parameter is deprecated, please use 'colors' instead",
                DeprecationWarning,
            )

        args = self._options[-2:]
        return Logger(self._core, exception, depth, record, lazy, colors, raw, capture, *args)

    def bind(__self, **kwargs):  # noqa: N805
        """Bind attributes to the ``extra`` dict of each logged message record.

        This is used to add custom context to each logging call.

        Parameters
        ----------
        **kwargs
            Mapping between keys and values that will be added to the ``extra`` dict.

        Returns
        -------
        :class:`~Logger`
            A logger wrapping the core logger, but which sends record with the customized ``extra``
            dict.

        Examples
        --------
        >>> logger.add(sys.stderr, format="{extra[ip]} - {message}")
        >>> class Server:
        ...     def __init__(self, ip):
        ...         self.ip = ip
        ...         self.logger = logger.bind(ip=ip)
        ...     def call(self, message):
        ...         self.logger.info(message)
        ...
        >>> instance_1 = Server("192.168.0.200")
        >>> instance_2 = Server("127.0.0.1")
        >>> instance_1.call("First instance")
        192.168.0.200 - First instance
        >>> instance_2.call("Second instance")
        127.0.0.1 - Second instance
        """
        *options, extra = __self._options
        return Logger(__self._core, *options, {**extra, **kwargs})

    @contextlib.contextmanager
    def contextualize(__self, **kwargs):  # noqa: N805
        """Bind attributes to the context-local ``extra`` dict while inside the ``with`` block.

        Contrary to |bind| there is no ``logger`` returned, the ``extra`` dict is modified in-place
        and updated globally. Most importantly, it uses |contextvars| which means that
        contextualized values are unique to each threads and asynchronous tasks.

        The ``extra`` dict will retrieve its initial state once the context manager is exited.

        Parameters
        ----------
        **kwargs
            Mapping between keys and values that will be added to the context-local ``extra`` dict.

        Returns
        -------
        :term:`context manager` / :term:`decorator`
            A context manager (usable as a decorator too) that will bind the attributes once entered
            and restore the initial state of the ``extra`` dict while exited.

        Examples
        --------
        >>> logger.add(sys.stderr, format="{message} | {extra}")
        1
        >>> def task():
        ...     logger.info("Processing!")
        ...
        >>> with logger.contextualize(task_id=123):
        ...     task()
        ...
        Processing! | {'task_id': 123}
        >>> logger.info("Done.")
        Done. | {}
        """
        with __self._core.lock:
            new_context = {**context.get(), **kwargs}
            token = context.set(new_context)

        try:
            yield
        finally:
            with __self._core.lock:
                context.reset(token)

    def patch(self, patcher):
        """Attach a function to modify the record dict created by each logging call.

        The ``patcher`` may be used to update the record on-the-fly before it's propagated to the
        handlers. This allows the "extra" dict to be populated with dynamic values and also permits
        advanced modifications of the record emitted while logging a message. The function is called
        once before sending the log message to the different handlers.

        It is recommended to apply modification on the ``record["extra"]`` dict rather than on the
        ``record`` dict itself, as some values are used internally by `Loguru`, and modify them may
        produce unexpected results.

        The logger can be patched multiple times. In this case, the functions are called in the
        same order as they are added.

        Parameters
        ----------
        patcher: |callable|_
            The function to which the record dict will be passed as the sole argument. This function
            is in charge of updating the record in-place, the function does not need to return any
            value, the modified record object will be re-used.

        Returns
        -------
        :class:`~Logger`
            A logger wrapping the core logger, but which records are passed through the ``patcher``
            function before being sent to the added handlers.

        Examples
        --------
        >>> logger.add(sys.stderr, format="{extra[utc]} {message}")
        >>> logger = logger.patch(lambda record: record["extra"].update(utc=datetime.utcnow())
        >>> logger.info("That's way, you can log messages with time displayed in UTC")

        >>> def wrapper(func):
        ...     @functools.wraps(func)
        ...     def wrapped(*args, **kwargs):
        ...         logger.patch(lambda r: r.update(function=func.__name__)).info("Wrapped!")
        ...         return func(*args, **kwargs)
        ...     return wrapped

        >>> def recv_record_from_network(pipe):
        ...     record = pickle.loads(pipe.read())
        ...     level, message = record["level"], record["message"]
        ...     logger.patch(lambda r: r.update(record)).log(level, message)
        """
        *options, patchers, extra = self._options
        return Logger(self._core, *options, patchers + [patcher], extra)

    def level(self, name, no=None, color=None, icon=None):
        """Add, update or retrieve a logging level.

        Logging levels are defined by their ``name`` to which a severity ``no``, an ansi ``color``
        tag and an ``icon`` are associated and possibly modified at run-time. To |log| to a custom
        level, you should necessarily use its name, the severity number is not linked back to levels
        name (this implies that several levels can share the same severity).

        To add a new level, its ``name`` and its ``no`` are required. A ``color`` and an ``icon``
        can also be specified or will be empty by default.

        To update an existing level, pass its ``name`` with the parameters to be changed. It is not
        possible to modify the ``no`` of a level once it has been added.

        To retrieve level information, the ``name`` solely suffices.

        Parameters
        ----------
        name : |str|
            The name of the logging level.
        no : |int|
            The severity of the level to be added or updated.
        color : |str|
            The color markup of the level to be added or updated.
        icon : |str|
            The icon of the level to be added or updated.

        Returns
        -------
        ``Level``
            A |namedtuple| containing information about the level.

        Raises
        ------
        ValueError
            If there is no level registered with such ``name``.

        Examples
        --------
        >>> level = logger.level("ERROR")
        >>> print(level)
        Level(name='ERROR', no=40, color='<red><bold>', icon='❌')
        >>> logger.add(sys.stderr, format="{level.no} {level.icon} {message}")
        1
        >>> logger.level("CUSTOM", no=15, color="<blue>", icon="@")
        Level(name='CUSTOM', no=15, color='<blue>', icon='@')
        >>> logger.log("CUSTOM", "Logging...")
        15 @ Logging...
        >>> logger.level("WARNING", icon=r"/!\\")
        Level(name='WARNING', no=30, color='<yellow><bold>', icon='/!\\\\')
        >>> logger.warning("Updated!")
        30 /!\\ Updated!
        """
        if not isinstance(name, str):
            raise TypeError(
                "Invalid level name, it should be a string, not: '%s'" % type(name).__name__
            )

        if no is color is icon is None:
            try:
                return self._core.levels[name]
            except KeyError:
                raise ValueError("Level '%s' does not exist" % name) from None

        if name not in self._core.levels:
            if no is None:
                raise ValueError(
                    "Level '%s' does not exist, you have to create it by specifying a level no"
                    % name
                )
            else:
                old_color, old_icon = "", " "
        elif no is not None:
            raise TypeError("Level '%s' already exists, you can't update its severity no" % name)
        else:
            _, no, old_color, old_icon = self.level(name)

        if color is None:
            color = old_color

        if icon is None:
            icon = old_icon

        if not isinstance(no, int):
            raise TypeError(
                "Invalid level no, it should be an integer, not: '%s'" % type(no).__name__
            )

        if no < 0:
            raise ValueError("Invalid level no, it should be a positive integer, not: %d" % no)

        ansi = Colorizer.ansify(color)
        level = Level(name, no, color, icon)

        with self._core.lock:
            self._core.levels[name] = level
            self._core.levels_ansi_codes[name] = ansi
            self._core.levels_lookup[name] = (name, name, no, icon)
            for handler in self._core.handlers.values():
                handler.update_format(name)

        return level

    def disable(self, name):
        """Disable logging of messages coming from ``name`` module and its children.

        Developers of library using `Loguru` should absolutely disable it to avoid disrupting
        users with unrelated logs messages.

        Note that in some rare circumstances, it is not possible for `Loguru` to
        determine the module's ``__name__`` value. In such situation, ``record["name"]`` will be
        equal to ``None``, this is why ``None`` is also a valid argument.

        Parameters
        ----------
        name : |str| or ``None``
            The name of the parent module to disable.

        Examples
        --------
        >>> logger.info("Allowed message by default")
        [22:21:55] Allowed message by default
        >>> logger.disable("my_library")
        >>> logger.info("While publishing a library, don't forget to disable logging")
        """
        self._change_activation(name, False)

    def enable(self, name):
        """Enable logging of messages coming from ``name`` module and its children.

        Logging is generally disabled by imported library using `Loguru`, hence this function
        allows users to receive these messages anyway.

        To enable all logs regardless of the module they are coming from, an empty string ``""`` can
        be passed.

        Parameters
        ----------
        name : |str| or ``None``
            The name of the parent module to re-allow.

        Examples
        --------
        >>> logger.disable("__main__")
        >>> logger.info("Disabled, so nothing is logged.")
        >>> logger.enable("__main__")
        >>> logger.info("Re-enabled, messages are logged.")
        [22:46:12] Re-enabled, messages are logged.
        """
        self._change_activation(name, True)

    def configure(self, *, handlers=None, levels=None, extra=None, patcher=None, activation=None):
        """Configure the core logger.

        It should be noted that ``extra`` values set using this function are available across all
        modules, so this is the best way to set overall default values.

        Parameters
        ----------
        handlers : |list| of |dict|, optional
            A list of each handler to be added. The list should contain dicts of params passed to
            the |add| function as keyword arguments. If not ``None``, all previously added
            handlers are first removed.
        levels : |list| of |dict|, optional
            A list of each level to be added or updated. The list should contain dicts of params
            passed to the |level| function as keyword arguments. This will never remove previously
            created levels.
        extra : |dict|, optional
            A dict containing additional parameters bound to the core logger, useful to share
            common properties if you call |bind| in several of your files modules. If not ``None``,
            this will remove previously configured ``extra`` dict.
        patcher : |callable|_, optional
            A function that will be applied to the record dict of each logged messages across all
            modules using the logger. It should modify the dict in-place without returning anything.
            The function is executed prior to the one possibly added by the |patch| method. If not
            ``None``, this will replace previously configured ``patcher`` function.
        activation : |list| of |tuple|, optional
            A list of ``(name, state)`` tuples which denotes which loggers should be enabled (if
            ``state`` is ``True``) or disabled (if ``state`` is ``False``). The calls to |enable|
            and |disable| are made accordingly to the list order. This will not modify previously
            activated loggers, so if you need a fresh start prepend your list with ``("", False)``
            or ``("", True)``.

        Returns
        -------
        :class:`list` of :class:`int`
            A list containing the identifiers of added sinks (if any).

        Examples
        --------
        >>> logger.configure(
        ...     handlers=[
        ...         dict(sink=sys.stderr, format="[{time}] {message}"),
        ...         dict(sink="file.log", enqueue=True, serialize=True),
        ...     ],
        ...     levels=[dict(name="NEW", no=13, icon="¤", color="")],
        ...     extra={"common_to_all": "default"},
        ...     patcher=lambda record: record["extra"].update(some_value=42),
        ...     activation=[("my_module.secret", False), ("another_library.module", True)],
        ... )
        [1, 2]

        >>> # Set a default "extra" dict to logger across all modules, without "bind()"
        >>> extra = {"context": "foo"}
        >>> logger.configure(extra=extra)
        >>> logger.add(sys.stderr, format="{extra[context]} - {message}")
        >>> logger.info("Context without bind")
        >>> # => "foo - Context without bind"
        >>> logger.bind(context="bar").info("Suppress global context")
        >>> # => "bar - Suppress global context"
        """
        if handlers is not None:
            self.remove()
        else:
            handlers = []

        if levels is not None:
            for params in levels:
                self.level(**params)

        if patcher is not None:
            with self._core.lock:
                self._core.patcher = patcher

        if extra is not None:
            with self._core.lock:
                self._core.extra.clear()
                self._core.extra.update(extra)

        if activation is not None:
            for name, state in activation:
                if state:
                    self.enable(name)
                else:
                    self.disable(name)

        return [self.add(**params) for params in handlers]

    def _change_activation(self, name, status):
        if not (name is None or isinstance(name, str)):
            raise TypeError(
                "Invalid name, it should be a string (or None), not: '%s'" % type(name).__name__
            )

        with self._core.lock:
            enabled = self._core.enabled.copy()

            if name is None:
                for n in enabled:
                    if n is None:
                        enabled[n] = status
                self._core.activation_none = status
                self._core.enabled = enabled
                return

            if name != "":
                name += "."

            activation_list = [
                (n, s) for n, s in self._core.activation_list if n[: len(name)] != name
            ]

            parent_status = next((s for n, s in activation_list if name[: len(n)] == n), None)
            if parent_status != status and not (name == "" and status is True):
                activation_list.append((name, status))

                def modules_depth(x):
                    return x[0].count(".")

                activation_list.sort(key=modules_depth, reverse=True)

            for n in enabled:
                if n is not None and (n + ".")[: len(name)] == name:
                    enabled[n] = status

            self._core.activation_list = activation_list
            self._core.enabled = enabled

    @staticmethod
    def parse(file, pattern, *, cast={}, chunk=2**16):  # noqa: B006
        """Parse raw logs and extract each entry as a |dict|.

        The logging format has to be specified as the regex ``pattern``, it will then be
        used to parse the ``file`` and retrieve each entry based on the named groups present
        in the regex.

        Parameters
        ----------
        file : |str|, |Path| or |file-like object|_
            The path of the log file to be parsed, or an already opened file object.
        pattern : |str| or |re.Pattern|_
            The regex to use for logs parsing, it should contain named groups which will be included
            in the returned dict.
        cast : |callable|_ or |dict|, optional
            A function that should convert in-place the regex groups parsed (a dict of string
            values) to more appropriate types. If a dict is passed, it should be a mapping between
            keys of parsed log dict and the function that should be used to convert the associated
            value.
        chunk : |int|, optional
            The number of bytes read while iterating through the logs, this avoids having to load
            the whole file in memory.

        Yields
        ------
        :class:`dict`
            The dict mapping regex named groups to matched values, as returned by |match.groupdict|
            and optionally converted according to ``cast`` argument.

        Examples
        --------
        >>> reg = r"(?P<lvl>[0-9]+): (?P<msg>.*)"    # If log format is "{level.no} - {message}"
        >>> for e in logger.parse("file.log", reg):  # A file line could be "10 - A debug message"
        ...     print(e)                             # => {'lvl': '10', 'msg': 'A debug message'}

        >>> caster = dict(lvl=int)                   # Parse 'lvl' key as an integer
        >>> for e in logger.parse("file.log", reg, cast=caster):
        ...     print(e)                             # => {'lvl': 10, 'msg': 'A debug message'}

        >>> def cast(groups):
        ...     if "date" in groups:
        ...         groups["date"] = datetime.strptime(groups["date"], "%Y-%m-%d %H:%M:%S")
        ...
        >>> with open("file.log") as file:
        ...     for log in logger.parse(file, reg, cast=cast):
        ...         print(log["date"], log["something_else"])
        """
        if isinstance(file, (str, PathLike)):
            should_close = True
            fileobj = open(str(file))
        elif hasattr(file, "read") and callable(file.read):
            should_close = False
            fileobj = file
        else:
            raise TypeError(
                "Invalid file, it should be a string path or a file object, not: '%s'"
                % type(file).__name__
            )

        if isinstance(cast, dict):

            def cast_function(groups):
                for key, converter in cast.items():
                    if key in groups:
                        groups[key] = converter(groups[key])

        elif callable(cast):
            cast_function = cast
        else:
            raise TypeError(
                "Invalid cast, it should be a function or a dict, not: '%s'" % type(cast).__name__
            )

        try:
            regex = re.compile(pattern)
        except TypeError:
            raise TypeError(
                "Invalid pattern, it should be a string or a compiled regex, not: '%s'"
                % type(pattern).__name__
            ) from None

        matches = Logger._find_iter(fileobj, regex, chunk)

        for match in matches:
            groups = match.groupdict()
            cast_function(groups)
            yield groups

        if should_close:
            fileobj.close()

    @staticmethod
    def _find_iter(fileobj, regex, chunk):
        buffer = fileobj.read(0)

        while 1:
            text = fileobj.read(chunk)
            buffer += text
            matches = list(regex.finditer(buffer))

            if not text:
                yield from matches
                break

            if len(matches) > 1:
                end = matches[-2].end()
                buffer = buffer[end:]
                yield from matches[:-1]

    def _log(self, level, from_decorator, options, message, args, kwargs):
        core = self._core

        if not core.handlers:
            return

        try:
            level_id, level_name, level_no, level_icon = core.levels_lookup[level]
        except (KeyError, TypeError):
            if isinstance(level, str):
                raise ValueError("Level '%s' does not exist" % level) from None
            if not isinstance(level, int):
                raise TypeError(
                    "Invalid level, it should be an integer or a string, not: '%s'"
                    % type(level).__name__
                ) from None
            if level < 0:
                raise ValueError(
                    "Invalid level value, it should be a positive integer, not: %d" % level
                ) from None
            cache = (None, "Level %d" % level, level, " ")
            level_id, level_name, level_no, level_icon = cache
            core.levels_lookup[level] = cache

        if level_no < core.min_level:
            return

        (exception, depth, record, lazy, colors, raw, capture, patchers, extra) = options

        frame = get_frame(depth + 2)

        try:
            name = frame.f_globals["__name__"]
        except KeyError:
            name = None

        try:
            if not core.enabled[name]:
                return
        except KeyError:
            enabled = core.enabled
            if name is None:
                status = core.activation_none
                enabled[name] = status
                if not status:
                    return
            else:
                dotted_name = name + "."
                for dotted_module_name, status in core.activation_list:
                    if dotted_name[: len(dotted_module_name)] == dotted_module_name:
                        if status:
                            break
                        enabled[name] = False
                        return
                enabled[name] = True

        current_datetime = aware_now()

        code = frame.f_code
        file_path = code.co_filename
        file_name = basename(file_path)
        thread = current_thread()
        process = current_process()
        elapsed = current_datetime - start_time

        if exception:
            if isinstance(exception, BaseException):
                type_, value, traceback = (type(exception), exception, exception.__traceback__)
            elif isinstance(exception, tuple):
                type_, value, traceback = exception
            else:
                type_, value, traceback = sys.exc_info()
            exception = RecordException(type_, value, traceback)
        else:
            exception = None

        log_record = {
            "elapsed": elapsed,
            "exception": exception,
            "extra": {**core.extra, **context.get(), **extra},
            "file": RecordFile(file_name, file_path),
            "function": code.co_name,
            "level": RecordLevel(level_name, level_no, level_icon),
            "line": frame.f_lineno,
            "message": str(message),
            "module": splitext(file_name)[0],
            "name": name,
            "process": RecordProcess(process.ident, process.name),
            "thread": RecordThread(thread.ident, thread.name),
            "time": current_datetime,
        }

        if lazy:
            args = [arg() for arg in args]
            kwargs = {key: value() for key, value in kwargs.items()}

        if capture and kwargs:
            log_record["extra"].update(kwargs)

        if record:
            if "record" in kwargs:
                raise TypeError(
                    "The message can't be formatted: 'record' shall not be used as a keyword "
                    "argument while logger has been configured with '.opt(record=True)'"
                )
            kwargs.update(record=log_record)

        if colors:
            if args or kwargs:
                colored_message = Colorizer.prepare_message(message, args, kwargs)
            else:
                colored_message = Colorizer.prepare_simple_message(str(message))
            log_record["message"] = colored_message.stripped
        elif args or kwargs:
            colored_message = None
            log_record["message"] = message.format(*args, **kwargs)
        else:
            colored_message = None

        if core.patcher:
            core.patcher(log_record)

        for patcher in patchers:
            patcher(log_record)

        for handler in core.handlers.values():
            handler.emit(log_record, level_id, from_decorator, raw, colored_message)

    def trace(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'TRACE'``."""
        __self._log("TRACE", False, __self._options, __message, args, kwargs)

    def debug(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'DEBUG'``."""
        __self._log("DEBUG", False, __self._options, __message, args, kwargs)

    def info(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'INFO'``."""
        __self._log("INFO", False, __self._options, __message, args, kwargs)

    def success(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'SUCCESS'``."""
        __self._log("SUCCESS", False, __self._options, __message, args, kwargs)

    def warning(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'WARNING'``."""
        __self._log("WARNING", False, __self._options, __message, args, kwargs)

    def error(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'ERROR'``."""
        __self._log("ERROR", False, __self._options, __message, args, kwargs)

    def critical(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'CRITICAL'``."""
        __self._log("CRITICAL", False, __self._options, __message, args, kwargs)

    def exception(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Convenience method for logging an ``'ERROR'`` with exception information."""
        options = (True,) + __self._options[1:]
        __self._log("ERROR", False, options, __message, args, kwargs)

    def log(__self, __level, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``level``."""
        __self._log(__level, False, __self._options, __message, args, kwargs)

    def start(self, *args, **kwargs):
        """Deprecated function to |add| a new handler.

        Warnings
        --------
        .. deprecated:: 0.2.2
          ``start()`` will be removed in Loguru 1.0.0, it is replaced by ``add()`` which is a less
          confusing name.
        """
        warnings.warn(
            "The 'start()' method is deprecated, please use 'add()' instead", DeprecationWarning
        )
        return self.add(*args, **kwargs)

    def stop(self, *args, **kwargs):
        """Deprecated function to |remove| an existing handler.

        Warnings
        --------
        .. deprecated:: 0.2.2
          ``stop()`` will be removed in Loguru 1.0.0, it is replaced by ``remove()`` which is a less
          confusing name.
        """
        warnings.warn(
            "The 'stop()' method is deprecated, please use 'remove()' instead", DeprecationWarning
        )
        return self.remove(*args, **kwargs)
