"""
Helper functionality for interoperability with stdlib `logging`.
"""
import logging
import sys
from contextlib import contextmanager

try:
    from typing import Iterator, List, Optional, Type  # pylint: disable=unused-import
except ImportError:
    pass

from ..std import tqdm as std_tqdm


class _TqdmLoggingHandler(logging.StreamHandler):
    def __init__(
        self,
        tqdm_class=std_tqdm  # type: Type[std_tqdm]
    ):
        super(_TqdmLoggingHandler, self).__init__()
        self.tqdm_class = tqdm_class

    def emit(self, record):
        try:
            msg = self.format(record)
            self.tqdm_class.write(msg, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:  # noqa pylint: disable=bare-except
            self.handleError(record)


def _is_console_logging_handler(handler):
    return (isinstance(handler, logging.StreamHandler)
            and handler.stream in {sys.stdout, sys.stderr})


def _get_first_found_console_logging_handler(handlers):
    for handler in handlers:
        if _is_console_logging_handler(handler):
            return handler


@contextmanager
def logging_redirect_tqdm(
    loggers=None,  # type: Optional[List[logging.Logger]],
    tqdm_class=std_tqdm  # type: Type[std_tqdm]
):
    # type: (...) -> Iterator[None]
    """
    Context manager redirecting console logging to `tqdm.write()`, leaving
    other logging handlers (e.g. log files) unaffected.

    Parameters
    ----------
    loggers  : list, optional
      Which handlers to redirect (default: [logging.root]).
    tqdm_class  : optional

    Example
    -------
    ```python
    import logging
    from tqdm import trange
    from tqdm.contrib.logging import logging_redirect_tqdm

    LOG = logging.getLogger(__name__)

    if __name__ == '__main__':
        logging.basicConfig(level=logging.INFO)
        with logging_redirect_tqdm():
            for i in trange(9):
                if i == 4:
                    LOG.info("console logging redirected to `tqdm.write()`")
        # logging restored
    ```
    """
    if loggers is None:
        loggers = [logging.root]
    original_handlers_list = [logger.handlers for logger in loggers]
    try:
        for logger in loggers:
            tqdm_handler = _TqdmLoggingHandler(tqdm_class)
            orig_handler = _get_first_found_console_logging_handler(logger.handlers)
            if orig_handler is not None:
                tqdm_handler.setFormatter(orig_handler.formatter)
                tqdm_handler.stream = orig_handler.stream
            logger.handlers = [
                handler for handler in logger.handlers
                if not _is_console_logging_handler(handler)] + [tqdm_handler]
        yield
    finally:
        for logger, original_handlers in zip(loggers, original_handlers_list):
            logger.handlers = original_handlers


@contextmanager
def tqdm_logging_redirect(
    *args,
    # loggers=None,  # type: Optional[List[logging.Logger]]
    # tqdm=None,  # type: Optional[Type[tqdm.tqdm]]
    **kwargs
):
    # type: (...) -> Iterator[None]
    """
    Convenience shortcut for:
    ```python
    with tqdm_class(*args, **tqdm_kwargs) as pbar:
        with logging_redirect_tqdm(loggers=loggers, tqdm_class=tqdm_class):
            yield pbar
    ```

    Parameters
    ----------
    tqdm_class  : optional, (default: tqdm.std.tqdm).
    loggers  : optional, list.
    **tqdm_kwargs  : passed to `tqdm_class`.
    """
    tqdm_kwargs = kwargs.copy()
    loggers = tqdm_kwargs.pop('loggers', None)
    tqdm_class = tqdm_kwargs.pop('tqdm_class', std_tqdm)
    with tqdm_class(*args, **tqdm_kwargs) as pbar:
        with logging_redirect_tqdm(loggers=loggers, tqdm_class=tqdm_class):
            yield pbar
