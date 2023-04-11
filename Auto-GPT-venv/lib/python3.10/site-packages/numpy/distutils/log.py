# Colored log
import sys
from distutils.log import *  # noqa: F403
from distutils.log import Log as old_Log
from distutils.log import _global_log

from numpy.distutils.misc_util import (red_text, default_text, cyan_text,
        green_text, is_sequence, is_string)


def _fix_args(args,flag=1):
    if is_string(args):
        return args.replace('%', '%%')
    if flag and is_sequence(args):
        return tuple([_fix_args(a, flag=0) for a in args])
    return args


class Log(old_Log):
    def _log(self, level, msg, args):
        if level >= self.threshold:
            if args:
                msg = msg % _fix_args(args)
            if 0:
                if msg.startswith('copying ') and msg.find(' -> ') != -1:
                    return
                if msg.startswith('byte-compiling '):
                    return
            print(_global_color_map[level](msg))
            sys.stdout.flush()

    def good(self, msg, *args):
        """
        If we log WARN messages, log this message as a 'nice' anti-warn
        message.

        """
        if WARN >= self.threshold:
            if args:
                print(green_text(msg % _fix_args(args)))
            else:
                print(green_text(msg))
            sys.stdout.flush()


_global_log.__class__ = Log

good = _global_log.good

def set_threshold(level, force=False):
    prev_level = _global_log.threshold
    if prev_level > DEBUG or force:
        # If we're running at DEBUG, don't change the threshold, as there's
        # likely a good reason why we're running at this level.
        _global_log.threshold = level
        if level <= DEBUG:
            info('set_threshold: setting threshold to DEBUG level,'
                    ' it can be changed only with force argument')
    else:
        info('set_threshold: not changing threshold from DEBUG level'
                ' %s to %s' % (prev_level, level))
    return prev_level

def get_threshold():
	return _global_log.threshold

def set_verbosity(v, force=False):
    prev_level = _global_log.threshold
    if v < 0:
        set_threshold(ERROR, force)
    elif v == 0:
        set_threshold(WARN, force)
    elif v == 1:
        set_threshold(INFO, force)
    elif v >= 2:
        set_threshold(DEBUG, force)
    return {FATAL:-2,ERROR:-1,WARN:0,INFO:1,DEBUG:2}.get(prev_level, 1)


_global_color_map = {
    DEBUG:cyan_text,
    INFO:default_text,
    WARN:red_text,
    ERROR:red_text,
    FATAL:red_text
}

# don't use INFO,.. flags in set_verbosity, these flags are for set_threshold.
set_verbosity(0, force=True)


_error = error
_warn = warn
_info = info
_debug = debug


def error(msg, *a, **kw):
    _error(f"ERROR: {msg}", *a, **kw)


def warn(msg, *a, **kw):
    _warn(f"WARN: {msg}", *a, **kw)


def info(msg, *a, **kw):
    _info(f"INFO: {msg}", *a, **kw)


def debug(msg, *a, **kw):
    _debug(f"DEBUG: {msg}", *a, **kw)
