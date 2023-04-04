"""adodbapi - A python DB API 2.0 (PEP 249) interface to Microsoft ADO

Copyright (C) 2002 Henrik Ekelund, version 2.1 by Vernon Cole
* http://sourceforge.net/projects/adodbapi
"""
import sys
import time

from .adodbapi import Connection, Cursor, __version__, connect, dateconverter
from .apibase import (
    BINARY,
    DATETIME,
    NUMBER,
    ROWID,
    STRING,
    DatabaseError,
    DataError,
    Error,
    FetchFailedError,
    IntegrityError,
    InterfaceError,
    InternalError,
    NotSupportedError,
    OperationalError,
    ProgrammingError,
    Warning,
    apilevel,
    paramstyle,
    threadsafety,
)


def Binary(aString):
    """This function constructs an object capable of holding a binary (long) string value."""
    return bytes(aString)


def Date(year, month, day):
    "This function constructs an object holding a date value."
    return dateconverter.Date(year, month, day)


def Time(hour, minute, second):
    "This function constructs an object holding a time value."
    return dateconverter.Time(hour, minute, second)


def Timestamp(year, month, day, hour, minute, second):
    "This function constructs an object holding a time stamp value."
    return dateconverter.Timestamp(year, month, day, hour, minute, second)


def DateFromTicks(ticks):
    """This function constructs an object holding a date value from the given ticks value
    (number of seconds since the epoch; see the documentation of the standard Python time module for details).
    """
    return Date(*time.gmtime(ticks)[:3])


def TimeFromTicks(ticks):
    """This function constructs an object holding a time value from the given ticks value
    (number of seconds since the epoch; see the documentation of the standard Python time module for details).
    """
    return Time(*time.gmtime(ticks)[3:6])


def TimestampFromTicks(ticks):
    """This function constructs an object holding a time stamp value from the given
    ticks value (number of seconds since the epoch;
    see the documentation of the standard Python time module for details)."""
    return Timestamp(*time.gmtime(ticks)[:6])


version = "adodbapi v" + __version__
