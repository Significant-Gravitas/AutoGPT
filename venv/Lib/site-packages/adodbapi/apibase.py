"""adodbapi.apibase - A python DB API 2.0 (PEP 249) interface to Microsoft ADO

Copyright (C) 2002 Henrik Ekelund, version 2.1 by Vernon Cole
* http://sourceforge.net/projects/pywin32
* http://sourceforge.net/projects/adodbapi
"""

import datetime
import decimal
import numbers
import sys
import time

# noinspection PyUnresolvedReferences
from . import ado_consts as adc

verbose = False  # debugging flag

onIronPython = sys.platform == "cli"
if onIronPython:  # we need type definitions for odd data we may need to convert
    # noinspection PyUnresolvedReferences
    from System import DateTime, DBNull

    NullTypes = (type(None), DBNull)
else:
    DateTime = type(NotImplemented)  # should never be seen on win32
    NullTypes = type(None)

# --- define objects to smooth out Python3 <-> Python 2.x differences
unicodeType = str
longType = int
StringTypes = str
makeByteBuffer = bytes
memoryViewType = memoryview
_BaseException = Exception

try:  # jdhardy -- handle bytes under IronPython & Py3
    bytes
except NameError:
    bytes = str  # define it for old Pythons


# ------- Error handlers ------
def standardErrorHandler(connection, cursor, errorclass, errorvalue):
    err = (errorclass, errorvalue)
    try:
        connection.messages.append(err)
    except:
        pass
    if cursor is not None:
        try:
            cursor.messages.append(err)
        except:
            pass
    raise errorclass(errorvalue)


# Note: _BaseException is defined differently between Python 2.x and 3.x
class Error(_BaseException):
    pass  # Exception that is the base class of all other error
    # exceptions. You can use this to catch all errors with one
    # single 'except' statement. Warnings are not considered
    # errors and thus should not use this class as base. It must
    # be a subclass of the Python StandardError (defined in the
    # module exceptions).


class Warning(_BaseException):
    pass


class InterfaceError(Error):
    pass


class DatabaseError(Error):
    pass


class InternalError(DatabaseError):
    pass


class OperationalError(DatabaseError):
    pass


class ProgrammingError(DatabaseError):
    pass


class IntegrityError(DatabaseError):
    pass


class DataError(DatabaseError):
    pass


class NotSupportedError(DatabaseError):
    pass


class FetchFailedError(OperationalError):
    """
    Error is used by RawStoredProcedureQuerySet to determine when a fetch
    failed due to a connection being closed or there is no record set
    returned. (Non-standard, added especially for django)
    """

    pass


# # # # # ----- Type Objects and Constructors ----- # # # # #
# Many databases need to have the input in a particular format for binding to an operation's input parameters.
# For example, if an input is destined for a DATE column, then it must be bound to the database in a particular
# string format. Similar problems exist for "Row ID" columns or large binary items (e.g. blobs or RAW columns).
# This presents problems for Python since the parameters to the executeXXX() method are untyped.
# When the database module sees a Python string object, it doesn't know if it should be bound as a simple CHAR
# column, as a raw BINARY item, or as a DATE.
#
# To overcome this problem, a module must provide the constructors defined below to create objects that can
# hold special values. When passed to the cursor methods, the module can then detect the proper type of
# the input parameter and bind it accordingly.

# A Cursor Object's description attribute returns information about each of the result columns of a query.
# The type_code must compare equal to one of Type Objects defined below. Type Objects may be equal to more than
# one type code (e.g. DATETIME could be equal to the type codes for date, time and timestamp columns;
# see the Implementation Hints below for details).

# SQL NULL values are represented by the Python None singleton on input and output.

# Note: Usage of Unix ticks for database interfacing can cause troubles because of the limited date range they cover.


# def Date(year,month,day):
#     "This function constructs an object holding a date value. "
#     return dateconverter.date(year,month,day)  #dateconverter.Date(year,month,day)
#
# def Time(hour,minute,second):
#     "This function constructs an object holding a time value. "
#     return dateconverter.time(hour, minute, second) # dateconverter.Time(hour,minute,second)
#
# def Timestamp(year,month,day,hour,minute,second):
#     "This function constructs an object holding a time stamp value. "
#     return dateconverter.datetime(year,month,day,hour,minute,second)
#
# def DateFromTicks(ticks):
#     """This function constructs an object holding a date value from the given ticks value
#     (number of seconds since the epoch; see the documentation of the standard Python time module for details). """
#     return Date(*time.gmtime(ticks)[:3])
#
# def TimeFromTicks(ticks):
#     """This function constructs an object holding a time value from the given ticks value
#     (number of seconds since the epoch; see the documentation of the standard Python time module for details). """
#     return Time(*time.gmtime(ticks)[3:6])
#
# def TimestampFromTicks(ticks):
#     """This function constructs an object holding a time stamp value from the given
#     ticks value (number of seconds since the epoch;
#     see the documentation of the standard Python time module for details). """
#     return Timestamp(*time.gmtime(ticks)[:6])
#
# def Binary(aString):
#     """This function constructs an object capable of holding a binary (long) string value. """
#     b = makeByteBuffer(aString)
#     return b
# -----     Time converters ----------------------------------------------
class TimeConverter(object):  # this is a generic time converter skeleton
    def __init__(self):  # the details will be filled in by instances
        self._ordinal_1899_12_31 = datetime.date(1899, 12, 31).toordinal() - 1
        # Use cls.types to compare if an input parameter is a datetime
        self.types = {
            type(self.Date(2000, 1, 1)),
            type(self.Time(12, 1, 1)),
            type(self.Timestamp(2000, 1, 1, 12, 1, 1)),
            datetime.datetime,
            datetime.time,
            datetime.date,
        }

    def COMDate(self, obj):
        """Returns a ComDate from a date-time"""
        try:  # most likely a datetime
            tt = obj.timetuple()

            try:
                ms = obj.microsecond
            except:
                ms = 0
            return self.ComDateFromTuple(tt, ms)
        except:  # might be a tuple
            try:
                return self.ComDateFromTuple(obj)
            except:  # try an mxdate
                try:
                    return obj.COMDate()
                except:
                    raise ValueError('Cannot convert "%s" to COMdate.' % repr(obj))

    def ComDateFromTuple(self, t, microseconds=0):
        d = datetime.date(t[0], t[1], t[2])
        integerPart = d.toordinal() - self._ordinal_1899_12_31
        ms = (t[3] * 3600 + t[4] * 60 + t[5]) * 1000000 + microseconds
        fractPart = float(ms) / 86400000000.0
        return integerPart + fractPart

    def DateObjectFromCOMDate(self, comDate):
        "Returns an object of the wanted type from a ComDate"
        raise NotImplementedError  # "Abstract class"

    def Date(self, year, month, day):
        "This function constructs an object holding a date value."
        raise NotImplementedError  # "Abstract class"

    def Time(self, hour, minute, second):
        "This function constructs an object holding a time value."
        raise NotImplementedError  # "Abstract class"

    def Timestamp(self, year, month, day, hour, minute, second):
        "This function constructs an object holding a time stamp value."
        raise NotImplementedError  # "Abstract class"
        # all purpose date to ISO format converter

    def DateObjectToIsoFormatString(self, obj):
        "This function should return a string in the format 'YYYY-MM-dd HH:MM:SS:ms' (ms optional)"
        try:  # most likely, a datetime.datetime
            s = obj.isoformat(" ")
        except (TypeError, AttributeError):
            if isinstance(obj, datetime.date):
                s = obj.isoformat() + " 00:00:00"  # return exact midnight
            else:
                try:  # maybe it has a strftime method, like mx
                    s = obj.strftime("%Y-%m-%d %H:%M:%S")
                except AttributeError:
                    try:  # but may be time.struct_time
                        s = time.strftime("%Y-%m-%d %H:%M:%S", obj)
                    except:
                        raise ValueError('Cannot convert "%s" to isoformat' % repr(obj))
        return s


# -- Optional: if mx extensions are installed you may use mxDateTime ----
try:
    import mx.DateTime

    mxDateTime = True
except:
    mxDateTime = False
if mxDateTime:

    class mxDateTimeConverter(TimeConverter):  # used optionally if installed
        def __init__(self):
            TimeConverter.__init__(self)
            self.types.add(type(mx.DateTime))

        def DateObjectFromCOMDate(self, comDate):
            return mx.DateTime.DateTimeFromCOMDate(comDate)

        def Date(self, year, month, day):
            return mx.DateTime.Date(year, month, day)

        def Time(self, hour, minute, second):
            return mx.DateTime.Time(hour, minute, second)

        def Timestamp(self, year, month, day, hour, minute, second):
            return mx.DateTime.Timestamp(year, month, day, hour, minute, second)

else:

    class mxDateTimeConverter(TimeConverter):
        pass  # if no mx is installed


class pythonDateTimeConverter(TimeConverter):  # standard since Python 2.3
    def __init__(self):
        TimeConverter.__init__(self)

    def DateObjectFromCOMDate(self, comDate):
        if isinstance(comDate, datetime.datetime):
            odn = comDate.toordinal()
            tim = comDate.time()
            new = datetime.datetime.combine(datetime.datetime.fromordinal(odn), tim)
            return new
            # return comDate.replace(tzinfo=None) # make non aware
        elif isinstance(comDate, DateTime):
            fComDate = comDate.ToOADate()  # ironPython clr Date/Time
        else:
            fComDate = float(comDate)  # ComDate is number of days since 1899-12-31
        integerPart = int(fComDate)
        floatpart = fComDate - integerPart
        ##if floatpart == 0.0:
        ##    return datetime.date.fromordinal(integerPart + self._ordinal_1899_12_31)
        dte = datetime.datetime.fromordinal(
            integerPart + self._ordinal_1899_12_31
        ) + datetime.timedelta(milliseconds=floatpart * 86400000)
        # millisecondsperday=86400000 # 24*60*60*1000
        return dte

    def Date(self, year, month, day):
        return datetime.date(year, month, day)

    def Time(self, hour, minute, second):
        return datetime.time(hour, minute, second)

    def Timestamp(self, year, month, day, hour, minute, second):
        return datetime.datetime(year, month, day, hour, minute, second)


class pythonTimeConverter(TimeConverter):  # the old, ?nix type date and time
    def __init__(self):  # caution: this Class gets confised by timezones and DST
        TimeConverter.__init__(self)
        self.types.add(time.struct_time)

    def DateObjectFromCOMDate(self, comDate):
        "Returns ticks since 1970"
        if isinstance(comDate, datetime.datetime):
            return comDate.timetuple()
        elif isinstance(comDate, DateTime):  # ironPython clr date/time
            fcomDate = comDate.ToOADate()
        else:
            fcomDate = float(comDate)
        secondsperday = 86400  # 24*60*60
        # ComDate is number of days since 1899-12-31, gmtime epoch is 1970-1-1 = 25569 days
        t = time.gmtime(secondsperday * (fcomDate - 25569.0))
        return t  # year,month,day,hour,minute,second,weekday,julianday,daylightsaving=t

    def Date(self, year, month, day):
        return self.Timestamp(year, month, day, 0, 0, 0)

    def Time(self, hour, minute, second):
        return time.gmtime((hour * 60 + minute) * 60 + second)

    def Timestamp(self, year, month, day, hour, minute, second):
        return time.localtime(
            time.mktime((year, month, day, hour, minute, second, 0, 0, -1))
        )


base_dateconverter = pythonDateTimeConverter()

# ------ DB API required module attributes ---------------------
threadsafety = 1  # TODO -- find out whether this module is actually BETTER than 1.

apilevel = "2.0"  # String constant stating the supported DB API level.

paramstyle = "qmark"  # the default parameter style

# ------ control for an extension which may become part of DB API 3.0 ---
accepted_paramstyles = ("qmark", "named", "format", "pyformat", "dynamic")

# ------------------------------------------------------------------------------------------
# define similar types for generic conversion routines
adoIntegerTypes = (
    adc.adInteger,
    adc.adSmallInt,
    adc.adTinyInt,
    adc.adUnsignedInt,
    adc.adUnsignedSmallInt,
    adc.adUnsignedTinyInt,
    adc.adBoolean,
    adc.adError,
)  # max 32 bits
adoRowIdTypes = (adc.adChapter,)  # v2.1 Rose
adoLongTypes = (adc.adBigInt, adc.adFileTime, adc.adUnsignedBigInt)
adoExactNumericTypes = (
    adc.adDecimal,
    adc.adNumeric,
    adc.adVarNumeric,
    adc.adCurrency,
)  # v2.3 Cole
adoApproximateNumericTypes = (adc.adDouble, adc.adSingle)  # v2.1 Cole
adoStringTypes = (
    adc.adBSTR,
    adc.adChar,
    adc.adLongVarChar,
    adc.adLongVarWChar,
    adc.adVarChar,
    adc.adVarWChar,
    adc.adWChar,
)
adoBinaryTypes = (adc.adBinary, adc.adLongVarBinary, adc.adVarBinary)
adoDateTimeTypes = (adc.adDBTime, adc.adDBTimeStamp, adc.adDate, adc.adDBDate)
adoRemainingTypes = (
    adc.adEmpty,
    adc.adIDispatch,
    adc.adIUnknown,
    adc.adPropVariant,
    adc.adArray,
    adc.adUserDefined,
    adc.adVariant,
    adc.adGUID,
)


# this class is a trick to determine whether a type is a member of a related group of types. see PEP notes
class DBAPITypeObject(object):
    def __init__(self, valuesTuple):
        self.values = frozenset(valuesTuple)

    def __eq__(self, other):
        return other in self.values

    def __ne__(self, other):
        return other not in self.values


"""This type object is used to describe columns in a database that are string-based (e.g. CHAR). """
STRING = DBAPITypeObject(adoStringTypes)

"""This type object is used to describe (long) binary columns in a database (e.g. LONG, RAW, BLOBs). """
BINARY = DBAPITypeObject(adoBinaryTypes)

"""This type object is used to describe numeric columns in a database. """
NUMBER = DBAPITypeObject(
    adoIntegerTypes + adoLongTypes + adoExactNumericTypes + adoApproximateNumericTypes
)

"""This type object is used to describe date/time columns in a database. """

DATETIME = DBAPITypeObject(adoDateTimeTypes)
"""This type object is used to describe the "Row ID" column in a database. """
ROWID = DBAPITypeObject(adoRowIdTypes)

OTHER = DBAPITypeObject(adoRemainingTypes)

# ------- utilities for translating python data types to ADO data types ---------------------------------
typeMap = {
    memoryViewType: adc.adVarBinary,
    float: adc.adDouble,
    type(None): adc.adEmpty,
    str: adc.adBSTR,
    bool: adc.adBoolean,  # v2.1 Cole
    decimal.Decimal: adc.adDecimal,
    int: adc.adBigInt,
    bytes: adc.adVarBinary,
}


def pyTypeToADOType(d):
    tp = type(d)
    try:
        return typeMap[tp]
    except KeyError:  #   The type was not defined in the pre-computed Type table
        from . import dateconverter

        if (
            tp in dateconverter.types
        ):  # maybe it is one of our supported Date/Time types
            return adc.adDate
        #  otherwise, attempt to discern the type by probing the data object itself -- to handle duck typing
        if isinstance(d, StringTypes):
            return adc.adBSTR
        if isinstance(d, numbers.Integral):
            return adc.adBigInt
        if isinstance(d, numbers.Real):
            return adc.adDouble
        raise DataError('cannot convert "%s" (type=%s) to ADO' % (repr(d), tp))


# # # # # # # # # # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# functions to convert database values to Python objects
# ------------------------------------------------------------------------
# variant type : function converting variant to Python value
def variantConvertDate(v):
    from . import dateconverter  # this function only called when adodbapi is running

    return dateconverter.DateObjectFromCOMDate(v)


def cvtString(variant):  # use to get old action of adodbapi v1 if desired
    if onIronPython:
        try:
            return variant.ToString()
        except:
            pass
    return str(variant)


def cvtDecimal(variant):  # better name
    return _convertNumberWithCulture(variant, decimal.Decimal)


def cvtNumeric(variant):  # older name - don't break old code
    return cvtDecimal(variant)


def cvtFloat(variant):
    return _convertNumberWithCulture(variant, float)


def _convertNumberWithCulture(variant, f):
    try:
        return f(variant)
    except (ValueError, TypeError, decimal.InvalidOperation):
        try:
            europeVsUS = str(variant).replace(",", ".")
            return f(europeVsUS)
        except (ValueError, TypeError, decimal.InvalidOperation):
            pass


def cvtInt(variant):
    return int(variant)


def cvtLong(variant):  # only important in old versions where long and int differ
    return int(variant)


def cvtBuffer(variant):
    return bytes(variant)


def cvtUnicode(variant):
    return str(variant)


def identity(x):
    return x


def cvtUnusual(variant):
    if verbose > 1:
        sys.stderr.write("Conversion called for Unusual data=%s\n" % repr(variant))
    if isinstance(variant, DateTime):  # COMdate or System.Date
        from .adodbapi import (  # this will only be called when adodbapi is in use, and very rarely
            dateconverter,
        )

        return dateconverter.DateObjectFromCOMDate(variant)
    return variant  # cannot find conversion function -- just give the data to the user


def convert_to_python(variant, func):  # convert DB value into Python value
    if isinstance(variant, NullTypes):  # IronPython Null or None
        return None
    return func(variant)  # call the appropriate conversion function


class MultiMap(dict):  # builds a dictionary from {(sequence,of,keys) : function}
    """A dictionary of ado.type : function -- but you can set multiple items by passing a sequence of keys"""

    # useful for defining conversion functions for groups of similar data types.
    def __init__(self, aDict):
        for k, v in list(aDict.items()):
            self[k] = v  # we must call __setitem__

    def __setitem__(self, adoType, cvtFn):
        "set a single item, or a whole sequence of items"
        try:  # user passed us a sequence, set them individually
            for type in adoType:
                dict.__setitem__(self, type, cvtFn)
        except TypeError:  # a single value fails attempt to iterate
            dict.__setitem__(self, adoType, cvtFn)


# initialize variantConversions dictionary used to convert SQL to Python
# this is the dictionary of default conversion functions, built by the class above.
# this becomes a class attribute for the Connection, and that attribute is used
# to build the list of column conversion functions for the Cursor
variantConversions = MultiMap(
    {
        adoDateTimeTypes: variantConvertDate,
        adoApproximateNumericTypes: cvtFloat,
        adoExactNumericTypes: cvtDecimal,  # use to force decimal rather than unicode
        adoLongTypes: cvtLong,
        adoIntegerTypes: cvtInt,
        adoRowIdTypes: cvtInt,
        adoStringTypes: identity,
        adoBinaryTypes: cvtBuffer,
        adoRemainingTypes: cvtUnusual,
    }
)

# # # # # classes to emulate the result of cursor.fetchxxx() as a sequence of sequences # # # # #
# "an ENUM of how my low level records are laid out"
RS_WIN_32, RS_ARRAY, RS_REMOTE = list(range(1, 4))


class SQLrow(object):  # a single database row
    # class to emulate a sequence, so that a column may be retrieved by either number or name
    def __init__(self, rows, index):  # "rows" is an _SQLrows object, index is which row
        self.rows = rows  # parent 'fetch' container object
        self.index = index  # my row number within parent

    def __getattr__(self, name):  # used for row.columnName type of value access
        try:
            return self._getValue(self.rows.columnNames[name.lower()])
        except KeyError:
            raise AttributeError('Unknown column name "{}"'.format(name))

    def _getValue(self, key):  # key must be an integer
        if (
            self.rows.recordset_format == RS_ARRAY
        ):  # retrieve from two-dimensional array
            v = self.rows.ado_results[key, self.index]
        elif self.rows.recordset_format == RS_REMOTE:
            v = self.rows.ado_results[self.index][key]
        else:  # pywin32 - retrieve from tuple of tuples
            v = self.rows.ado_results[key][self.index]
        if self.rows.converters is NotImplemented:
            return v
        return convert_to_python(v, self.rows.converters[key])

    def __len__(self):
        return self.rows.numberOfColumns

    def __getitem__(self, key):  # used for row[key] type of value access
        if isinstance(key, int):  # normal row[1] designation
            try:
                return self._getValue(key)
            except IndexError:
                raise
        if isinstance(key, slice):
            indices = key.indices(self.rows.numberOfColumns)
            vl = [self._getValue(i) for i in range(*indices)]
            return tuple(vl)
        try:
            return self._getValue(
                self.rows.columnNames[key.lower()]
            )  # extension row[columnName] designation
        except (KeyError, TypeError):
            er, st, tr = sys.exc_info()
            raise er(
                'No such key as "%s" in %s' % (repr(key), self.__repr__())
            ).with_traceback(tr)

    def __iter__(self):
        return iter(self.__next__())

    def __next__(self):
        for n in range(self.rows.numberOfColumns):
            yield self._getValue(n)

    def __repr__(self):  # create a human readable representation
        taglist = sorted(list(self.rows.columnNames.items()), key=lambda x: x[1])
        s = "<SQLrow={"
        for name, i in taglist:
            s += name + ":" + repr(self._getValue(i)) + ", "
        return s[:-2] + "}>"

    def __str__(self):  # create a pretty human readable representation
        return str(
            tuple(str(self._getValue(i)) for i in range(self.rows.numberOfColumns))
        )

    # TO-DO implement pickling an SQLrow directly
    # def __getstate__(self): return self.__dict__
    # def __setstate__(self, d): self.__dict__.update(d)
    # which basically tell pickle to treat your class just like a normal one,
    # taking self.__dict__ as representing the whole of the instance state,
    #  despite the existence of the __getattr__.
    # # # #


class SQLrows(object):
    # class to emulate a sequence for multiple rows using a container object
    def __init__(self, ado_results, numberOfRows, cursor):
        self.ado_results = ado_results  # raw result of SQL get
        try:
            self.recordset_format = cursor.recordset_format
            self.numberOfColumns = cursor.numberOfColumns
            self.converters = cursor.converters
            self.columnNames = cursor.columnNames
        except AttributeError:
            self.recordset_format = RS_ARRAY
            self.numberOfColumns = 0
            self.converters = []
            self.columnNames = {}
        self.numberOfRows = numberOfRows

    def __len__(self):
        return self.numberOfRows

    def __getitem__(self, item):  # used for row or row,column access
        if not self.ado_results:
            return []
        if isinstance(item, slice):  # will return a list of row objects
            indices = item.indices(self.numberOfRows)
            return [SQLrow(self, k) for k in range(*indices)]
        elif isinstance(item, tuple) and len(item) == 2:
            # d = some_rowsObject[i,j] will return a datum from a two-dimension address
            i, j = item
            if not isinstance(j, int):
                try:
                    j = self.columnNames[j.lower()]  # convert named column to numeric
                except KeyError:
                    raise KeyError('adodbapi: no such column name as "%s"' % repr(j))
            if self.recordset_format == RS_ARRAY:  # retrieve from two-dimensional array
                v = self.ado_results[j, i]
            elif self.recordset_format == RS_REMOTE:
                v = self.ado_results[i][j]
            else:  # pywin32 - retrieve from tuple of tuples
                v = self.ado_results[j][i]
            if self.converters is NotImplemented:
                return v
            return convert_to_python(v, self.converters[j])
        else:
            row = SQLrow(self, item)  # new row descriptor
            return row

    def __iter__(self):
        return iter(self.__next__())

    def __next__(self):
        for n in range(self.numberOfRows):
            row = SQLrow(self, n)
            yield row
            # # # # #

    # # # # # functions to re-format SQL requests to other paramstyle requirements # # # # # # # # # #


def changeNamedToQmark(
    op,
):  # convert from 'named' paramstyle to ADO required '?'mark parameters
    outOp = ""
    outparms = []
    chunks = op.split(
        "'"
    )  # quote all literals -- odd numbered list results are literals.
    inQuotes = False
    for chunk in chunks:
        if inQuotes:  # this is inside a quote
            if chunk == "":  # double apostrophe to quote one apostrophe
                outOp = outOp[:-1]  # so take one away
            else:
                outOp += "'" + chunk + "'"  # else pass the quoted string as is.
        else:  # is SQL code -- look for a :namedParameter
            while chunk:  # some SQL string remains
                sp = chunk.split(":", 1)
                outOp += sp[0]  # concat the part up to the :
                s = ""
                try:
                    chunk = sp[1]
                except IndexError:
                    chunk = None
                if chunk:  # there was a parameter - parse it out
                    i = 0
                    c = chunk[0]
                    while c.isalnum() or c == "_":
                        i += 1
                        try:
                            c = chunk[i]
                        except IndexError:
                            break
                    s = chunk[:i]
                    chunk = chunk[i:]
                if s:
                    outparms.append(s)  # list the parameters in order
                    outOp += "?"  # put in the Qmark
        inQuotes = not inQuotes
    return outOp, outparms


def changeFormatToQmark(
    op,
):  # convert from 'format' paramstyle to ADO required '?'mark parameters
    outOp = ""
    outparams = []
    chunks = op.split(
        "'"
    )  # quote all literals -- odd numbered list results are literals.
    inQuotes = False
    for chunk in chunks:
        if inQuotes:
            if (
                outOp != "" and chunk == ""
            ):  # he used a double apostrophe to quote one apostrophe
                outOp = outOp[:-1]  # so take one away
            else:
                outOp += "'" + chunk + "'"  # else pass the quoted string as is.
        else:  # is SQL code -- look for a %s parameter
            if "%(" in chunk:  # ugh! pyformat!
                while chunk:  # some SQL string remains
                    sp = chunk.split("%(", 1)
                    outOp += sp[0]  # concat the part up to the %
                    if len(sp) > 1:
                        try:
                            s, chunk = sp[1].split(")s", 1)  # find the ')s'
                        except ValueError:
                            raise ProgrammingError(
                                'Pyformat SQL has incorrect format near "%s"' % chunk
                            )
                        outparams.append(s)
                        outOp += "?"  # put in the Qmark
                    else:
                        chunk = None
            else:  # proper '%s' format
                sp = chunk.split("%s")  # make each %s
                outOp += "?".join(sp)  # into ?
        inQuotes = not inQuotes  # every other chunk is a quoted string
    return outOp, outparams
