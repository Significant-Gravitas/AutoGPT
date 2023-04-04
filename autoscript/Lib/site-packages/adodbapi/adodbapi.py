"""adodbapi - A python DB API 2.0 (PEP 249) interface to Microsoft ADO

Copyright (C) 2002 Henrik Ekelund, versions 2.1 and later by Vernon Cole
* http://sourceforge.net/projects/pywin32
* https://github.com/mhammond/pywin32
* http://sourceforge.net/projects/adodbapi

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    django adaptations and refactoring by Adam Vandenberg

DB-API 2.0 specification: http://www.python.org/dev/peps/pep-0249/

This module source should run correctly in CPython versions 2.7 and later,
or IronPython version 2.7 and later,
or, after running through 2to3.py, CPython 3.4 or later.
"""

__version__ = "2.6.2.0"
version = "adodbapi v" + __version__

import copy
import decimal
import os
import sys
import weakref

from . import ado_consts as adc, apibase as api, process_connect_string

try:
    verbose = int(os.environ["ADODBAPI_VERBOSE"])
except:
    verbose = False
if verbose:
    print(version)

# --- define objects to smooth out IronPython <-> CPython differences
onWin32 = False  # assume the worst
if api.onIronPython:
    from clr import Reference
    from System import (
        Activator,
        Array,
        Byte,
        DateTime,
        DBNull,
        Decimal as SystemDecimal,
        Type,
    )

    def Dispatch(dispatch):
        type = Type.GetTypeFromProgID(dispatch)
        return Activator.CreateInstance(type)

    def getIndexedValue(obj, index):
        return obj.Item[index]

else:  # try pywin32
    try:
        import pythoncom
        import pywintypes
        import win32com.client

        onWin32 = True

        def Dispatch(dispatch):
            return win32com.client.Dispatch(dispatch)

    except ImportError:
        import warnings

        warnings.warn(
            "pywin32 package (or IronPython) required for adodbapi.", ImportWarning
        )

    def getIndexedValue(obj, index):
        return obj(index)


from collections.abc import Mapping

# --- define objects to smooth out Python3000 <-> Python 2.x differences
unicodeType = str
longType = int
StringTypes = str
maxint = sys.maxsize


# -----------------  The .connect method -----------------
def make_COM_connecter():
    try:
        if onWin32:
            pythoncom.CoInitialize()  # v2.1 Paj
        c = Dispatch("ADODB.Connection")  # connect _after_ CoIninialize v2.1.1 adamvan
    except:
        raise api.InterfaceError(
            "Windows COM Error: Dispatch('ADODB.Connection') failed."
        )
    return c


def connect(*args, **kwargs):  # --> a db-api connection object
    """Connect to a database.

    call using:
    :connection_string -- An ADODB formatted connection string, see:
         * http://www.connectionstrings.com
         * http://www.asp101.com/articles/john/connstring/default.asp
    :timeout -- A command timeout value, in seconds (default 30 seconds)
    """
    co = Connection()  # make an empty connection object

    kwargs = process_connect_string.process(args, kwargs, True)

    try:  # connect to the database, using the connection information in kwargs
        co.connect(kwargs)
        return co
    except Exception as e:
        message = 'Error opening connection to "%s"' % co.connection_string
        raise api.OperationalError(e, message)


# so you could use something like:
#   myConnection.paramstyle = 'named'
# The programmer may also change the default.
#   For example, if I were using django, I would say:
#     import adodbapi as Database
#     Database.adodbapi.paramstyle = 'format'

# ------- other module level defaults --------
defaultIsolationLevel = adc.adXactReadCommitted
#  Set defaultIsolationLevel on module level before creating the connection.
#   For example:
#   import adodbapi, ado_consts
#   adodbapi.adodbapi.defaultIsolationLevel=ado_consts.adXactBrowse"
#
#  Set defaultCursorLocation on module level before creating the connection.
# It may be one of the "adUse..." consts.
defaultCursorLocation = adc.adUseClient  # changed from adUseServer as of v 2.3.0

dateconverter = api.pythonDateTimeConverter()  # default


def format_parameters(ADOparameters, show_value=False):
    """Format a collection of ADO Command Parameters.

    Used by error reporting in _execute_command.
    """
    try:
        if show_value:
            desc = [
                'Name: %s, Dir.: %s, Type: %s, Size: %s, Value: "%s", Precision: %s, NumericScale: %s'
                % (
                    p.Name,
                    adc.directions[p.Direction],
                    adc.adTypeNames.get(p.Type, str(p.Type) + " (unknown type)"),
                    p.Size,
                    p.Value,
                    p.Precision,
                    p.NumericScale,
                )
                for p in ADOparameters
            ]
        else:
            desc = [
                "Name: %s, Dir.: %s, Type: %s, Size: %s, Precision: %s, NumericScale: %s"
                % (
                    p.Name,
                    adc.directions[p.Direction],
                    adc.adTypeNames.get(p.Type, str(p.Type) + " (unknown type)"),
                    p.Size,
                    p.Precision,
                    p.NumericScale,
                )
                for p in ADOparameters
            ]
        return "[" + "\n".join(desc) + "]"
    except:
        return "[]"


def _configure_parameter(p, value, adotype, settings_known):
    """Configure the given ADO Parameter 'p' with the Python 'value'."""

    if adotype in api.adoBinaryTypes:
        p.Size = len(value)
        p.AppendChunk(value)

    elif isinstance(value, StringTypes):  # v2.1 Jevon
        L = len(value)
        if adotype in api.adoStringTypes:  # v2.2.1 Cole
            if settings_known:
                L = min(L, p.Size)  # v2.1 Cole limit data to defined size
            p.Value = value[:L]  # v2.1 Jevon & v2.1 Cole
        else:
            p.Value = value  # dont limit if db column is numeric
        if L > 0:  # v2.1 Cole something does not like p.Size as Zero
            p.Size = L  # v2.1 Jevon

    elif isinstance(value, decimal.Decimal):
        if api.onIronPython:
            s = str(value)
            p.Value = s
            p.Size = len(s)
        else:
            p.Value = value
        exponent = value.as_tuple()[2]
        digit_count = len(value.as_tuple()[1])
        p.Precision = digit_count
        if exponent == 0:
            p.NumericScale = 0
        elif exponent < 0:
            p.NumericScale = -exponent
            if p.Precision < p.NumericScale:
                p.Precision = p.NumericScale
        else:  # exponent > 0:
            p.NumericScale = 0
            p.Precision = digit_count + exponent

    elif type(value) in dateconverter.types:
        if settings_known and adotype in api.adoDateTimeTypes:
            p.Value = dateconverter.COMDate(value)
        else:  # probably a string
            # provide the date as a string in the format 'YYYY-MM-dd'
            s = dateconverter.DateObjectToIsoFormatString(value)
            p.Value = s
            p.Size = len(s)

    elif api.onIronPython and isinstance(value, longType):  # Iron Python Long
        s = str(value)  # feature workaround for IPy 2.0
        p.Value = s

    elif adotype == adc.adEmpty:  # ADO will not let you specify a null column
        p.Type = (
            adc.adInteger
        )  # so we will fake it to be an integer (just to have something)
        p.Value = None  # and pass in a Null *value*

        # For any other type, set the value and let pythoncom do the right thing.
    else:
        p.Value = value


# # # # # ----- the Class that defines a connection ----- # # # # #
class Connection(object):
    # include connection attributes as class attributes required by api definition.
    Warning = api.Warning
    Error = api.Error
    InterfaceError = api.InterfaceError
    DataError = api.DataError
    DatabaseError = api.DatabaseError
    OperationalError = api.OperationalError
    IntegrityError = api.IntegrityError
    InternalError = api.InternalError
    NotSupportedError = api.NotSupportedError
    ProgrammingError = api.ProgrammingError
    FetchFailedError = api.FetchFailedError  # (special for django)
    # ...class attributes... (can be overridden by instance attributes)
    verbose = api.verbose

    @property
    def dbapi(self):  # a proposed db-api version 3 extension.
        "Return a reference to the DBAPI module for this Connection."
        return api

    def __init__(self):  # now define the instance attributes
        self.connector = None
        self.paramstyle = api.paramstyle
        self.supportsTransactions = False
        self.connection_string = ""
        self.cursors = weakref.WeakValueDictionary()
        self.dbms_name = ""
        self.dbms_version = ""
        self.errorhandler = None  # use the standard error handler for this instance
        self.transaction_level = 0  # 0 == Not in a transaction, at the top level
        self._autocommit = False

    def connect(self, kwargs, connection_maker=make_COM_connecter):
        if verbose > 9:
            print("kwargs=", repr(kwargs))
        try:
            self.connection_string = (
                kwargs["connection_string"] % kwargs
            )  # insert keyword arguments
        except Exception as e:
            self._raiseConnectionError(
                KeyError, "Python string format error in connection string->"
            )
        self.timeout = kwargs.get("timeout", 30)
        self.mode = kwargs.get("mode", adc.adModeUnknown)
        self.kwargs = kwargs
        if verbose:
            print('%s attempting: "%s"' % (version, self.connection_string))
        self.connector = connection_maker()
        self.connector.ConnectionTimeout = self.timeout
        self.connector.ConnectionString = self.connection_string
        self.connector.Mode = self.mode

        try:
            self.connector.Open()  # Open the ADO connection
        except api.Error:
            self._raiseConnectionError(
                api.DatabaseError,
                "ADO error trying to Open=%s" % self.connection_string,
            )

        try:  # Stefan Fuchs; support WINCCOLEDBProvider
            if getIndexedValue(self.connector.Properties, "Transaction DDL").Value != 0:
                self.supportsTransactions = True
        except pywintypes.com_error:
            pass  # Stefan Fuchs
        self.dbms_name = getIndexedValue(self.connector.Properties, "DBMS Name").Value
        try:  # Stefan Fuchs
            self.dbms_version = getIndexedValue(
                self.connector.Properties, "DBMS Version"
            ).Value
        except pywintypes.com_error:
            pass  # Stefan Fuchs
        self.connector.CursorLocation = defaultCursorLocation  # v2.1 Rose
        if self.supportsTransactions:
            self.connector.IsolationLevel = defaultIsolationLevel
            self._autocommit = bool(kwargs.get("autocommit", False))
            if not self._autocommit:
                self.transaction_level = (
                    self.connector.BeginTrans()
                )  # Disables autocommit & inits transaction_level
        else:
            self._autocommit = True
        if "paramstyle" in kwargs:
            self.paramstyle = kwargs["paramstyle"]  # let setattr do the error checking
        self.messages = []
        if verbose:
            print("adodbapi New connection at %X" % id(self))

    def _raiseConnectionError(self, errorclass, errorvalue):
        eh = self.errorhandler
        if eh is None:
            eh = api.standardErrorHandler
        eh(self, None, errorclass, errorvalue)

    def _closeAdoConnection(self):  # all v2.1 Rose
        """close the underlying ADO Connection object,
        rolling it back first if it supports transactions."""
        if self.connector is None:
            return
        if not self._autocommit:
            if self.transaction_level:
                try:
                    self.connector.RollbackTrans()
                except:
                    pass
        self.connector.Close()
        if verbose:
            print("adodbapi Closed connection at %X" % id(self))

    def close(self):
        """Close the connection now (rather than whenever __del__ is called).

        The connection will be unusable from this point forward;
        an Error (or subclass) exception will be raised if any operation is attempted with the connection.
        The same applies to all cursor objects trying to use the connection.
        """
        for crsr in list(self.cursors.values())[
            :
        ]:  # copy the list, then close each one
            crsr.close(dont_tell_me=True)  # close without back-link clearing
        self.messages = []
        try:
            self._closeAdoConnection()  # v2.1 Rose
        except Exception as e:
            self._raiseConnectionError(sys.exc_info()[0], sys.exc_info()[1])

        self.connector = None  # v2.4.2.2 fix subtle timeout bug
        # per M.Hammond: "I expect the benefits of uninitializing are probably fairly small,
        #    so never uninitializing will probably not cause any problems."

    def commit(self):
        """Commit any pending transaction to the database.

        Note that if the database supports an auto-commit feature,
        this must be initially off. An interface method may be provided to turn it back on.
        Database modules that do not support transactions should implement this method with void functionality.
        """
        self.messages = []
        if not self.supportsTransactions:
            return

        try:
            self.transaction_level = self.connector.CommitTrans()
            if verbose > 1:
                print("commit done on connection at %X" % id(self))
            if not (
                self._autocommit
                or (self.connector.Attributes & adc.adXactAbortRetaining)
            ):
                # If attributes has adXactCommitRetaining it performs retaining commits that is,
                # calling CommitTrans automatically starts a new transaction. Not all providers support this.
                # If not, we will have to start a new transaction by this command:
                self.transaction_level = self.connector.BeginTrans()
        except Exception as e:
            self._raiseConnectionError(api.ProgrammingError, e)

    def _rollback(self):
        """In case a database does provide transactions this method causes the the database to roll back to
        the start of any pending transaction. Closing a connection without committing the changes first will
        cause an implicit rollback to be performed.

        If the database does not support the functionality required by the method, the interface should
        throw an exception in case the method is used.
        The preferred approach is to not implement the method and thus have Python generate
        an AttributeError in case the method is requested. This allows the programmer to check for database
        capabilities using the standard hasattr() function.

        For some dynamically configured interfaces it may not be appropriate to require dynamically making
        the method available. These interfaces should then raise a NotSupportedError to indicate the
        non-ability to perform the roll back when the method is invoked.
        """
        self.messages = []
        if (
            self.transaction_level
        ):  # trying to roll back with no open transaction causes an error
            try:
                self.transaction_level = self.connector.RollbackTrans()
                if verbose > 1:
                    print("rollback done on connection at %X" % id(self))
                if not self._autocommit and not (
                    self.connector.Attributes & adc.adXactAbortRetaining
                ):
                    # If attributes has adXactAbortRetaining it performs retaining aborts that is,
                    # calling RollbackTrans automatically starts a new transaction. Not all providers support this.
                    # If not, we will have to start a new transaction by this command:
                    if (
                        not self.transaction_level
                    ):  # if self.transaction_level == 0 or self.transaction_level is None:
                        self.transaction_level = self.connector.BeginTrans()
            except Exception as e:
                self._raiseConnectionError(api.ProgrammingError, e)

    def __setattr__(self, name, value):
        if name == "autocommit":  # extension: allow user to turn autocommit on or off
            if self.supportsTransactions:
                object.__setattr__(self, "_autocommit", bool(value))
                try:
                    self._rollback()  # must clear any outstanding transactions
                except:
                    pass
            return
        elif name == "paramstyle":
            if value not in api.accepted_paramstyles:
                self._raiseConnectionError(
                    api.NotSupportedError,
                    'paramstyle="%s" not in:%s'
                    % (value, repr(api.accepted_paramstyles)),
                )
        elif name == "variantConversions":
            value = copy.copy(
                value
            )  # make a new copy -- no changes in the default, please
        object.__setattr__(self, name, value)

    def __getattr__(self, item):
        if (
            item == "rollback"
        ):  # the rollback method only appears if the database supports transactions
            if self.supportsTransactions:
                return (
                    self._rollback
                )  # return the rollback method so the caller can execute it.
            else:
                raise AttributeError("this data provider does not support Rollback")
        elif item == "autocommit":
            return self._autocommit
        else:
            raise AttributeError(
                'no such attribute in ADO connection object as="%s"' % item
            )

    def cursor(self):
        "Return a new Cursor Object using the connection."
        self.messages = []
        c = Cursor(self)
        return c

    def _i_am_here(self, crsr):
        "message from a new cursor proclaiming its existence"
        oid = id(crsr)
        self.cursors[oid] = crsr

    def _i_am_closing(self, crsr):
        "message from a cursor giving connection a chance to clean up"
        try:
            del self.cursors[id(crsr)]
        except:
            pass

    def printADOerrors(self):
        j = self.connector.Errors.Count
        if j:
            print("ADO Errors:(%i)" % j)
        for e in self.connector.Errors:
            print("Description: %s" % e.Description)
            print("Error: %s %s " % (e.Number, adc.adoErrors.get(e.Number, "unknown")))
            if e.Number == adc.ado_error_TIMEOUT:
                print(
                    "Timeout Error: Try using adodbpi.connect(constr,timeout=Nseconds)"
                )
            print("Source: %s" % e.Source)
            print("NativeError: %s" % e.NativeError)
            print("SQL State: %s" % e.SQLState)

    def _suggest_error_class(self):
        """Introspect the current ADO Errors and determine an appropriate error class.

        Error.SQLState is a SQL-defined error condition, per the SQL specification:
        http://www.contrib.andrew.cmu.edu/~shadow/sql/sql1992.txt

        The 23000 class of errors are integrity errors.
        Error 40002 is a transactional integrity error.
        """
        if self.connector is not None:
            for e in self.connector.Errors:
                state = str(e.SQLState)
                if state.startswith("23") or state == "40002":
                    return api.IntegrityError
        return api.DatabaseError

    def __del__(self):
        try:
            self._closeAdoConnection()  # v2.1 Rose
        except:
            pass
        self.connector = None

    def __enter__(self):  # Connections are context managers
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._rollback()  # automatic rollback on errors
        else:
            self.commit()

    def get_table_names(self):
        schema = self.connector.OpenSchema(20)  # constant = adSchemaTables

        tables = []
        while not schema.EOF:
            name = getIndexedValue(schema.Fields, "TABLE_NAME").Value
            tables.append(name)
            schema.MoveNext()
        del schema
        return tables


# # # # # ----- the Class that defines a cursor ----- # # # # #
class Cursor(object):
    ## ** api required attributes:
    ## description...
    ##    This read-only attribute is a sequence of 7-item sequences.
    ##    Each of these sequences contains information describing one result column:
    ##        (name, type_code, display_size, internal_size, precision, scale, null_ok).
    ##    This attribute will be None for operations that do not return rows or if the
    ##    cursor has not had an operation invoked via the executeXXX() method yet.
    ##    The type_code can be interpreted by comparing it to the Type Objects specified in the section below.
    ## rowcount...
    ##    This read-only attribute specifies the number of rows that the last executeXXX() produced
    ##    (for DQL statements like select) or affected (for DML statements like update or insert).
    ##    The attribute is -1 in case no executeXXX() has been performed on the cursor or
    ##    the rowcount of the last operation is not determinable by the interface.[7]
    ## arraysize...
    ##    This read/write attribute specifies the number of rows to fetch at a time with fetchmany().
    ##    It defaults to 1 meaning to fetch a single row at a time.
    ##    Implementations must observe this value with respect to the fetchmany() method,
    ##    but are free to interact with the database a single row at a time.
    ##    It may also be used in the implementation of executemany().
    ## ** extension attributes:
    ## paramstyle...
    ##   allows the programmer to override the connection's default paramstyle
    ## errorhandler...
    ##   allows the programmer to override the connection's default error handler

    def __init__(self, connection):
        self.command = None
        self._ado_prepared = False
        self.messages = []
        self.connection = connection
        self.paramstyle = connection.paramstyle  # used for overriding the paramstyle
        self._parameter_names = []
        self.recordset_is_remote = False
        self.rs = None  # the ADO recordset for this cursor
        self.converters = []  # conversion function for each column
        self.columnNames = {}  # names of columns {lowercase name : number,...}
        self.numberOfColumns = 0
        self._description = None
        self.rowcount = -1
        self.errorhandler = connection.errorhandler
        self.arraysize = 1
        connection._i_am_here(self)
        if verbose:
            print(
                "%s New cursor at %X on conn %X"
                % (version, id(self), id(self.connection))
            )

    def __iter__(self):  # [2.1 Zamarev]
        return iter(self.fetchone, None)  # [2.1 Zamarev]

    def prepare(self, operation):
        self.command = operation
        self._description = None
        self._ado_prepared = "setup"

    def __next__(self):
        r = self.fetchone()
        if r:
            return r
        raise StopIteration

    def __enter__(self):
        "Allow database cursors to be used with context managers."
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        "Allow database cursors to be used with context managers."
        self.close()

    def _raiseCursorError(self, errorclass, errorvalue):
        eh = self.errorhandler
        if eh is None:
            eh = api.standardErrorHandler
        eh(self.connection, self, errorclass, errorvalue)

    def build_column_info(self, recordset):
        self.converters = []  # convertion function for each column
        self.columnNames = {}  # names of columns {lowercase name : number,...}
        self._description = None

        # if EOF and BOF are true at the same time, there are no records in the recordset
        if (recordset is None) or (recordset.State == adc.adStateClosed):
            self.rs = None
            self.numberOfColumns = 0
            return
        self.rs = recordset  # v2.1.1 bkline
        self.recordset_format = api.RS_ARRAY if api.onIronPython else api.RS_WIN_32
        self.numberOfColumns = recordset.Fields.Count
        try:
            varCon = self.connection.variantConversions
        except AttributeError:
            varCon = api.variantConversions
        for i in range(self.numberOfColumns):
            f = getIndexedValue(self.rs.Fields, i)
            try:
                self.converters.append(
                    varCon[f.Type]
                )  # conversion function for this column
            except KeyError:
                self._raiseCursorError(
                    api.InternalError, "Data column of Unknown ADO type=%s" % f.Type
                )
            self.columnNames[f.Name.lower()] = i  # columnNames lookup

    def _makeDescriptionFromRS(self):
        # Abort if closed or no recordset.
        if self.rs is None:
            self._description = None
            return
        desc = []
        for i in range(self.numberOfColumns):
            f = getIndexedValue(self.rs.Fields, i)
            if self.rs.EOF or self.rs.BOF:
                display_size = None
            else:
                display_size = (
                    f.ActualSize
                )  # TODO: Is this the correct defintion according to the DB API 2 Spec ?
            null_ok = bool(f.Attributes & adc.adFldMayBeNull)  # v2.1 Cole
            desc.append(
                (
                    f.Name,
                    f.Type,
                    display_size,
                    f.DefinedSize,
                    f.Precision,
                    f.NumericScale,
                    null_ok,
                )
            )
        self._description = desc

    def get_description(self):
        if not self._description:
            self._makeDescriptionFromRS()
        return self._description

    def __getattr__(self, item):
        if item == "description":
            return self.get_description()
        object.__getattribute__(
            self, item
        )  # may get here on Remote attribute calls for existing attributes

    def format_description(self, d):
        """Format db_api description tuple for printing."""
        if self.description is None:
            self._makeDescriptionFromRS()
        if isinstance(d, int):
            d = self.description[d]
        desc = (
            "Name= %s, Type= %s, DispSize= %s, IntSize= %s, Precision= %s, Scale= %s NullOK=%s"
            % (
                d[0],
                adc.adTypeNames.get(d[1], str(d[1]) + " (unknown type)"),
                d[2],
                d[3],
                d[4],
                d[5],
                d[6],
            )
        )
        return desc

    def close(self, dont_tell_me=False):
        """Close the cursor now (rather than whenever __del__ is called).
        The cursor will be unusable from this point forward; an Error (or subclass)
        exception will be raised if any operation is attempted with the cursor.
        """
        if self.connection is None:
            return
        self.messages = []
        if (
            self.rs and self.rs.State != adc.adStateClosed
        ):  # rs exists and is open      #v2.1 Rose
            self.rs.Close()  # v2.1 Rose
            self.rs = None  # let go of the recordset so ADO will let it be disposed #v2.1 Rose
        if not dont_tell_me:
            self.connection._i_am_closing(
                self
            )  # take me off the connection's cursors list
        self.connection = (
            None  # this will make all future method calls on me throw an exception
        )
        if verbose:
            print("adodbapi Closed cursor at %X" % id(self))

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def _new_command(self, command_type=adc.adCmdText):
        self.cmd = None
        self.messages = []

        if self.connection is None:
            self._raiseCursorError(api.InterfaceError, None)
            return
        try:
            self.cmd = Dispatch("ADODB.Command")
            self.cmd.ActiveConnection = self.connection.connector
            self.cmd.CommandTimeout = self.connection.timeout
            self.cmd.CommandType = command_type
            self.cmd.CommandText = self.commandText
            self.cmd.Prepared = bool(self._ado_prepared)
        except:
            self._raiseCursorError(
                api.DatabaseError,
                'Error creating new ADODB.Command object for "%s"'
                % repr(self.commandText),
            )

    def _execute_command(self):
        # Stored procedures may have an integer return value
        self.return_value = None
        recordset = None
        count = -1  # default value
        if verbose:
            print('Executing command="%s"' % self.commandText)
        try:
            # ----- the actual SQL is executed here ---
            if api.onIronPython:
                ra = Reference[int]()
                recordset = self.cmd.Execute(ra)
                count = ra.Value
            else:  # pywin32
                recordset, count = self.cmd.Execute()
            # ----- ------------------------------- ---
        except Exception as e:
            _message = ""
            if hasattr(e, "args"):
                _message += str(e.args) + "\n"
            _message += "Command:\n%s\nParameters:\n%s" % (
                self.commandText,
                format_parameters(self.cmd.Parameters, True),
            )
            klass = self.connection._suggest_error_class()
            self._raiseCursorError(klass, _message)
        try:
            self.rowcount = recordset.RecordCount
        except:
            self.rowcount = count
        self.build_column_info(recordset)

        # The ADO documentation hints that obtaining the recordcount may be timeconsuming
        #   "If the Recordset object does not support approximate positioning, this property
        #    may be a significant drain on resources # [ekelund]
        # Therefore, COM will not return rowcount for server-side cursors. [Cole]
        # Client-side cursors (the default since v2.8) will force a static
        # cursor, and rowcount will then be set accurately [Cole]

    def get_rowcount(self):
        return self.rowcount

    def get_returned_parameters(self):
        """with some providers, returned parameters and the .return_value are not available until
        after the last recordset has been read.  In that case, you must coll nextset() until it
        returns None, then call this method to get your returned information."""

        retLst = (
            []
        )  # store procedures may return altered parameters, including an added "return value" item
        for p in tuple(self.cmd.Parameters):
            if verbose > 2:
                print(
                    'Returned=Name: %s, Dir.: %s, Type: %s, Size: %s, Value: "%s",'
                    " Precision: %s, NumericScale: %s"
                    % (
                        p.Name,
                        adc.directions[p.Direction],
                        adc.adTypeNames.get(p.Type, str(p.Type) + " (unknown type)"),
                        p.Size,
                        p.Value,
                        p.Precision,
                        p.NumericScale,
                    )
                )
            pyObject = api.convert_to_python(p.Value, api.variantConversions[p.Type])
            if p.Direction == adc.adParamReturnValue:
                self.returnValue = (
                    pyObject  # also load the undocumented attribute (Vernon's Error!)
                )
                self.return_value = pyObject
            else:
                retLst.append(pyObject)
        return retLst  # return the parameter list to the caller

    def callproc(self, procname, parameters=None):
        """Call a stored database procedure with the given name.
        The sequence of parameters must contain one entry for each
        argument that the sproc expects. The result of the
        call is returned as modified copy of the input
        sequence.  Input parameters are left untouched, output and
        input/output parameters replaced with possibly new values.

        The sproc may also provide a result set as output,
        which is available through the standard .fetch*() methods.
        Extension: A "return_value" property may be set on the
        cursor if the sproc defines an integer return value.
        """
        self._parameter_names = []
        self.commandText = procname
        self._new_command(command_type=adc.adCmdStoredProc)
        self._buildADOparameterList(parameters, sproc=True)
        if verbose > 2:
            print(
                "Calling Stored Proc with Params=",
                format_parameters(self.cmd.Parameters, True),
            )
        self._execute_command()
        return self.get_returned_parameters()

    def _reformat_operation(self, operation, parameters):
        if self.paramstyle in ("format", "pyformat"):  # convert %s to ?
            operation, self._parameter_names = api.changeFormatToQmark(operation)
        elif self.paramstyle == "named" or (
            self.paramstyle == "dynamic" and isinstance(parameters, Mapping)
        ):
            operation, self._parameter_names = api.changeNamedToQmark(
                operation
            )  # convert :name to ?
        return operation

    def _buildADOparameterList(self, parameters, sproc=False):
        self.parameters = parameters
        if parameters is None:
            parameters = []

        # Note: ADO does not preserve the parameter list, even if "Prepared" is True, so we must build every time.
        parameters_known = False
        if sproc:  # needed only if we are calling a stored procedure
            try:  # attempt to use ADO's parameter list
                self.cmd.Parameters.Refresh()
                if verbose > 2:
                    print(
                        "ADO detected Params=",
                        format_parameters(self.cmd.Parameters, True),
                    )
                    print("Program Parameters=", repr(parameters))
                parameters_known = True
            except api.Error:
                if verbose:
                    print("ADO Parameter Refresh failed")
                pass
            else:
                if len(parameters) != self.cmd.Parameters.Count - 1:
                    raise api.ProgrammingError(
                        "You must supply %d parameters for this stored procedure"
                        % (self.cmd.Parameters.Count - 1)
                    )
        if sproc or parameters != []:
            i = 0
            if parameters_known:  # use ado parameter list
                if self._parameter_names:  # named parameters
                    for i, pm_name in enumerate(self._parameter_names):
                        p = getIndexedValue(self.cmd.Parameters, i)
                        try:
                            _configure_parameter(
                                p, parameters[pm_name], p.Type, parameters_known
                            )
                        except Exception as e:
                            _message = (
                                "Error Converting Parameter %s: %s, %s <- %s\n"
                                % (
                                    p.Name,
                                    adc.ado_type_name(p.Type),
                                    p.Value,
                                    repr(parameters[pm_name]),
                                )
                            )
                            self._raiseCursorError(
                                api.DataError, _message + "->" + repr(e.args)
                            )
                else:  # regular sequence of parameters
                    for value in parameters:
                        p = getIndexedValue(self.cmd.Parameters, i)
                        if (
                            p.Direction == adc.adParamReturnValue
                        ):  # this is an extra parameter added by ADO
                            i += 1  # skip the extra
                            p = getIndexedValue(self.cmd.Parameters, i)
                        try:
                            _configure_parameter(p, value, p.Type, parameters_known)
                        except Exception as e:
                            _message = (
                                "Error Converting Parameter %s: %s, %s <- %s\n"
                                % (
                                    p.Name,
                                    adc.ado_type_name(p.Type),
                                    p.Value,
                                    repr(value),
                                )
                            )
                            self._raiseCursorError(
                                api.DataError, _message + "->" + repr(e.args)
                            )
                        i += 1
            else:  # -- build own parameter list
                if (
                    self._parameter_names
                ):  # we expect a dictionary of parameters, this is the list of expected names
                    for parm_name in self._parameter_names:
                        elem = parameters[parm_name]
                        adotype = api.pyTypeToADOType(elem)
                        p = self.cmd.CreateParameter(
                            parm_name, adotype, adc.adParamInput
                        )
                        _configure_parameter(p, elem, adotype, parameters_known)
                        try:
                            self.cmd.Parameters.Append(p)
                        except Exception as e:
                            _message = "Error Building Parameter %s: %s, %s <- %s\n" % (
                                p.Name,
                                adc.ado_type_name(p.Type),
                                p.Value,
                                repr(elem),
                            )
                            self._raiseCursorError(
                                api.DataError, _message + "->" + repr(e.args)
                            )
                else:  # expecting the usual sequence of parameters
                    if sproc:
                        p = self.cmd.CreateParameter(
                            "@RETURN_VALUE", adc.adInteger, adc.adParamReturnValue
                        )
                        self.cmd.Parameters.Append(p)

                    for elem in parameters:
                        name = "p%i" % i
                        adotype = api.pyTypeToADOType(elem)
                        p = self.cmd.CreateParameter(
                            name, adotype, adc.adParamInput
                        )  # Name, Type, Direction, Size, Value
                        _configure_parameter(p, elem, adotype, parameters_known)
                        try:
                            self.cmd.Parameters.Append(p)
                        except Exception as e:
                            _message = "Error Building Parameter %s: %s, %s <- %s\n" % (
                                p.Name,
                                adc.ado_type_name(p.Type),
                                p.Value,
                                repr(elem),
                            )
                            self._raiseCursorError(
                                api.DataError, _message + "->" + repr(e.args)
                            )
                        i += 1
                if self._ado_prepared == "setup":
                    self._ado_prepared = (
                        True  # parameters will be "known" by ADO next loop
                    )

    def execute(self, operation, parameters=None):
        """Prepare and execute a database operation (query or command).

        Parameters may be provided as sequence or mapping and will be bound to variables in the operation.
        Variables are specified in a database-specific notation
        (see the module's paramstyle attribute for details). [5]
        A reference to the operation will be retained by the cursor.
        If the same operation object is passed in again, then the cursor
        can optimize its behavior. This is most effective for algorithms
        where the same operation is used, but different parameters are bound to it (many times).

        For maximum efficiency when reusing an operation, it is best to use
        the setinputsizes() method to specify the parameter types and sizes ahead of time.
        It is legal for a parameter to not match the predefined information;
        the implementation should compensate, possibly with a loss of efficiency.

        The parameters may also be specified as list of tuples to e.g. insert multiple rows in
        a single operation, but this kind of usage is depreciated: executemany() should be used instead.

        Return value is not defined.

        [5] The module will use the __getitem__ method of the parameters object to map either positions
        (integers) or names (strings) to parameter values. This allows for both sequences and mappings
        to be used as input.
        The term "bound" refers to the process of binding an input value to a database execution buffer.
        In practical terms, this means that the input value is directly used as a value in the operation.
        The client should not be required to "escape" the value so that it can be used -- the value
        should be equal to the actual database value."""
        if (
            self.command is not operation
            or self._ado_prepared == "setup"
            or not hasattr(self, "commandText")
        ):
            if self.command is not operation:
                self._ado_prepared = False
                self.command = operation
            self._parameter_names = []
            self.commandText = (
                operation
                if (self.paramstyle == "qmark" or not parameters)
                else self._reformat_operation(operation, parameters)
            )
        self._new_command()
        self._buildADOparameterList(parameters)
        if verbose > 3:
            print("Params=", format_parameters(self.cmd.Parameters, True))
        self._execute_command()

    def executemany(self, operation, seq_of_parameters):
        """Prepare a database operation (query or command)
        and then execute it against all parameter sequences or mappings found in the sequence seq_of_parameters.

            Return values are not defined.
        """
        self.messages = list()
        total_recordcount = 0

        self.prepare(operation)
        for params in seq_of_parameters:
            self.execute(self.command, params)
            if self.rowcount == -1:
                total_recordcount = -1
            if total_recordcount != -1:
                total_recordcount += self.rowcount
        self.rowcount = total_recordcount

    def _fetch(self, limit=None):
        """Fetch rows from the current recordset.

        limit -- Number of rows to fetch, or None (default) to fetch all rows.
        """
        if self.connection is None or self.rs is None:
            self._raiseCursorError(
                api.FetchFailedError, "fetch() on closed connection or empty query set"
            )
            return

        if self.rs.State == adc.adStateClosed or self.rs.BOF or self.rs.EOF:
            return list()
        if limit:  # limit number of rows retrieved
            ado_results = self.rs.GetRows(limit)
        else:  # get all rows
            ado_results = self.rs.GetRows()
        if (
            self.recordset_format == api.RS_ARRAY
        ):  # result of GetRows is a two-dimension array
            length = (
                len(ado_results) // self.numberOfColumns
            )  # length of first dimension
        else:  # pywin32
            length = len(ado_results[0])  # result of GetRows is tuples in a tuple
        fetchObject = api.SQLrows(
            ado_results, length, self
        )  # new object to hold the results of the fetch
        return fetchObject

    def fetchone(self):
        """Fetch the next row of a query result set, returning a single sequence,
        or None when no more data is available.

        An Error (or subclass) exception is raised if the previous call to executeXXX()
        did not produce any result set or no call was issued yet.
        """
        self.messages = []
        result = self._fetch(1)
        if result:  # return record (not list of records)
            return result[0]
        return None

    def fetchmany(self, size=None):
        """Fetch the next set of rows of a query result, returning a list of tuples. An empty sequence is returned when no more rows are available.

        The number of rows to fetch per call is specified by the parameter.
        If it is not given, the cursor's arraysize determines the number of rows to be fetched.
        The method should try to fetch as many rows as indicated by the size parameter.
        If this is not possible due to the specified number of rows not being available,
        fewer rows may be returned.

        An Error (or subclass) exception is raised if the previous call to executeXXX()
        did not produce any result set or no call was issued yet.

        Note there are performance considerations involved with the size parameter.
        For optimal performance, it is usually best to use the arraysize attribute.
        If the size parameter is used, then it is best for it to retain the same value from
        one fetchmany() call to the next.
        """
        self.messages = []
        if size is None:
            size = self.arraysize
        return self._fetch(size)

    def fetchall(self):
        """Fetch all (remaining) rows of a query result, returning them as a sequence of sequences (e.g. a list of tuples).

        Note that the cursor's arraysize attribute
        can affect the performance of this operation.
        An Error (or subclass) exception is raised if the previous call to executeXXX()
        did not produce any result set or no call was issued yet.
        """
        self.messages = []
        return self._fetch()

    def nextset(self):
        """Skip to the next available recordset, discarding any remaining rows from the current recordset.

        If there are no more sets, the method returns None. Otherwise, it returns a true
        value and subsequent calls to the fetch methods will return rows from the next result set.

        An Error (or subclass) exception is raised if the previous call to executeXXX()
        did not produce any result set or no call was issued yet.
        """
        self.messages = []
        if self.connection is None or self.rs is None:
            self._raiseCursorError(
                api.OperationalError,
                ("nextset() on closed connection or empty query set"),
            )
            return None

        if api.onIronPython:
            try:
                recordset = self.rs.NextRecordset()
            except TypeError:
                recordset = None
            except api.Error as exc:
                self._raiseCursorError(api.NotSupportedError, exc.args)
        else:  # pywin32
            try:  # [begin 2.1 ekelund]
                rsTuple = self.rs.NextRecordset()  #
            except pywintypes.com_error as exc:  # return appropriate error
                self._raiseCursorError(
                    api.NotSupportedError, exc.args
                )  # [end 2.1 ekelund]
            recordset = rsTuple[0]
        if recordset is None:
            return None
        self.build_column_info(recordset)
        return True

    def setinputsizes(self, sizes):
        pass

    def setoutputsize(self, size, column=None):
        pass

    def _last_query(self):  # let the programmer see what query we actually used
        try:
            if self.parameters == None:
                ret = self.commandText
            else:
                ret = "%s,parameters=%s" % (self.commandText, repr(self.parameters))
        except:
            ret = None
        return ret

    query = property(_last_query, None, None, "returns the last query executed")


if __name__ == "__main__":
    raise api.ProgrammingError(version + " cannot be run as a main program.")
