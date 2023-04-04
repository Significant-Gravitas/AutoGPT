"""adodbapi.remote - A python DB API 2.0 (PEP 249) interface to Microsoft ADO

Copyright (C) 2002 Henrik Ekelund, version 2.1 by Vernon Cole
* http://sourceforge.net/projects/pywin32
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

    django adaptations and refactoring thanks to Adam Vandenberg

DB-API 2.0 specification: http://www.python.org/dev/peps/pep-0249/

This module source should run correctly in CPython versions 2.5 and later,
or IronPython version 2.7 and later,
or, after running through 2to3.py, CPython 3.0 or later.
"""

__version__ = "2.6.0.4"
version = "adodbapi.remote v" + __version__

import array
import datetime
import os
import sys
import time

# Pyro4 is required for server and remote operation --> https://pypi.python.org/pypi/Pyro4/
try:
    import Pyro4
except ImportError:
    print('* * * Sorry, server operation requires Pyro4. Please "pip import" it.')
    exit(11)

import adodbapi
import adodbapi.apibase as api
import adodbapi.process_connect_string
from adodbapi.apibase import ProgrammingError

_BaseException = api._BaseException

sys.excepthook = Pyro4.util.excepthook
Pyro4.config.PREFER_IP_VERSION = 0  # allow system to prefer IPv6
Pyro4.config.COMMTIMEOUT = 40.0  # a bit longer than the default SQL server Gtimeout
Pyro4.config.SERIALIZER = "pickle"

try:
    verbose = int(os.environ["ADODBAPI_VERBOSE"])
except:
    verbose = False
if verbose:
    print(version)

# --- define objects to smooth out Python3 <-> Python 2.x differences
unicodeType = str  # this line will be altered by 2to3.py to '= str'
longType = int  # this line will be altered by 2to3.py to '= int'
StringTypes = str
makeByteBuffer = bytes
memoryViewType = memoryview

# -----------------------------------------------------------
# conversion functions mandated by PEP 249
Binary = makeByteBuffer  # override the function from apibase.py


def Date(year, month, day):
    return datetime.date(year, month, day)  # dateconverter.Date(year,month,day)


def Time(hour, minute, second):
    return datetime.time(hour, minute, second)  # dateconverter.Time(hour,minute,second)


def Timestamp(year, month, day, hour, minute, second):
    return datetime.datetime(year, month, day, hour, minute, second)


def DateFromTicks(ticks):
    return Date(*time.gmtime(ticks)[:3])


def TimeFromTicks(ticks):
    return Time(*time.gmtime(ticks)[3:6])


def TimestampFromTicks(ticks):
    return Timestamp(*time.gmtime(ticks)[:6])


def connect(*args, **kwargs):  # --> a remote db-api connection object
    """Create and open a remote db-api database connection object"""
    # process the argument list the programmer gave us
    kwargs = adodbapi.process_connect_string.process(args, kwargs)
    # the "proxy_xxx" keys tell us where to find the PyRO proxy server
    kwargs.setdefault(
        "pyro_connection", "PYRO:ado.connection@%(proxy_host)s:%(proxy_port)s"
    )
    if not "proxy_port" in kwargs:
        try:
            pport = os.environ["PROXY_PORT"]
        except KeyError:
            pport = 9099
        kwargs["proxy_port"] = pport
    if not "proxy_host" in kwargs or not kwargs["proxy_host"]:
        try:
            phost = os.environ["PROXY_HOST"]
        except KeyError:
            phost = "[::1]"  # '127.0.0.1'
        kwargs["proxy_host"] = phost
    ado_uri = kwargs["pyro_connection"] % kwargs
    # ask PyRO make us a remote connection object
    auto_retry = 3
    while auto_retry:
        try:
            dispatcher = Pyro4.Proxy(ado_uri)
            if "comm_timeout" in kwargs:
                dispatcher._pyroTimeout = float(kwargs["comm_timeout"])
            uri = dispatcher.make_connection()
            break
        except Pyro4.core.errors.PyroError:
            auto_retry -= 1
            if auto_retry:
                time.sleep(1)
            else:
                raise api.DatabaseError("Cannot create connection to=%s" % ado_uri)

    conn_uri = fix_uri(uri, kwargs)  # get a host connection from the proxy server
    while auto_retry:
        try:
            host_conn = Pyro4.Proxy(
                conn_uri
            )  # bring up an exclusive Pyro connection for my ADO connection
            break
        except Pyro4.core.errors.PyroError:
            auto_retry -= 1
            if auto_retry:
                time.sleep(1)
            else:
                raise api.DatabaseError(
                    "Cannot create ADO connection object using=%s" % conn_uri
                )
    if "comm_timeout" in kwargs:
        host_conn._pyroTimeout = float(kwargs["comm_timeout"])
    # make a local clone
    myConn = Connection()
    while auto_retry:
        try:
            myConn.connect(
                kwargs, host_conn
            )  # call my connect method -- hand him the host connection
            break
        except Pyro4.core.errors.PyroError:
            auto_retry -= 1
            if auto_retry:
                time.sleep(1)
            else:
                raise api.DatabaseError(
                    "Pyro error creating connection to/thru=%s" % repr(kwargs)
                )
        except _BaseException as e:
            raise api.DatabaseError(
                "Error creating remote connection to=%s, e=%s, %s"
                % (repr(kwargs), repr(e), sys.exc_info()[2])
            )
    return myConn


def fix_uri(uri, kwargs):
    """convert a generic pyro uri with '0.0.0.0' into the address we actually called"""
    u = uri.asString()
    s = u.split("[::0]")  # IPv6 generic address
    if len(s) == 1:  # did not find one
        s = u.split("0.0.0.0")  # IPv4 generic address
    if len(s) > 1:  # found a generic
        return kwargs["proxy_host"].join(s)  # fill in our address for the host
    return uri


# # # # # ----- the Class that defines a connection ----- # # # # #
class Connection(object):
    # include connection attributes required by api definition.
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
    # set up some class attributes
    paramstyle = api.paramstyle

    @property
    def dbapi(self):  # a proposed db-api version 3 extension.
        "Return a reference to the DBAPI module for this Connection."
        return api

    def __init__(self):
        self.proxy = None
        self.kwargs = {}
        self.errorhandler = None
        self.supportsTransactions = False
        self.paramstyle = api.paramstyle
        self.timeout = 30
        self.cursors = {}

    def connect(self, kwargs, connection_maker):
        self.kwargs = kwargs
        if verbose:
            print('%s attempting: "%s"' % (version, repr(kwargs)))
        self.proxy = connection_maker
        ##try:
        ret = self.proxy.connect(kwargs)  # ask the server to hook us up
        ##except ImportError, e:   # Pyro is trying to import pywinTypes.comerrer
        ##    self._raiseConnectionError(api.DatabaseError, 'Proxy cannot connect using=%s' % repr(kwargs))
        if ret is not True:
            self._raiseConnectionError(
                api.OperationalError, "Proxy returns error message=%s" % repr(ret)
            )

        self.supportsTransactions = self.getIndexedValue("supportsTransactions")
        self.paramstyle = self.getIndexedValue("paramstyle")
        self.timeout = self.getIndexedValue("timeout")
        if verbose:
            print("adodbapi.remote New connection at %X" % id(self))

    def _raiseConnectionError(self, errorclass, errorvalue):
        eh = self.errorhandler
        if eh is None:
            eh = api.standardErrorHandler
        eh(self, None, errorclass, errorvalue)

    def close(self):
        """Close the connection now (rather than whenever __del__ is called).

        The connection will be unusable from this point forward;
        an Error (or subclass) exception will be raised if any operation is attempted with the connection.
        The same applies to all cursor objects trying to use the connection.
        """
        for crsr in list(self.cursors.values())[
            :
        ]:  # copy the list, then close each one
            crsr.close()
        try:
            """close the underlying remote Connection object"""
            self.proxy.close()
            if verbose:
                print("adodbapi.remote Closed connection at %X" % id(self))
            object.__delattr__(
                self, "proxy"
            )  # future attempts to use closed cursor will be caught by __getattr__
        except Exception:
            pass

    def __del__(self):
        try:
            self.proxy.close()
        except:
            pass

    def commit(self):
        """Commit any pending transaction to the database.

        Note that if the database supports an auto-commit feature,
        this must be initially off. An interface method may be provided to turn it back on.
        Database modules that do not support transactions should implement this method with void functionality.
        """
        if not self.supportsTransactions:
            return
        result = self.proxy.commit()
        if result:
            self._raiseConnectionError(
                api.OperationalError, "Error during commit: %s" % result
            )

    def _rollback(self):
        """In case a database does provide transactions this method causes the the database to roll back to
        the start of any pending transaction. Closing a connection without committing the changes first will
        cause an implicit rollback to be performed.
        """
        result = self.proxy.rollback()
        if result:
            self._raiseConnectionError(
                api.OperationalError, "Error during rollback: %s" % result
            )

    def __setattr__(self, name, value):
        if name in ("paramstyle", "timeout", "autocommit"):
            if self.proxy:
                self.proxy.send_attribute_to_host(name, value)
        object.__setattr__(self, name, value)  # store attribute locally (too)

    def __getattr__(self, item):
        if (
            item == "rollback"
        ):  # the rollback method only appears if the database supports transactions
            if self.supportsTransactions:
                return (
                    self._rollback
                )  # return the rollback method so the caller can execute it.
            else:
                raise self.ProgrammingError(
                    "this data provider does not support Rollback"
                )
        elif item in (
            "dbms_name",
            "dbms_version",
            "connection_string",
            "autocommit",
        ):  # 'messages' ):
            return self.getIndexedValue(item)
        elif item == "proxy":
            raise self.ProgrammingError("Attempting to use closed connection")
        else:
            raise self.ProgrammingError('No remote access for attribute="%s"' % item)

    def getIndexedValue(self, index):
        r = self.proxy.get_attribute_for_remote(index)
        return r

    def cursor(self):
        "Return a new Cursor Object using the connection."
        myCursor = Cursor(self)
        return myCursor

    def _i_am_here(self, crsr):
        "message from a new cursor proclaiming its existence"
        self.cursors[crsr.id] = crsr

    def _i_am_closing(self, crsr):
        "message from a cursor giving connection a chance to clean up"
        try:
            del self.cursors[crsr.id]
        except:
            pass

    def __enter__(self):  # Connections are context managers
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._rollback()  # automatic rollback on errors
        else:
            self.commit()

    def get_table_names(self):
        return self.proxy.get_table_names()


def fixpickle(x):
    """pickle barfs on buffer(x) so we pass as array.array(x) then restore to original form for .execute()"""
    if x is None:
        return None
    if isinstance(x, dict):
        # for 'named' paramstyle user will pass a mapping
        newargs = {}
        for arg, val in list(x.items()):
            if isinstance(val, memoryViewType):
                newval = array.array("B")
                newval.fromstring(val)
                newargs[arg] = newval
            else:
                newargs[arg] = val
        return newargs
    # if not a mapping, then a sequence
    newargs = []
    for arg in x:
        if isinstance(arg, memoryViewType):
            newarg = array.array("B")
            newarg.fromstring(arg)
            newargs.append(newarg)
        else:
            newargs.append(arg)
    return newargs


class Cursor(object):
    def __init__(self, connection):
        self.command = None
        self.errorhandler = None  ## was: connection.errorhandler
        self.connection = connection
        self.proxy = self.connection.proxy
        self.rs = None  # the fetchable data for this cursor
        self.converters = NotImplemented
        self.id = connection.proxy.build_cursor()
        connection._i_am_here(self)
        self.recordset_format = api.RS_REMOTE
        if verbose:
            print(
                "%s New cursor at %X on conn %X"
                % (version, id(self), id(self.connection))
            )

    def prepare(self, operation):
        self.command = operation
        try:
            del self.description
        except AttributeError:
            pass
        self.proxy.crsr_prepare(self.id, operation)

    def __iter__(self):  # [2.1 Zamarev]
        return iter(self.fetchone, None)  # [2.1 Zamarev]

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

    def __getattr__(self, key):
        if key == "numberOfColumns":
            try:
                return len(self.rs[0])
            except:
                return 0
        if key == "description":
            try:
                self.description = self.proxy.crsr_get_description(self.id)[:]
                return self.description
            except TypeError:
                return None
        if key == "columnNames":
            try:
                r = dict(
                    self.proxy.crsr_get_columnNames(self.id)
                )  # copy the remote columns

            except TypeError:
                r = {}
            self.columnNames = r
            return r

        if key == "remote_cursor":
            raise api.OperationalError
        try:
            return self.proxy.crsr_get_attribute_for_remote(self.id, key)
        except AttributeError:
            raise api.InternalError(
                'Failure getting attribute "%s" from proxy cursor.' % key
            )

    def __setattr__(self, key, value):
        if key == "arraysize":
            self.proxy.crsr_set_arraysize(self.id, value)
        if key == "paramstyle":
            if value in api.accepted_paramstyles:
                self.proxy.crsr_set_paramstyle(self.id, value)
            else:
                self._raiseCursorError(
                    api.ProgrammingError, 'invalid paramstyle ="%s"' % value
                )
        object.__setattr__(self, key, value)

    def _raiseCursorError(self, errorclass, errorvalue):
        eh = self.errorhandler
        if eh is None:
            eh = api.standardErrorHandler
        eh(self.connection, self, errorclass, errorvalue)

    def execute(self, operation, parameters=None):
        if self.connection is None:
            self._raiseCursorError(
                ProgrammingError, "Attempted operation on closed cursor"
            )
        self.command = operation
        try:
            del self.description
        except AttributeError:
            pass
        try:
            del self.columnNames
        except AttributeError:
            pass
        fp = fixpickle(parameters)
        if verbose > 2:
            print(
                (
                    '%s executing "%s" with params=%s'
                    % (version, operation, repr(parameters))
                )
            )
        result = self.proxy.crsr_execute(self.id, operation, fp)
        if result:  # an exception was triggered
            self._raiseCursorError(result[0], result[1])

    def executemany(self, operation, seq_of_parameters):
        if self.connection is None:
            self._raiseCursorError(
                ProgrammingError, "Attempted operation on closed cursor"
            )
        self.command = operation
        try:
            del self.description
        except AttributeError:
            pass
        try:
            del self.columnNames
        except AttributeError:
            pass
        sq = [fixpickle(x) for x in seq_of_parameters]
        if verbose > 2:
            print(
                (
                    '%s executemany "%s" with params=%s'
                    % (version, operation, repr(seq_of_parameters))
                )
            )
        self.proxy.crsr_executemany(self.id, operation, sq)

    def nextset(self):
        try:
            del self.description
        except AttributeError:
            pass
        try:
            del self.columnNames
        except AttributeError:
            pass
        if verbose > 2:
            print(("%s nextset" % version))
        return self.proxy.crsr_nextset(self.id)

    def callproc(self, procname, parameters=None):
        if self.connection is None:
            self._raiseCursorError(
                ProgrammingError, "Attempted operation on closed cursor"
            )
        self.command = procname
        try:
            del self.description
        except AttributeError:
            pass
        try:
            del self.columnNames
        except AttributeError:
            pass
        fp = fixpickle(parameters)
        if verbose > 2:
            print(
                (
                    '%s callproc "%s" with params=%s'
                    % (version, procname, repr(parameters))
                )
            )
        return self.proxy.crsr_callproc(self.id, procname, fp)

    def fetchone(self):
        try:
            f1 = self.proxy.crsr_fetchone(self.id)
        except _BaseException as e:
            self._raiseCursorError(api.DatabaseError, e)
        else:
            if f1 is None:
                return None
            self.rs = [f1]
            return api.SQLrows(self.rs, 1, self)[
                0
            ]  # new object to hold the results of the fetch

    def fetchmany(self, size=None):
        try:
            self.rs = self.proxy.crsr_fetchmany(self.id, size)
            if not self.rs:
                return []
            r = api.SQLrows(self.rs, len(self.rs), self)
            return r
        except Exception as e:
            self._raiseCursorError(api.DatabaseError, e)

    def fetchall(self):
        try:
            self.rs = self.proxy.crsr_fetchall(self.id)
            if not self.rs:
                return []
            return api.SQLrows(self.rs, len(self.rs), self)
        except Exception as e:
            self._raiseCursorError(api.DatabaseError, e)

    def close(self):
        if self.connection is None:
            return
        self.connection._i_am_closing(self)  # take me off the connection's cursors list
        try:
            self.proxy.crsr_close(self.id)
        except:
            pass
        try:
            del self.description
        except:
            pass
        try:
            del self.rs  # let go of the recordset
        except:
            pass
        self.connection = (
            None  # this will make all future method calls on me throw an exception
        )
        self.proxy = None
        if verbose:
            print("adodbapi.remote Closed cursor at %X" % id(self))

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def setinputsizes(self, sizes):
        pass

    def setoutputsize(self, size, column=None):
        pass
