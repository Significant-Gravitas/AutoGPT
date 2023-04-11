import copy
import errno
import io
import os
import socket
import sys
import threading
import weakref
from abc import abstractmethod
from io import SEEK_END
from itertools import chain
from queue import Empty, Full, LifoQueue
from time import time
from typing import Optional, Union
from urllib.parse import parse_qs, unquote, urlparse

from redis.backoff import NoBackoff
from redis.credentials import CredentialProvider, UsernamePasswordCredentialProvider
from redis.exceptions import (
    AuthenticationError,
    AuthenticationWrongNumberOfArgsError,
    BusyLoadingError,
    ChildDeadlockedError,
    ConnectionError,
    DataError,
    ExecAbortError,
    InvalidResponse,
    ModuleError,
    NoPermissionError,
    NoScriptError,
    ReadOnlyError,
    RedisError,
    ResponseError,
    TimeoutError,
)
from redis.retry import Retry
from redis.utils import (
    CRYPTOGRAPHY_AVAILABLE,
    HIREDIS_AVAILABLE,
    HIREDIS_PACK_AVAILABLE,
    str_if_bytes,
)

try:
    import ssl

    ssl_available = True
except ImportError:
    ssl_available = False

NONBLOCKING_EXCEPTION_ERROR_NUMBERS = {BlockingIOError: errno.EWOULDBLOCK}

if ssl_available:
    if hasattr(ssl, "SSLWantReadError"):
        NONBLOCKING_EXCEPTION_ERROR_NUMBERS[ssl.SSLWantReadError] = 2
        NONBLOCKING_EXCEPTION_ERROR_NUMBERS[ssl.SSLWantWriteError] = 2
    else:
        NONBLOCKING_EXCEPTION_ERROR_NUMBERS[ssl.SSLError] = 2

NONBLOCKING_EXCEPTIONS = tuple(NONBLOCKING_EXCEPTION_ERROR_NUMBERS.keys())

if HIREDIS_AVAILABLE:
    import hiredis

SYM_STAR = b"*"
SYM_DOLLAR = b"$"
SYM_CRLF = b"\r\n"
SYM_EMPTY = b""

SERVER_CLOSED_CONNECTION_ERROR = "Connection closed by server."

SENTINEL = object()
MODULE_LOAD_ERROR = "Error loading the extension. Please check the server logs."
NO_SUCH_MODULE_ERROR = "Error unloading module: no such module with that name"
MODULE_UNLOAD_NOT_POSSIBLE_ERROR = "Error unloading module: operation not possible."
MODULE_EXPORTS_DATA_TYPES_ERROR = (
    "Error unloading module: the module "
    "exports one or more module-side data "
    "types, can't unload"
)
# user send an AUTH cmd to a server without authorization configured
NO_AUTH_SET_ERROR = {
    # Redis >= 6.0
    "AUTH <password> called without any password "
    "configured for the default user. Are you sure "
    "your configuration is correct?": AuthenticationError,
    # Redis < 6.0
    "Client sent AUTH, but no password is set": AuthenticationError,
}


class Encoder:
    "Encode strings to bytes-like and decode bytes-like to strings"

    def __init__(self, encoding, encoding_errors, decode_responses):
        self.encoding = encoding
        self.encoding_errors = encoding_errors
        self.decode_responses = decode_responses

    def encode(self, value):
        "Return a bytestring or bytes-like representation of the value"
        if isinstance(value, (bytes, memoryview)):
            return value
        elif isinstance(value, bool):
            # special case bool since it is a subclass of int
            raise DataError(
                "Invalid input of type: 'bool'. Convert to a "
                "bytes, string, int or float first."
            )
        elif isinstance(value, (int, float)):
            value = repr(value).encode()
        elif not isinstance(value, str):
            # a value we don't know how to deal with. throw an error
            typename = type(value).__name__
            raise DataError(
                f"Invalid input of type: '{typename}'. "
                f"Convert to a bytes, string, int or float first."
            )
        if isinstance(value, str):
            value = value.encode(self.encoding, self.encoding_errors)
        return value

    def decode(self, value, force=False):
        "Return a unicode string from the bytes-like representation"
        if self.decode_responses or force:
            if isinstance(value, memoryview):
                value = value.tobytes()
            if isinstance(value, bytes):
                value = value.decode(self.encoding, self.encoding_errors)
        return value


class BaseParser:
    EXCEPTION_CLASSES = {
        "ERR": {
            "max number of clients reached": ConnectionError,
            "invalid password": AuthenticationError,
            # some Redis server versions report invalid command syntax
            # in lowercase
            "wrong number of arguments "
            "for 'auth' command": AuthenticationWrongNumberOfArgsError,
            # some Redis server versions report invalid command syntax
            # in uppercase
            "wrong number of arguments "
            "for 'AUTH' command": AuthenticationWrongNumberOfArgsError,
            MODULE_LOAD_ERROR: ModuleError,
            MODULE_EXPORTS_DATA_TYPES_ERROR: ModuleError,
            NO_SUCH_MODULE_ERROR: ModuleError,
            MODULE_UNLOAD_NOT_POSSIBLE_ERROR: ModuleError,
            **NO_AUTH_SET_ERROR,
        },
        "WRONGPASS": AuthenticationError,
        "EXECABORT": ExecAbortError,
        "LOADING": BusyLoadingError,
        "NOSCRIPT": NoScriptError,
        "READONLY": ReadOnlyError,
        "NOAUTH": AuthenticationError,
        "NOPERM": NoPermissionError,
    }

    def parse_error(self, response):
        "Parse an error response"
        error_code = response.split(" ")[0]
        if error_code in self.EXCEPTION_CLASSES:
            response = response[len(error_code) + 1 :]
            exception_class = self.EXCEPTION_CLASSES[error_code]
            if isinstance(exception_class, dict):
                exception_class = exception_class.get(response, ResponseError)
            return exception_class(response)
        return ResponseError(response)


class SocketBuffer:
    def __init__(
        self, socket: socket.socket, socket_read_size: int, socket_timeout: float
    ):
        self._sock = socket
        self.socket_read_size = socket_read_size
        self.socket_timeout = socket_timeout
        self._buffer = io.BytesIO()

    def unread_bytes(self) -> int:
        """
        Remaining unread length of buffer
        """
        pos = self._buffer.tell()
        end = self._buffer.seek(0, SEEK_END)
        self._buffer.seek(pos)
        return end - pos

    def _read_from_socket(
        self,
        length: Optional[int] = None,
        timeout: Union[float, object] = SENTINEL,
        raise_on_timeout: Optional[bool] = True,
    ) -> bool:
        sock = self._sock
        socket_read_size = self.socket_read_size
        marker = 0
        custom_timeout = timeout is not SENTINEL

        buf = self._buffer
        current_pos = buf.tell()
        buf.seek(0, SEEK_END)
        if custom_timeout:
            sock.settimeout(timeout)
        try:
            while True:
                data = self._sock.recv(socket_read_size)
                # an empty string indicates the server shutdown the socket
                if isinstance(data, bytes) and len(data) == 0:
                    raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
                buf.write(data)
                data_length = len(data)
                marker += data_length

                if length is not None and length > marker:
                    continue
                return True
        except socket.timeout:
            if raise_on_timeout:
                raise TimeoutError("Timeout reading from socket")
            return False
        except NONBLOCKING_EXCEPTIONS as ex:
            # if we're in nonblocking mode and the recv raises a
            # blocking error, simply return False indicating that
            # there's no data to be read. otherwise raise the
            # original exception.
            allowed = NONBLOCKING_EXCEPTION_ERROR_NUMBERS.get(ex.__class__, -1)
            if not raise_on_timeout and ex.errno == allowed:
                return False
            raise ConnectionError(f"Error while reading from socket: {ex.args}")
        finally:
            buf.seek(current_pos)
            if custom_timeout:
                sock.settimeout(self.socket_timeout)

    def can_read(self, timeout: float) -> bool:
        return bool(self.unread_bytes()) or self._read_from_socket(
            timeout=timeout, raise_on_timeout=False
        )

    def read(self, length: int) -> bytes:
        length = length + 2  # make sure to read the \r\n terminator
        # BufferIO will return less than requested if buffer is short
        data = self._buffer.read(length)
        missing = length - len(data)
        if missing:
            # fill up the buffer and read the remainder
            self._read_from_socket(missing)
            data += self._buffer.read(missing)
        return data[:-2]

    def readline(self) -> bytes:
        buf = self._buffer
        data = buf.readline()
        while not data.endswith(SYM_CRLF):
            # there's more data in the socket that we need
            self._read_from_socket()
            data += buf.readline()

        return data[:-2]

    def get_pos(self) -> int:
        """
        Get current read position
        """
        return self._buffer.tell()

    def rewind(self, pos: int) -> None:
        """
        Rewind the buffer to a specific position, to re-start reading
        """
        self._buffer.seek(pos)

    def purge(self) -> None:
        """
        After a successful read, purge the read part of buffer
        """
        unread = self.unread_bytes()

        # Only if we have read all of the buffer do we truncate, to
        # reduce the amount of memory thrashing.  This heuristic
        # can be changed or removed later.
        if unread > 0:
            return

        if unread > 0:
            # move unread data to the front
            view = self._buffer.getbuffer()
            view[:unread] = view[-unread:]
        self._buffer.truncate(unread)
        self._buffer.seek(0)

    def close(self) -> None:
        try:
            self._buffer.close()
        except Exception:
            # issue #633 suggests the purge/close somehow raised a
            # BadFileDescriptor error. Perhaps the client ran out of
            # memory or something else? It's probably OK to ignore
            # any error being raised from purge/close since we're
            # removing the reference to the instance below.
            pass
        self._buffer = None
        self._sock = None


class PythonParser(BaseParser):
    "Plain Python parsing class"

    def __init__(self, socket_read_size):
        self.socket_read_size = socket_read_size
        self.encoder = None
        self._sock = None
        self._buffer = None

    def __del__(self):
        try:
            self.on_disconnect()
        except Exception:
            pass

    def on_connect(self, connection):
        "Called when the socket connects"
        self._sock = connection._sock
        self._buffer = SocketBuffer(
            self._sock, self.socket_read_size, connection.socket_timeout
        )
        self.encoder = connection.encoder

    def on_disconnect(self):
        "Called when the socket disconnects"
        self._sock = None
        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None
        self.encoder = None

    def can_read(self, timeout):
        return self._buffer and self._buffer.can_read(timeout)

    def read_response(self, disable_decoding=False):
        pos = self._buffer.get_pos() if self._buffer else None
        try:
            result = self._read_response(disable_decoding=disable_decoding)
        except BaseException:
            if self._buffer:
                self._buffer.rewind(pos)
            raise
        else:
            self._buffer.purge()
            return result

    def _read_response(self, disable_decoding=False):
        raw = self._buffer.readline()
        if not raw:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)

        byte, response = raw[:1], raw[1:]

        # server returned an error
        if byte == b"-":
            response = response.decode("utf-8", errors="replace")
            error = self.parse_error(response)
            # if the error is a ConnectionError, raise immediately so the user
            # is notified
            if isinstance(error, ConnectionError):
                raise error
            # otherwise, we're dealing with a ResponseError that might belong
            # inside a pipeline response. the connection's read_response()
            # and/or the pipeline's execute() will raise this error if
            # necessary, so just return the exception instance here.
            return error
        # single value
        elif byte == b"+":
            pass
        # int value
        elif byte == b":":
            return int(response)
        # bulk response
        elif byte == b"$" and response == b"-1":
            return None
        elif byte == b"$":
            response = self._buffer.read(int(response))
        # multi-bulk response
        elif byte == b"*" and response == b"-1":
            return None
        elif byte == b"*":
            response = [
                self._read_response(disable_decoding=disable_decoding)
                for i in range(int(response))
            ]
        else:
            raise InvalidResponse(f"Protocol Error: {raw!r}")

        if disable_decoding is False:
            response = self.encoder.decode(response)
        return response


class HiredisParser(BaseParser):
    "Parser class for connections using Hiredis"

    def __init__(self, socket_read_size):
        if not HIREDIS_AVAILABLE:
            raise RedisError("Hiredis is not installed")
        self.socket_read_size = socket_read_size
        self._buffer = bytearray(socket_read_size)

    def __del__(self):
        try:
            self.on_disconnect()
        except Exception:
            pass

    def on_connect(self, connection, **kwargs):
        self._sock = connection._sock
        self._socket_timeout = connection.socket_timeout
        kwargs = {
            "protocolError": InvalidResponse,
            "replyError": self.parse_error,
            "errors": connection.encoder.encoding_errors,
        }

        if connection.encoder.decode_responses:
            kwargs["encoding"] = connection.encoder.encoding
        self._reader = hiredis.Reader(**kwargs)
        self._next_response = False

    def on_disconnect(self):
        self._sock = None
        self._reader = None
        self._next_response = False

    def can_read(self, timeout):
        if not self._reader:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)

        if self._next_response is False:
            self._next_response = self._reader.gets()
            if self._next_response is False:
                return self.read_from_socket(timeout=timeout, raise_on_timeout=False)
        return True

    def read_from_socket(self, timeout=SENTINEL, raise_on_timeout=True):
        sock = self._sock
        custom_timeout = timeout is not SENTINEL
        try:
            if custom_timeout:
                sock.settimeout(timeout)
            bufflen = self._sock.recv_into(self._buffer)
            if bufflen == 0:
                raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
            self._reader.feed(self._buffer, 0, bufflen)
            # data was read from the socket and added to the buffer.
            # return True to indicate that data was read.
            return True
        except socket.timeout:
            if raise_on_timeout:
                raise TimeoutError("Timeout reading from socket")
            return False
        except NONBLOCKING_EXCEPTIONS as ex:
            # if we're in nonblocking mode and the recv raises a
            # blocking error, simply return False indicating that
            # there's no data to be read. otherwise raise the
            # original exception.
            allowed = NONBLOCKING_EXCEPTION_ERROR_NUMBERS.get(ex.__class__, -1)
            if not raise_on_timeout and ex.errno == allowed:
                return False
            raise ConnectionError(f"Error while reading from socket: {ex.args}")
        finally:
            if custom_timeout:
                sock.settimeout(self._socket_timeout)

    def read_response(self, disable_decoding=False):
        if not self._reader:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)

        # _next_response might be cached from a can_read() call
        if self._next_response is not False:
            response = self._next_response
            self._next_response = False
            return response

        if disable_decoding:
            response = self._reader.gets(False)
        else:
            response = self._reader.gets()

        while response is False:
            self.read_from_socket()
            if disable_decoding:
                response = self._reader.gets(False)
            else:
                response = self._reader.gets()
        # if the response is a ConnectionError or the response is a list and
        # the first item is a ConnectionError, raise it as something bad
        # happened
        if isinstance(response, ConnectionError):
            raise response
        elif (
            isinstance(response, list)
            and response
            and isinstance(response[0], ConnectionError)
        ):
            raise response[0]
        return response


DefaultParser: BaseParser
if HIREDIS_AVAILABLE:
    DefaultParser = HiredisParser
else:
    DefaultParser = PythonParser


class HiredisRespSerializer:
    def pack(self, *args):
        """Pack a series of arguments into the Redis protocol"""
        output = []

        if isinstance(args[0], str):
            args = tuple(args[0].encode().split()) + args[1:]
        elif b" " in args[0]:
            args = tuple(args[0].split()) + args[1:]
        try:
            output.append(hiredis.pack_command(args))
        except TypeError:
            _, value, traceback = sys.exc_info()
            raise DataError(value).with_traceback(traceback)

        return output


class PythonRespSerializer:
    def __init__(self, buffer_cutoff, encode) -> None:
        self._buffer_cutoff = buffer_cutoff
        self.encode = encode

    def pack(self, *args):
        """Pack a series of arguments into the Redis protocol"""
        output = []
        # the client might have included 1 or more literal arguments in
        # the command name, e.g., 'CONFIG GET'. The Redis server expects these
        # arguments to be sent separately, so split the first argument
        # manually. These arguments should be bytestrings so that they are
        # not encoded.
        if isinstance(args[0], str):
            args = tuple(args[0].encode().split()) + args[1:]
        elif b" " in args[0]:
            args = tuple(args[0].split()) + args[1:]

        buff = SYM_EMPTY.join((SYM_STAR, str(len(args)).encode(), SYM_CRLF))

        buffer_cutoff = self._buffer_cutoff
        for arg in map(self.encode, args):
            # to avoid large string mallocs, chunk the command into the
            # output list if we're sending large values or memoryviews
            arg_length = len(arg)
            if (
                len(buff) > buffer_cutoff
                or arg_length > buffer_cutoff
                or isinstance(arg, memoryview)
            ):
                buff = SYM_EMPTY.join(
                    (buff, SYM_DOLLAR, str(arg_length).encode(), SYM_CRLF)
                )
                output.append(buff)
                output.append(arg)
                buff = SYM_CRLF
            else:
                buff = SYM_EMPTY.join(
                    (
                        buff,
                        SYM_DOLLAR,
                        str(arg_length).encode(),
                        SYM_CRLF,
                        arg,
                        SYM_CRLF,
                    )
                )
        output.append(buff)
        return output


class AbstractConnection:
    "Manages communication to and from a Redis server"

    def __init__(
        self,
        db=0,
        password=None,
        retry_on_timeout=False,
        retry_on_error=SENTINEL,
        encoding="utf-8",
        encoding_errors="strict",
        decode_responses=False,
        parser_class=DefaultParser,
        socket_read_size=65536,
        health_check_interval=0,
        client_name=None,
        username=None,
        retry=None,
        redis_connect_func=None,
        credential_provider: Optional[CredentialProvider] = None,
        command_packer=None,
    ):
        """
        Initialize a new Connection.
        To specify a retry policy for specific errors, first set
        `retry_on_error` to a list of the error/s to retry on, then set
        `retry` to a valid `Retry` object.
        To retry on TimeoutError, `retry_on_timeout` can also be set to `True`.
        """
        if (username or password) and credential_provider is not None:
            raise DataError(
                "'username' and 'password' cannot be passed along with 'credential_"
                "provider'. Please provide only one of the following arguments: \n"
                "1. 'password' and (optional) 'username'\n"
                "2. 'credential_provider'"
            )
        self.pid = os.getpid()
        self.db = db
        self.client_name = client_name
        self.credential_provider = credential_provider
        self.password = password
        self.username = username
        self.retry_on_timeout = retry_on_timeout
        if retry_on_error is SENTINEL:
            retry_on_error = []
        if retry_on_timeout:
            # Add TimeoutError to the errors list to retry on
            retry_on_error.append(TimeoutError)
        self.retry_on_error = retry_on_error
        if retry or retry_on_error:
            if retry is None:
                self.retry = Retry(NoBackoff(), 1)
            else:
                # deep-copy the Retry object as it is mutable
                self.retry = copy.deepcopy(retry)
            # Update the retry's supported errors with the specified errors
            self.retry.update_supported_errors(retry_on_error)
        else:
            self.retry = Retry(NoBackoff(), 0)
        self.health_check_interval = health_check_interval
        self.next_health_check = 0
        self.redis_connect_func = redis_connect_func
        self.encoder = Encoder(encoding, encoding_errors, decode_responses)
        self._sock = None
        self._socket_read_size = socket_read_size
        self.set_parser(parser_class)
        self._connect_callbacks = []
        self._buffer_cutoff = 6000
        self._command_packer = self._construct_command_packer(command_packer)

    def __repr__(self):
        repr_args = ",".join([f"{k}={v}" for k, v in self.repr_pieces()])
        return f"{self.__class__.__name__}<{repr_args}>"

    @abstractmethod
    def repr_pieces(self):
        pass

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

    def _construct_command_packer(self, packer):
        if packer is not None:
            return packer
        elif HIREDIS_PACK_AVAILABLE:
            return HiredisRespSerializer()
        else:
            return PythonRespSerializer(self._buffer_cutoff, self.encoder.encode)

    def register_connect_callback(self, callback):
        self._connect_callbacks.append(weakref.WeakMethod(callback))

    def clear_connect_callbacks(self):
        self._connect_callbacks = []

    def set_parser(self, parser_class):
        """
        Creates a new instance of parser_class with socket size:
        _socket_read_size and assigns it to the parser for the connection
        :param parser_class: The required parser class
        """
        self._parser = parser_class(socket_read_size=self._socket_read_size)

    def connect(self):
        "Connects to the Redis server if not already connected"
        if self._sock:
            return
        try:
            sock = self.retry.call_with_retry(
                lambda: self._connect(), lambda error: self.disconnect(error)
            )
        except socket.timeout:
            raise TimeoutError("Timeout connecting to server")
        except OSError as e:
            raise ConnectionError(self._error_message(e))

        self._sock = sock
        try:
            if self.redis_connect_func is None:
                # Use the default on_connect function
                self.on_connect()
            else:
                # Use the passed function redis_connect_func
                self.redis_connect_func(self)
        except RedisError:
            # clean up after any error in on_connect
            self.disconnect()
            raise

        # run any user callbacks. right now the only internal callback
        # is for pubsub channel/pattern resubscription
        for ref in self._connect_callbacks:
            callback = ref()
            if callback:
                callback(self)

    @abstractmethod
    def _connect(self):
        pass

    @abstractmethod
    def _host_error(self):
        pass

    @abstractmethod
    def _error_message(self, exception):
        pass

    def on_connect(self):
        "Initialize the connection, authenticate and select a database"
        self._parser.on_connect(self)

        # if credential provider or username and/or password are set, authenticate
        if self.credential_provider or (self.username or self.password):
            cred_provider = (
                self.credential_provider
                or UsernamePasswordCredentialProvider(self.username, self.password)
            )
            auth_args = cred_provider.get_credentials()
            # avoid checking health here -- PING will fail if we try
            # to check the health prior to the AUTH
            self.send_command("AUTH", *auth_args, check_health=False)

            try:
                auth_response = self.read_response()
            except AuthenticationWrongNumberOfArgsError:
                # a username and password were specified but the Redis
                # server seems to be < 6.0.0 which expects a single password
                # arg. retry auth with just the password.
                # https://github.com/andymccurdy/redis-py/issues/1274
                self.send_command("AUTH", auth_args[-1], check_health=False)
                auth_response = self.read_response()

            if str_if_bytes(auth_response) != "OK":
                raise AuthenticationError("Invalid Username or Password")

        # if a client_name is given, set it
        if self.client_name:
            self.send_command("CLIENT", "SETNAME", self.client_name)
            if str_if_bytes(self.read_response()) != "OK":
                raise ConnectionError("Error setting client name")

        # if a database is specified, switch to it
        if self.db:
            self.send_command("SELECT", self.db)
            if str_if_bytes(self.read_response()) != "OK":
                raise ConnectionError("Invalid Database")

    def disconnect(self, *args):
        "Disconnects from the Redis server"
        self._parser.on_disconnect()
        if self._sock is None:
            return

        if os.getpid() == self.pid:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        try:
            self._sock.close()
        except OSError:
            pass
        self._sock = None

    def _send_ping(self):
        """Send PING, expect PONG in return"""
        self.send_command("PING", check_health=False)
        if str_if_bytes(self.read_response()) != "PONG":
            raise ConnectionError("Bad response from PING health check")

    def _ping_failed(self, error):
        """Function to call when PING fails"""
        self.disconnect()

    def check_health(self):
        """Check the health of the connection with a PING/PONG"""
        if self.health_check_interval and time() > self.next_health_check:
            self.retry.call_with_retry(self._send_ping, self._ping_failed)

    def send_packed_command(self, command, check_health=True):
        """Send an already packed command to the Redis server"""
        if not self._sock:
            self.connect()
        # guard against health check recursion
        if check_health:
            self.check_health()
        try:
            if isinstance(command, str):
                command = [command]
            for item in command:
                self._sock.sendall(item)
        except socket.timeout:
            self.disconnect()
            raise TimeoutError("Timeout writing to socket")
        except OSError as e:
            self.disconnect()
            if len(e.args) == 1:
                errno, errmsg = "UNKNOWN", e.args[0]
            else:
                errno = e.args[0]
                errmsg = e.args[1]
            raise ConnectionError(f"Error {errno} while writing to socket. {errmsg}.")
        except Exception:
            self.disconnect()
            raise

    def send_command(self, *args, **kwargs):
        """Pack and send a command to the Redis server"""
        self.send_packed_command(
            self._command_packer.pack(*args),
            check_health=kwargs.get("check_health", True),
        )

    def can_read(self, timeout=0):
        """Poll the socket to see if there's data that can be read."""
        sock = self._sock
        if not sock:
            self.connect()

        host_error = self._host_error()

        try:
            return self._parser.can_read(timeout)
        except OSError as e:
            self.disconnect()
            raise ConnectionError(f"Error while reading from {host_error}: {e.args}")

    def read_response(self, disable_decoding=False):
        """Read the response from a previously sent command"""

        host_error = self._host_error()

        try:
            response = self._parser.read_response(disable_decoding=disable_decoding)
        except socket.timeout:
            self.disconnect()
            raise TimeoutError(f"Timeout reading from {host_error}")
        except OSError as e:
            self.disconnect()
            raise ConnectionError(
                f"Error while reading from {host_error}" f" : {e.args}"
            )
        except Exception:
            self.disconnect()
            raise

        if self.health_check_interval:
            self.next_health_check = time() + self.health_check_interval

        if isinstance(response, ResponseError):
            raise response
        return response

    def pack_command(self, *args):
        """Pack a series of arguments into the Redis protocol"""
        return self._command_packer.pack(*args)

    def pack_commands(self, commands):
        """Pack multiple commands into the Redis protocol"""
        output = []
        pieces = []
        buffer_length = 0
        buffer_cutoff = self._buffer_cutoff

        for cmd in commands:
            for chunk in self._command_packer.pack(*cmd):
                chunklen = len(chunk)
                if (
                    buffer_length > buffer_cutoff
                    or chunklen > buffer_cutoff
                    or isinstance(chunk, memoryview)
                ):
                    if pieces:
                        output.append(SYM_EMPTY.join(pieces))
                    buffer_length = 0
                    pieces = []

                if chunklen > buffer_cutoff or isinstance(chunk, memoryview):
                    output.append(chunk)
                else:
                    pieces.append(chunk)
                    buffer_length += chunklen

        if pieces:
            output.append(SYM_EMPTY.join(pieces))
        return output


class Connection(AbstractConnection):
    "Manages TCP communication to and from a Redis server"

    def __init__(
        self,
        host="localhost",
        port=6379,
        socket_timeout=None,
        socket_connect_timeout=None,
        socket_keepalive=False,
        socket_keepalive_options=None,
        socket_type=0,
        **kwargs,
    ):
        self.host = host
        self.port = int(port)
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout or socket_timeout
        self.socket_keepalive = socket_keepalive
        self.socket_keepalive_options = socket_keepalive_options or {}
        self.socket_type = socket_type
        super().__init__(**kwargs)

    def repr_pieces(self):
        pieces = [("host", self.host), ("port", self.port), ("db", self.db)]
        if self.client_name:
            pieces.append(("client_name", self.client_name))
        return pieces

    def _connect(self):
        "Create a TCP socket connection"
        # we want to mimic what socket.create_connection does to support
        # ipv4/ipv6, but we want to set options prior to calling
        # socket.connect()
        err = None
        for res in socket.getaddrinfo(
            self.host, self.port, self.socket_type, socket.SOCK_STREAM
        ):
            family, socktype, proto, canonname, socket_address = res
            sock = None
            try:
                sock = socket.socket(family, socktype, proto)
                # TCP_NODELAY
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                # TCP_KEEPALIVE
                if self.socket_keepalive:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    for k, v in self.socket_keepalive_options.items():
                        sock.setsockopt(socket.IPPROTO_TCP, k, v)

                # set the socket_connect_timeout before we connect
                sock.settimeout(self.socket_connect_timeout)

                # connect
                sock.connect(socket_address)

                # set the socket_timeout now that we're connected
                sock.settimeout(self.socket_timeout)
                return sock

            except OSError as _:
                err = _
                if sock is not None:
                    sock.close()

        if err is not None:
            raise err
        raise OSError("socket.getaddrinfo returned an empty list")

    def _host_error(self):
        return f"{self.host}:{self.port}"

    def _error_message(self, exception):
        # args for socket.error can either be (errno, "message")
        # or just "message"

        host_error = self._host_error()

        if len(exception.args) == 1:
            try:
                return f"Error connecting to {host_error}. \
                        {exception.args[0]}."
            except AttributeError:
                return f"Connection Error: {exception.args[0]}"
        else:
            try:
                return (
                    f"Error {exception.args[0]} connecting to "
                    f"{host_error}. {exception.args[1]}."
                )
            except AttributeError:
                return f"Connection Error: {exception.args[0]}"


class SSLConnection(Connection):
    """Manages SSL connections to and from the Redis server(s).
    This class extends the Connection class, adding SSL functionality, and making
    use of ssl.SSLContext (https://docs.python.org/3/library/ssl.html#ssl.SSLContext)
    """  # noqa

    def __init__(
        self,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_cert_reqs="required",
        ssl_ca_certs=None,
        ssl_ca_data=None,
        ssl_check_hostname=False,
        ssl_ca_path=None,
        ssl_password=None,
        ssl_validate_ocsp=False,
        ssl_validate_ocsp_stapled=False,
        ssl_ocsp_context=None,
        ssl_ocsp_expected_cert=None,
        **kwargs,
    ):
        """Constructor

        Args:
            ssl_keyfile: Path to an ssl private key. Defaults to None.
            ssl_certfile: Path to an ssl certificate. Defaults to None.
            ssl_cert_reqs: The string value for the SSLContext.verify_mode (none, optional, required). Defaults to "required".
            ssl_ca_certs: The path to a file of concatenated CA certificates in PEM format. Defaults to None.
            ssl_ca_data: Either an ASCII string of one or more PEM-encoded certificates or a bytes-like object of DER-encoded certificates.
            ssl_check_hostname: If set, match the hostname during the SSL handshake. Defaults to False.
            ssl_ca_path: The path to a directory containing several CA certificates in PEM format. Defaults to None.
            ssl_password: Password for unlocking an encrypted private key. Defaults to None.

            ssl_validate_ocsp: If set, perform a full ocsp validation (i.e not a stapled verification)
            ssl_validate_ocsp_stapled: If set, perform a validation on a stapled ocsp response
            ssl_ocsp_context: A fully initialized OpenSSL.SSL.Context object to be used in verifying the ssl_ocsp_expected_cert
            ssl_ocsp_expected_cert: A PEM armoured string containing the expected certificate to be returned from the ocsp verification service.

        Raises:
            RedisError
        """  # noqa
        if not ssl_available:
            raise RedisError("Python wasn't built with SSL support")

        self.keyfile = ssl_keyfile
        self.certfile = ssl_certfile
        if ssl_cert_reqs is None:
            ssl_cert_reqs = ssl.CERT_NONE
        elif isinstance(ssl_cert_reqs, str):
            CERT_REQS = {
                "none": ssl.CERT_NONE,
                "optional": ssl.CERT_OPTIONAL,
                "required": ssl.CERT_REQUIRED,
            }
            if ssl_cert_reqs not in CERT_REQS:
                raise RedisError(
                    f"Invalid SSL Certificate Requirements Flag: {ssl_cert_reqs}"
                )
            ssl_cert_reqs = CERT_REQS[ssl_cert_reqs]
        self.cert_reqs = ssl_cert_reqs
        self.ca_certs = ssl_ca_certs
        self.ca_data = ssl_ca_data
        self.ca_path = ssl_ca_path
        self.check_hostname = ssl_check_hostname
        self.certificate_password = ssl_password
        self.ssl_validate_ocsp = ssl_validate_ocsp
        self.ssl_validate_ocsp_stapled = ssl_validate_ocsp_stapled
        self.ssl_ocsp_context = ssl_ocsp_context
        self.ssl_ocsp_expected_cert = ssl_ocsp_expected_cert
        super().__init__(**kwargs)

    def _connect(self):
        "Wrap the socket with SSL support"
        sock = super()._connect()
        context = ssl.create_default_context()
        context.check_hostname = self.check_hostname
        context.verify_mode = self.cert_reqs
        if self.certfile or self.keyfile:
            context.load_cert_chain(
                certfile=self.certfile,
                keyfile=self.keyfile,
                password=self.certificate_password,
            )
        if (
            self.ca_certs is not None
            or self.ca_path is not None
            or self.ca_data is not None
        ):
            context.load_verify_locations(
                cafile=self.ca_certs, capath=self.ca_path, cadata=self.ca_data
            )
        sslsock = context.wrap_socket(sock, server_hostname=self.host)
        if self.ssl_validate_ocsp is True and CRYPTOGRAPHY_AVAILABLE is False:
            raise RedisError("cryptography is not installed.")

        if self.ssl_validate_ocsp_stapled and self.ssl_validate_ocsp:
            raise RedisError(
                "Either an OCSP staple or pure OCSP connection must be validated "
                "- not both."
            )

        # validation for the stapled case
        if self.ssl_validate_ocsp_stapled:
            import OpenSSL

            from .ocsp import ocsp_staple_verifier

            # if a context is provided use it - otherwise, a basic context
            if self.ssl_ocsp_context is None:
                staple_ctx = OpenSSL.SSL.Context(OpenSSL.SSL.SSLv23_METHOD)
                staple_ctx.use_certificate_file(self.certfile)
                staple_ctx.use_privatekey_file(self.keyfile)
            else:
                staple_ctx = self.ssl_ocsp_context

            staple_ctx.set_ocsp_client_callback(
                ocsp_staple_verifier, self.ssl_ocsp_expected_cert
            )

            #  need another socket
            con = OpenSSL.SSL.Connection(staple_ctx, socket.socket())
            con.request_ocsp()
            con.connect((self.host, self.port))
            con.do_handshake()
            con.shutdown()
            return sslsock

        # pure ocsp validation
        if self.ssl_validate_ocsp is True and CRYPTOGRAPHY_AVAILABLE:
            from .ocsp import OCSPVerifier

            o = OCSPVerifier(sslsock, self.host, self.port, self.ca_certs)
            if o.is_valid():
                return sslsock
            else:
                raise ConnectionError("ocsp validation error")
        return sslsock


class UnixDomainSocketConnection(AbstractConnection):
    "Manages UDS communication to and from a Redis server"

    def __init__(self, path="", socket_timeout=None, **kwargs):
        self.path = path
        self.socket_timeout = socket_timeout
        super().__init__(**kwargs)

    def repr_pieces(self):
        pieces = [("path", self.path), ("db", self.db)]
        if self.client_name:
            pieces.append(("client_name", self.client_name))
        return pieces

    def _connect(self):
        "Create a Unix domain socket connection"
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.socket_timeout)
        sock.connect(self.path)
        return sock

    def _host_error(self):
        return self.path

    def _error_message(self, exception):
        # args for socket.error can either be (errno, "message")
        # or just "message"
        host_error = self._host_error()
        if len(exception.args) == 1:
            return (
                f"Error connecting to unix socket: {host_error}. {exception.args[0]}."
            )
        else:
            return (
                f"Error {exception.args[0]} connecting to unix socket: "
                f"{host_error}. {exception.args[1]}."
            )


FALSE_STRINGS = ("0", "F", "FALSE", "N", "NO")


def to_bool(value):
    if value is None or value == "":
        return None
    if isinstance(value, str) and value.upper() in FALSE_STRINGS:
        return False
    return bool(value)


URL_QUERY_ARGUMENT_PARSERS = {
    "db": int,
    "socket_timeout": float,
    "socket_connect_timeout": float,
    "socket_keepalive": to_bool,
    "retry_on_timeout": to_bool,
    "retry_on_error": list,
    "max_connections": int,
    "health_check_interval": int,
    "ssl_check_hostname": to_bool,
}


def parse_url(url):
    if not (
        url.startswith("redis://")
        or url.startswith("rediss://")
        or url.startswith("unix://")
    ):
        raise ValueError(
            "Redis URL must specify one of the following "
            "schemes (redis://, rediss://, unix://)"
        )

    url = urlparse(url)
    kwargs = {}

    for name, value in parse_qs(url.query).items():
        if value and len(value) > 0:
            value = unquote(value[0])
            parser = URL_QUERY_ARGUMENT_PARSERS.get(name)
            if parser:
                try:
                    kwargs[name] = parser(value)
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid value for `{name}` in connection URL.")
            else:
                kwargs[name] = value

    if url.username:
        kwargs["username"] = unquote(url.username)
    if url.password:
        kwargs["password"] = unquote(url.password)

    # We only support redis://, rediss:// and unix:// schemes.
    if url.scheme == "unix":
        if url.path:
            kwargs["path"] = unquote(url.path)
        kwargs["connection_class"] = UnixDomainSocketConnection

    else:  # implied:  url.scheme in ("redis", "rediss"):
        if url.hostname:
            kwargs["host"] = unquote(url.hostname)
        if url.port:
            kwargs["port"] = int(url.port)

        # If there's a path argument, use it as the db argument if a
        # querystring value wasn't specified
        if url.path and "db" not in kwargs:
            try:
                kwargs["db"] = int(unquote(url.path).replace("/", ""))
            except (AttributeError, ValueError):
                pass

        if url.scheme == "rediss":
            kwargs["connection_class"] = SSLConnection

    return kwargs


class ConnectionPool:
    """
    Create a connection pool. ``If max_connections`` is set, then this
    object raises :py:class:`~redis.exceptions.ConnectionError` when the pool's
    limit is reached.

    By default, TCP connections are created unless ``connection_class``
    is specified. Use class:`.UnixDomainSocketConnection` for
    unix sockets.

    Any additional keyword arguments are passed to the constructor of
    ``connection_class``.
    """

    @classmethod
    def from_url(cls, url, **kwargs):
        """
        Return a connection pool configured from the given URL.

        For example::

            redis://[[username]:[password]]@localhost:6379/0
            rediss://[[username]:[password]]@localhost:6379/0
            unix://[username@]/path/to/socket.sock?db=0[&password=password]

        Three URL schemes are supported:

        - `redis://` creates a TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/redis>
        - `rediss://` creates a SSL wrapped TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/rediss>
        - ``unix://``: creates a Unix Domain Socket connection.

        The username, password, hostname, path and all querystring values
        are passed through urllib.parse.unquote in order to replace any
        percent-encoded values with their corresponding characters.

        There are several ways to specify a database number. The first value
        found will be used:

            1. A ``db`` querystring option, e.g. redis://localhost?db=0
            2. If using the redis:// or rediss:// schemes, the path argument
               of the url, e.g. redis://localhost/0
            3. A ``db`` keyword argument to this function.

        If none of these options are specified, the default db=0 is used.

        All querystring options are cast to their appropriate Python types.
        Boolean arguments can be specified with string values "True"/"False"
        or "Yes"/"No". Values that cannot be properly cast cause a
        ``ValueError`` to be raised. Once parsed, the querystring arguments
        and keyword arguments are passed to the ``ConnectionPool``'s
        class initializer. In the case of conflicting arguments, querystring
        arguments always win.
        """
        url_options = parse_url(url)

        if "connection_class" in kwargs:
            url_options["connection_class"] = kwargs["connection_class"]

        kwargs.update(url_options)
        return cls(**kwargs)

    def __init__(
        self, connection_class=Connection, max_connections=None, **connection_kwargs
    ):
        max_connections = max_connections or 2**31
        if not isinstance(max_connections, int) or max_connections < 0:
            raise ValueError('"max_connections" must be a positive integer')

        self.connection_class = connection_class
        self.connection_kwargs = connection_kwargs
        self.max_connections = max_connections

        # a lock to protect the critical section in _checkpid().
        # this lock is acquired when the process id changes, such as
        # after a fork. during this time, multiple threads in the child
        # process could attempt to acquire this lock. the first thread
        # to acquire the lock will reset the data structures and lock
        # object of this pool. subsequent threads acquiring this lock
        # will notice the first thread already did the work and simply
        # release the lock.
        self._fork_lock = threading.Lock()
        self.reset()

    def __repr__(self):
        return (
            f"{type(self).__name__}"
            f"<{repr(self.connection_class(**self.connection_kwargs))}>"
        )

    def reset(self):
        self._lock = threading.Lock()
        self._created_connections = 0
        self._available_connections = []
        self._in_use_connections = set()

        # this must be the last operation in this method. while reset() is
        # called when holding _fork_lock, other threads in this process
        # can call _checkpid() which compares self.pid and os.getpid() without
        # holding any lock (for performance reasons). keeping this assignment
        # as the last operation ensures that those other threads will also
        # notice a pid difference and block waiting for the first thread to
        # release _fork_lock. when each of these threads eventually acquire
        # _fork_lock, they will notice that another thread already called
        # reset() and they will immediately release _fork_lock and continue on.
        self.pid = os.getpid()

    def _checkpid(self):
        # _checkpid() attempts to keep ConnectionPool fork-safe on modern
        # systems. this is called by all ConnectionPool methods that
        # manipulate the pool's state such as get_connection() and release().
        #
        # _checkpid() determines whether the process has forked by comparing
        # the current process id to the process id saved on the ConnectionPool
        # instance. if these values are the same, _checkpid() simply returns.
        #
        # when the process ids differ, _checkpid() assumes that the process
        # has forked and that we're now running in the child process. the child
        # process cannot use the parent's file descriptors (e.g., sockets).
        # therefore, when _checkpid() sees the process id change, it calls
        # reset() in order to reinitialize the child's ConnectionPool. this
        # will cause the child to make all new connection objects.
        #
        # _checkpid() is protected by self._fork_lock to ensure that multiple
        # threads in the child process do not call reset() multiple times.
        #
        # there is an extremely small chance this could fail in the following
        # scenario:
        #   1. process A calls _checkpid() for the first time and acquires
        #      self._fork_lock.
        #   2. while holding self._fork_lock, process A forks (the fork()
        #      could happen in a different thread owned by process A)
        #   3. process B (the forked child process) inherits the
        #      ConnectionPool's state from the parent. that state includes
        #      a locked _fork_lock. process B will not be notified when
        #      process A releases the _fork_lock and will thus never be
        #      able to acquire the _fork_lock.
        #
        # to mitigate this possible deadlock, _checkpid() will only wait 5
        # seconds to acquire _fork_lock. if _fork_lock cannot be acquired in
        # that time it is assumed that the child is deadlocked and a
        # redis.ChildDeadlockedError error is raised.
        if self.pid != os.getpid():
            acquired = self._fork_lock.acquire(timeout=5)
            if not acquired:
                raise ChildDeadlockedError
            # reset() the instance for the new process if another thread
            # hasn't already done so
            try:
                if self.pid != os.getpid():
                    self.reset()
            finally:
                self._fork_lock.release()

    def get_connection(self, command_name, *keys, **options):
        "Get a connection from the pool"
        self._checkpid()
        with self._lock:
            try:
                connection = self._available_connections.pop()
            except IndexError:
                connection = self.make_connection()
            self._in_use_connections.add(connection)

        try:
            # ensure this connection is connected to Redis
            connection.connect()
            # connections that the pool provides should be ready to send
            # a command. if not, the connection was either returned to the
            # pool before all data has been read or the socket has been
            # closed. either way, reconnect and verify everything is good.
            try:
                if connection.can_read():
                    raise ConnectionError("Connection has data")
            except (ConnectionError, OSError):
                connection.disconnect()
                connection.connect()
                if connection.can_read():
                    raise ConnectionError("Connection not ready")
        except BaseException:
            # release the connection back to the pool so that we don't
            # leak it
            self.release(connection)
            raise

        return connection

    def get_encoder(self):
        "Return an encoder based on encoding settings"
        kwargs = self.connection_kwargs
        return Encoder(
            encoding=kwargs.get("encoding", "utf-8"),
            encoding_errors=kwargs.get("encoding_errors", "strict"),
            decode_responses=kwargs.get("decode_responses", False),
        )

    def make_connection(self):
        "Create a new connection"
        if self._created_connections >= self.max_connections:
            raise ConnectionError("Too many connections")
        self._created_connections += 1
        return self.connection_class(**self.connection_kwargs)

    def release(self, connection):
        "Releases the connection back to the pool"
        self._checkpid()
        with self._lock:
            try:
                self._in_use_connections.remove(connection)
            except KeyError:
                # Gracefully fail when a connection is returned to this pool
                # that the pool doesn't actually own
                pass

            if self.owns_connection(connection):
                self._available_connections.append(connection)
            else:
                # pool doesn't own this connection. do not add it back
                # to the pool and decrement the count so that another
                # connection can take its place if needed
                self._created_connections -= 1
                connection.disconnect()
                return

    def owns_connection(self, connection):
        return connection.pid == self.pid

    def disconnect(self, inuse_connections=True):
        """
        Disconnects connections in the pool

        If ``inuse_connections`` is True, disconnect connections that are
        current in use, potentially by other threads. Otherwise only disconnect
        connections that are idle in the pool.
        """
        self._checkpid()
        with self._lock:
            if inuse_connections:
                connections = chain(
                    self._available_connections, self._in_use_connections
                )
            else:
                connections = self._available_connections

            for connection in connections:
                connection.disconnect()

    def set_retry(self, retry: "Retry") -> None:
        self.connection_kwargs.update({"retry": retry})
        for conn in self._available_connections:
            conn.retry = retry
        for conn in self._in_use_connections:
            conn.retry = retry


class BlockingConnectionPool(ConnectionPool):
    """
    Thread-safe blocking connection pool::

        >>> from redis.client import Redis
        >>> client = Redis(connection_pool=BlockingConnectionPool())

    It performs the same function as the default
    :py:class:`~redis.ConnectionPool` implementation, in that,
    it maintains a pool of reusable connections that can be shared by
    multiple redis clients (safely across threads if required).

    The difference is that, in the event that a client tries to get a
    connection from the pool when all of connections are in use, rather than
    raising a :py:class:`~redis.ConnectionError` (as the default
    :py:class:`~redis.ConnectionPool` implementation does), it
    makes the client wait ("blocks") for a specified number of seconds until
    a connection becomes available.

    Use ``max_connections`` to increase / decrease the pool size::

        >>> pool = BlockingConnectionPool(max_connections=10)

    Use ``timeout`` to tell it either how many seconds to wait for a connection
    to become available, or to block forever:

        >>> # Block forever.
        >>> pool = BlockingConnectionPool(timeout=None)

        >>> # Raise a ``ConnectionError`` after five seconds if a connection is
        >>> # not available.
        >>> pool = BlockingConnectionPool(timeout=5)
    """

    def __init__(
        self,
        max_connections=50,
        timeout=20,
        connection_class=Connection,
        queue_class=LifoQueue,
        **connection_kwargs,
    ):

        self.queue_class = queue_class
        self.timeout = timeout
        super().__init__(
            connection_class=connection_class,
            max_connections=max_connections,
            **connection_kwargs,
        )

    def reset(self):
        # Create and fill up a thread safe queue with ``None`` values.
        self.pool = self.queue_class(self.max_connections)
        while True:
            try:
                self.pool.put_nowait(None)
            except Full:
                break

        # Keep a list of actual connection instances so that we can
        # disconnect them later.
        self._connections = []

        # this must be the last operation in this method. while reset() is
        # called when holding _fork_lock, other threads in this process
        # can call _checkpid() which compares self.pid and os.getpid() without
        # holding any lock (for performance reasons). keeping this assignment
        # as the last operation ensures that those other threads will also
        # notice a pid difference and block waiting for the first thread to
        # release _fork_lock. when each of these threads eventually acquire
        # _fork_lock, they will notice that another thread already called
        # reset() and they will immediately release _fork_lock and continue on.
        self.pid = os.getpid()

    def make_connection(self):
        "Make a fresh connection."
        connection = self.connection_class(**self.connection_kwargs)
        self._connections.append(connection)
        return connection

    def get_connection(self, command_name, *keys, **options):
        """
        Get a connection, blocking for ``self.timeout`` until a connection
        is available from the pool.

        If the connection returned is ``None`` then creates a new connection.
        Because we use a last-in first-out queue, the existing connections
        (having been returned to the pool after the initial ``None`` values
        were added) will be returned before ``None`` values. This means we only
        create new connections when we need to, i.e.: the actual number of
        connections will only increase in response to demand.
        """
        # Make sure we haven't changed process.
        self._checkpid()

        # Try and get a connection from the pool. If one isn't available within
        # self.timeout then raise a ``ConnectionError``.
        connection = None
        try:
            connection = self.pool.get(block=True, timeout=self.timeout)
        except Empty:
            # Note that this is not caught by the redis client and will be
            # raised unless handled by application code. If you want never to
            raise ConnectionError("No connection available.")

        # If the ``connection`` is actually ``None`` then that's a cue to make
        # a new connection to add to the pool.
        if connection is None:
            connection = self.make_connection()

        try:
            # ensure this connection is connected to Redis
            connection.connect()
            # connections that the pool provides should be ready to send
            # a command. if not, the connection was either returned to the
            # pool before all data has been read or the socket has been
            # closed. either way, reconnect and verify everything is good.
            try:
                if connection.can_read():
                    raise ConnectionError("Connection has data")
            except (ConnectionError, OSError):
                connection.disconnect()
                connection.connect()
                if connection.can_read():
                    raise ConnectionError("Connection not ready")
        except BaseException:
            # release the connection back to the pool so that we don't leak it
            self.release(connection)
            raise

        return connection

    def release(self, connection):
        "Releases the connection back to the pool."
        # Make sure we haven't changed process.
        self._checkpid()
        if not self.owns_connection(connection):
            # pool doesn't own this connection. do not add it back
            # to the pool. instead add a None value which is a placeholder
            # that will cause the pool to recreate the connection if
            # its needed.
            connection.disconnect()
            self.pool.put_nowait(None)
            return

        # Put the connection back into the pool.
        try:
            self.pool.put_nowait(connection)
        except Full:
            # perhaps the pool has been reset() after a fork? regardless,
            # we don't want this connection
            pass

    def disconnect(self):
        "Disconnects all connections in the pool."
        self._checkpid()
        for connection in self._connections:
            connection.disconnect()
