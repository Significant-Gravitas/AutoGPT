# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

import socket
import ssl
import selectors
import struct
import threading
import time

import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import aioquic.quic.events  # type: ignore
import dns.inet

from dns.quic._common import (
    BaseQuicStream,
    BaseQuicConnection,
    BaseQuicManager,
    QUIC_MAX_DATAGRAM,
)

# Avoid circularity with dns.query
if hasattr(selectors, "PollSelector"):
    _selector_class = selectors.PollSelector  # type: ignore
else:
    _selector_class = selectors.SelectSelector  # type: ignore


class SyncQuicStream(BaseQuicStream):
    def __init__(self, connection, stream_id):
        super().__init__(connection, stream_id)
        self._wake_up = threading.Condition()
        self._lock = threading.Lock()

    def wait_for(self, amount, expiration):
        timeout = self._timeout_from_expiration(expiration)
        while True:
            with self._lock:
                if self._buffer.have(amount):
                    return
                self._expecting = amount
            with self._wake_up:
                self._wake_up.wait(timeout)
            self._expecting = 0

    def receive(self, timeout=None):
        expiration = self._expiration_from_timeout(timeout)
        self.wait_for(2, expiration)
        with self._lock:
            (size,) = struct.unpack("!H", self._buffer.get(2))
        self.wait_for(size, expiration)
        with self._lock:
            return self._buffer.get(size)

    def send(self, datagram, is_end=False):
        data = self._encapsulate(datagram)
        self._connection.write(self._stream_id, data, is_end)

    def _add_input(self, data, is_end):
        if self._common_add_input(data, is_end):
            with self._wake_up:
                self._wake_up.notify()

    def close(self):
        with self._lock:
            self._close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        with self._wake_up:
            self._wake_up.notify()
        return False


class SyncQuicConnection(BaseQuicConnection):
    def __init__(self, connection, address, port, source, source_port, manager):
        super().__init__(connection, address, port, source, source_port, manager)
        self._socket = socket.socket(self._af, socket.SOCK_DGRAM, 0)
        self._socket.connect(self._peer)
        (self._send_wakeup, self._receive_wakeup) = socket.socketpair()
        self._receive_wakeup.setblocking(False)
        self._socket.setblocking(False)
        if self._source is not None:
            try:
                self._socket.bind(
                    dns.inet.low_level_address_tuple(self._source, self._af)
                )
            except Exception:
                self._socket.close()
                raise
        self._handshake_complete = threading.Event()
        self._worker_thread = None
        self._lock = threading.Lock()

    def _read(self):
        count = 0
        while count < 10:
            count += 1
            try:
                datagram = self._socket.recv(QUIC_MAX_DATAGRAM)
            except BlockingIOError:
                return
            with self._lock:
                self._connection.receive_datagram(datagram, self._peer[0], time.time())

    def _drain_wakeup(self):
        while True:
            try:
                self._receive_wakeup.recv(32)
            except BlockingIOError:
                return

    def _worker(self):
        sel = _selector_class()
        sel.register(self._socket, selectors.EVENT_READ, self._read)
        sel.register(self._receive_wakeup, selectors.EVENT_READ, self._drain_wakeup)
        while not self._done:
            (expiration, interval) = self._get_timer_values(False)
            items = sel.select(interval)
            for (key, _) in items:
                key.data()
            with self._lock:
                self._handle_timer(expiration)
                datagrams = self._connection.datagrams_to_send(time.time())
            for (datagram, _) in datagrams:
                try:
                    self._socket.send(datagram)
                except BlockingIOError:
                    # we let QUIC handle any lossage
                    pass
            self._handle_events()

    def _handle_events(self):
        while True:
            with self._lock:
                event = self._connection.next_event()
            if event is None:
                return
            if isinstance(event, aioquic.quic.events.StreamDataReceived):
                with self._lock:
                    stream = self._streams.get(event.stream_id)
                if stream:
                    stream._add_input(event.data, event.end_stream)
            elif isinstance(event, aioquic.quic.events.HandshakeCompleted):
                self._handshake_complete.set()
            elif isinstance(
                event, aioquic.quic.events.ConnectionTerminated
            ) or isinstance(event, aioquic.quic.events.StreamReset):
                with self._lock:
                    self._done = True

    def write(self, stream, data, is_end=False):
        with self._lock:
            self._connection.send_stream_data(stream, data, is_end)
        self._send_wakeup.send(b"\x01")

    def run(self):
        if self._closed:
            return
        self._worker_thread = threading.Thread(target=self._worker)
        self._worker_thread.start()

    def make_stream(self):
        self._handshake_complete.wait()
        with self._lock:
            stream_id = self._connection.get_next_available_stream_id(False)
            stream = SyncQuicStream(self, stream_id)
            self._streams[stream_id] = stream
        return stream

    def close_stream(self, stream_id):
        with self._lock:
            super().close_stream(stream_id)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._manager.closed(self._peer[0], self._peer[1])
            self._closed = True
            self._connection.close()
            self._send_wakeup.send(b"\x01")
        self._worker_thread.join()


class SyncQuicManager(BaseQuicManager):
    def __init__(self, conf=None, verify_mode=ssl.CERT_REQUIRED):
        super().__init__(conf, verify_mode, SyncQuicConnection)
        self._lock = threading.Lock()

    def connect(self, address, port=853, source=None, source_port=0):
        with self._lock:
            (connection, start) = self._connect(address, port, source, source_port)
            if start:
                connection.run()
            return connection

    def closed(self, address, port):
        with self._lock:
            super().closed(address, port)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Copy the itertor into a list as exiting things will mutate the connections
        # table.
        connections = list(self._connections.values())
        for connection in connections:
            connection.close()
        return False
