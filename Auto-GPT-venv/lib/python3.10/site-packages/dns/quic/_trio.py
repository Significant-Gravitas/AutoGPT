# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

import socket
import ssl
import struct
import time

import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import aioquic.quic.events  # type: ignore
import trio

import dns.inet
from dns._asyncbackend import NullContext
from dns.quic._common import (
    BaseQuicStream,
    AsyncQuicConnection,
    AsyncQuicManager,
    QUIC_MAX_DATAGRAM,
)


class TrioQuicStream(BaseQuicStream):
    def __init__(self, connection, stream_id):
        super().__init__(connection, stream_id)
        self._wake_up = trio.Condition()

    async def wait_for(self, amount):
        while True:
            if self._buffer.have(amount):
                return
            self._expecting = amount
            async with self._wake_up:
                await self._wake_up.wait()
            self._expecting = 0

    async def receive(self, timeout=None):
        if timeout is None:
            context = NullContext(None)
        else:
            context = trio.move_on_after(timeout)
        with context:
            await self.wait_for(2)
            (size,) = struct.unpack("!H", self._buffer.get(2))
            await self.wait_for(size)
            return self._buffer.get(size)

    async def send(self, datagram, is_end=False):
        data = self._encapsulate(datagram)
        await self._connection.write(self._stream_id, data, is_end)

    async def _add_input(self, data, is_end):
        if self._common_add_input(data, is_end):
            async with self._wake_up:
                self._wake_up.notify()

    async def close(self):
        self._close()

    # Streams are async context managers

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        async with self._wake_up:
            self._wake_up.notify()
        return False


class TrioQuicConnection(AsyncQuicConnection):
    def __init__(self, connection, address, port, source, source_port, manager=None):
        super().__init__(connection, address, port, source, source_port, manager)
        self._socket = trio.socket.socket(self._af, socket.SOCK_DGRAM, 0)
        if self._source:
            trio.socket.bind(dns.inet.low_level_address_tuple(self._source, self._af))
        self._handshake_complete = trio.Event()
        self._run_done = trio.Event()
        self._worker_scope = None

    async def _worker(self):
        await self._socket.connect(self._peer)
        while not self._done:
            (expiration, interval) = self._get_timer_values(False)
            with trio.CancelScope(
                deadline=trio.current_time() + interval
            ) as self._worker_scope:
                datagram = await self._socket.recv(QUIC_MAX_DATAGRAM)
                self._connection.receive_datagram(datagram, self._peer[0], time.time())
            self._worker_scope = None
            self._handle_timer(expiration)
            datagrams = self._connection.datagrams_to_send(time.time())
            for (datagram, _) in datagrams:
                await self._socket.send(datagram)
            await self._handle_events()

    async def _handle_events(self):
        count = 0
        while True:
            event = self._connection.next_event()
            if event is None:
                return
            if isinstance(event, aioquic.quic.events.StreamDataReceived):
                stream = self._streams.get(event.stream_id)
                if stream:
                    await stream._add_input(event.data, event.end_stream)
            elif isinstance(event, aioquic.quic.events.HandshakeCompleted):
                self._handshake_complete.set()
            elif isinstance(
                event, aioquic.quic.events.ConnectionTerminated
            ) or isinstance(event, aioquic.quic.events.StreamReset):
                self._done = True
                self._socket.close()
            count += 1
            if count > 10:
                # yield
                count = 0
                await trio.sleep(0)

    async def write(self, stream, data, is_end=False):
        self._connection.send_stream_data(stream, data, is_end)
        if self._worker_scope is not None:
            self._worker_scope.cancel()

    async def run(self):
        if self._closed:
            return
        async with trio.open_nursery() as nursery:
            nursery.start_soon(self._worker)
        self._run_done.set()

    async def make_stream(self):
        await self._handshake_complete.wait()
        stream_id = self._connection.get_next_available_stream_id(False)
        stream = TrioQuicStream(self, stream_id)
        self._streams[stream_id] = stream
        return stream

    async def close(self):
        if not self._closed:
            self._manager.closed(self._peer[0], self._peer[1])
            self._closed = True
            self._connection.close()
            if self._worker_scope is not None:
                self._worker_scope.cancel()
            await self._run_done.wait()


class TrioQuicManager(AsyncQuicManager):
    def __init__(self, nursery, conf=None, verify_mode=ssl.CERT_REQUIRED):
        super().__init__(conf, verify_mode, TrioQuicConnection)
        self._nursery = nursery

    def connect(self, address, port=853, source=None, source_port=0):
        (connection, start) = self._connect(address, port, source, source_port)
        if start:
            self._nursery.start_soon(connection.run)
        return connection

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Copy the itertor into a list as exiting things will mutate the connections
        # table.
        connections = list(self._connections.values())
        for connection in connections:
            await connection.close()
        return False
