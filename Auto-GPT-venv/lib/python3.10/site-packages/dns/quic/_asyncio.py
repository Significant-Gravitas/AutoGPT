# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

import asyncio
import socket
import ssl
import struct
import time

import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import aioquic.quic.events  # type: ignore
import dns.inet
import dns.asyncbackend

from dns.quic._common import (
    BaseQuicStream,
    AsyncQuicConnection,
    AsyncQuicManager,
    QUIC_MAX_DATAGRAM,
)


class AsyncioQuicStream(BaseQuicStream):
    def __init__(self, connection, stream_id):
        super().__init__(connection, stream_id)
        self._wake_up = asyncio.Condition()

    async def _wait_for_wake_up(self):
        async with self._wake_up:
            await self._wake_up.wait()

    async def wait_for(self, amount, expiration):
        timeout = self._timeout_from_expiration(expiration)
        while True:
            if self._buffer.have(amount):
                return
            self._expecting = amount
            try:
                await asyncio.wait_for(self._wait_for_wake_up(), timeout)
            except Exception:
                pass
            self._expecting = 0

    async def receive(self, timeout=None):
        expiration = self._expiration_from_timeout(timeout)
        await self.wait_for(2, expiration)
        (size,) = struct.unpack("!H", self._buffer.get(2))
        await self.wait_for(size, expiration)
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


class AsyncioQuicConnection(AsyncQuicConnection):
    def __init__(self, connection, address, port, source, source_port, manager=None):
        super().__init__(connection, address, port, source, source_port, manager)
        self._socket = None
        self._handshake_complete = asyncio.Event()
        self._socket_created = asyncio.Event()
        self._wake_timer = asyncio.Condition()
        self._receiver_task = None
        self._sender_task = None

    async def _receiver(self):
        try:
            af = dns.inet.af_for_address(self._address)
            backend = dns.asyncbackend.get_backend("asyncio")
            self._socket = await backend.make_socket(
                af, socket.SOCK_DGRAM, 0, self._source, self._peer
            )
            self._socket_created.set()
            async with self._socket:
                while not self._done:
                    (datagram, address) = await self._socket.recvfrom(
                        QUIC_MAX_DATAGRAM, None
                    )
                    if address[0] != self._peer[0] or address[1] != self._peer[1]:
                        continue
                    self._connection.receive_datagram(
                        datagram, self._peer[0], time.time()
                    )
                    # Wake up the timer in case the sender is sleeping, as there may be
                    # stuff to send now.
                    async with self._wake_timer:
                        self._wake_timer.notify_all()
        except Exception:
            pass

    async def _wait_for_wake_timer(self):
        async with self._wake_timer:
            await self._wake_timer.wait()

    async def _sender(self):
        await self._socket_created.wait()
        while not self._done:
            datagrams = self._connection.datagrams_to_send(time.time())
            for (datagram, address) in datagrams:
                assert address == self._peer[0]
                await self._socket.sendto(datagram, self._peer, None)
            (expiration, interval) = self._get_timer_values()
            try:
                await asyncio.wait_for(self._wait_for_wake_timer(), interval)
            except Exception:
                pass
            self._handle_timer(expiration)
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
                self._receiver_task.cancel()
            count += 1
            if count > 10:
                # yield
                count = 0
                await asyncio.sleep(0)

    async def write(self, stream, data, is_end=False):
        self._connection.send_stream_data(stream, data, is_end)
        async with self._wake_timer:
            self._wake_timer.notify_all()

    def run(self):
        if self._closed:
            return
        self._receiver_task = asyncio.Task(self._receiver())
        self._sender_task = asyncio.Task(self._sender())

    async def make_stream(self):
        await self._handshake_complete.wait()
        stream_id = self._connection.get_next_available_stream_id(False)
        stream = AsyncioQuicStream(self, stream_id)
        self._streams[stream_id] = stream
        return stream

    async def close(self):
        if not self._closed:
            self._manager.closed(self._peer[0], self._peer[1])
            self._closed = True
            self._connection.close()
            async with self._wake_timer:
                self._wake_timer.notify_all()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass


class AsyncioQuicManager(AsyncQuicManager):
    def __init__(self, conf=None, verify_mode=ssl.CERT_REQUIRED):
        super().__init__(conf, verify_mode, AsyncioQuicConnection)

    def connect(self, address, port=853, source=None, source_port=0):
        (connection, start) = self._connect(address, port, source, source_port)
        if start:
            connection.run()
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
