"""Share imported test-service code through the existing forkserver."""

import gc
import importlib
import os
import socket

importlib.import_module("pytest")
importlib.import_module("backend.util.test")
importlib.import_module("backend.blocks").load_all_blocks()

# Importing the backend sets a default socket timeout
# (autogpt_libs/logging/config.py). From 3.14 the forkserver authenticates each
# request by reading the socket it accepted as a blocking file, and a socket
# accepted under a default timeout is non-blocking, so the forkserver dies on
# its first request. It serves without one, and each service it forks gets the
# backend's timeout back.
_service_socket_timeout = socket.getdefaulttimeout()
socket.setdefaulttimeout(None)
os.register_at_fork(
    after_in_child=lambda: socket.setdefaulttimeout(_service_socket_timeout)
)

# Keep cyclic GC from dirtying inherited objects and defeating copy-on-write.
gc.collect()
gc.freeze()
