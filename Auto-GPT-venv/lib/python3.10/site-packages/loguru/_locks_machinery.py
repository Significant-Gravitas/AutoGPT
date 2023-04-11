import os
import threading
import weakref

if not hasattr(os, "register_at_fork"):

    def create_logger_lock():
        return threading.Lock()

    def create_handler_lock():
        return threading.Lock()

else:
    # While forking, we need to sanitize all locks to make sure the child process doesn't run into
    # a deadlock (if a lock already acquired is inherited) and to protect sink from corrupted state.
    # It's very important to acquire logger locks before handlers one to prevent possible deadlock
    # while 'remove()' is called for example.

    logger_locks = weakref.WeakSet()
    handler_locks = weakref.WeakSet()

    def acquire_locks():
        for lock in logger_locks:
            lock.acquire()

        for lock in handler_locks:
            lock.acquire()

    def release_locks():
        for lock in logger_locks:
            lock.release()

        for lock in handler_locks:
            lock.release()

    os.register_at_fork(
        before=acquire_locks,
        after_in_parent=release_locks,
        after_in_child=release_locks,
    )

    def create_logger_lock():
        lock = threading.Lock()
        logger_locks.add(lock)
        return lock

    def create_handler_lock():
        lock = threading.Lock()
        handler_locks.add(lock)
        return lock
