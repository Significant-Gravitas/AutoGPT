import asyncio
import logging
import weakref

from ._asyncio_loop import get_running_loop, get_task_loop


class StreamSink:
    def __init__(self, stream):
        self._stream = stream
        self._flushable = callable(getattr(stream, "flush", None))
        self._stoppable = callable(getattr(stream, "stop", None))
        self._completable = asyncio.iscoroutinefunction(getattr(stream, "complete", None))

    def write(self, message):
        self._stream.write(message)
        if self._flushable:
            self._stream.flush()

    def stop(self):
        if self._stoppable:
            self._stream.stop()

    async def complete(self):
        if self._completable:
            await self._stream.complete()


class StandardSink:
    def __init__(self, handler):
        self._handler = handler

    def write(self, message):
        record = message.record
        message = str(message)
        exc = record["exception"]
        record = logging.getLogger().makeRecord(
            record["name"],
            record["level"].no,
            record["file"].path,
            record["line"],
            message,
            (),
            (exc.type, exc.value, exc.traceback) if exc else None,
            record["function"],
            {"extra": record["extra"]},
        )
        if exc:
            record.exc_text = "\n"
        self._handler.handle(record)

    def stop(self):
        self._handler.close()

    async def complete(self):
        pass


class AsyncSink:
    def __init__(self, function, loop, error_interceptor):
        self._function = function
        self._loop = loop
        self._error_interceptor = error_interceptor
        self._tasks = weakref.WeakSet()

    def write(self, message):
        try:
            loop = self._loop or get_running_loop()
        except RuntimeError:
            return

        coroutine = self._function(message)
        task = loop.create_task(coroutine)

        def check_exception(future):
            if future.cancelled() or future.exception() is None:
                return
            if not self._error_interceptor.should_catch():
                raise future.exception()
            self._error_interceptor.print(message.record, exception=future.exception())

        task.add_done_callback(check_exception)
        self._tasks.add(task)

    def stop(self):
        for task in self._tasks:
            task.cancel()

    async def complete(self):
        loop = get_running_loop()
        for task in self._tasks:
            if get_task_loop(task) is loop:
                try:
                    await task
                except Exception:
                    pass  # Handled in "check_exception()"

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tasks"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tasks = weakref.WeakSet()


class CallableSink:
    def __init__(self, function):
        self._function = function

    def write(self, message):
        self._function(message)

    def stop(self):
        pass

    async def complete(self):
        pass
