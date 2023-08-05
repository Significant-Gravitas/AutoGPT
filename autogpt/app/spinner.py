"""A simple spinner module"""
import io
import itertools
import os
import select
import sys
import threading
import types
from typing import Any, Callable, List, Optional

if os.name == "nt":
    import keyboard


class SpinnerInterrupted(Exception):
    pass


class RaisingThread(threading.Thread):
    def run(self) -> None:
        self._exc = None
        try:
            super().run()
        except Exception as e:
            self._exc = e

    def join(self, timeout=None) -> None:  # type: ignore
        super().join(timeout=timeout)
        if self._exc:
            raise self._exc


class Spinner:
    """A simple spinner class"""

    def __init__(
        self,
        message: str = "Loading...",
        delay: float = 0.1,
        plain_output: bool = False,
        interruptable: bool = False,
        on_soft_interrupt: Optional[Callable] = None,
    ) -> None:
        """Initialize the spinner class

        Args:
            message (str): The message to display.
            delay (float): The delay between each spinner update.
            plain_output (bool): Whether to display the spinner or not.
        """
        self.plain_output = plain_output
        self.spinner = itertools.cycle(["-", "/", "|", "\\"])
        self.delay = delay
        self.message = message
        self.running = False
        self.spinner_thread: Optional[RaisingThread] = None
        self.interruptable = interruptable
        self.on_soft_interrupt = on_soft_interrupt
        self.ended = threading.Event()

    def spin(self) -> None:
        """Spin the spinner"""

        def key_pressed(
            pressed_key: Optional[str],
            key_to_check: Optional[str],
            keyboard_key: Optional[str] = None,
        ) -> bool:
            if keyboard_key is None:
                keyboard_key = key_to_check
            if (
                pressed_key
                and pressed_key == key_to_check
                or (os.name == "nt" and keyboard.is_pressed(keyboard_key))
            ):
                return True
            return False

        if self.plain_output:
            self.print_message()
            return
        while self.running:
            self.print_message()
            if self.interruptable:
                if (self.oldtty) and sys.stdin in select.select(
                    [sys.stdin], [], [], 0.0
                )[0]:
                    key = sys.stdin.read(1)
                else:
                    key = None
                if key_pressed(key, " ", "space"):
                    if self.on_soft_interrupt is not None:
                        self.on_soft_interrupt()
                elif key_pressed(key, "q"):
                    self.ended.set()
                    raise SpinnerInterrupted("Spinner interrupted")

            if self.ended.wait(self.delay):
                break

    def print_message(self) -> None:
        sys.stdout.write(f"\r{' ' * (len(self.message) + 2)}\r")
        sys.stdout.write(f"{next(self.spinner)} {self.message}\r")
        sys.stdout.flush()

    def start(self) -> None:
        self.ended.clear()
        self.running = True
        try:
            import tty

            self.oldtty: Optional[List[Any]] = None

            self.stdin_no = sys.stdin.fileno()
            tty.setcbreak(self.stdin_no)
        except io.UnsupportedOperation:
            self.oldtty = None  #
        except ModuleNotFoundError:
            self.oldtty = None
        except:
            self.oldtty = None

        self.spinner_thread = RaisingThread(target=self.spin)
        self.spinner_thread.start()

    def stop(self) -> None:
        self.ended.set()
        try:
            if self.spinner_thread is not None:
                self.spinner_thread.join()
            self.running = False
            if self.spinner_thread is not None:
                self.spinner_thread.join()
            sys.stdout.write(f"\r{' ' * (len(self.message) + 2)}\r")
            sys.stdout.flush()
        finally:
            if self.oldtty:
                while sys.stdin in select.select([sys.stdin], [], [], 0.0)[0]:
                    sys.stdin.read(1)

    def __enter__(self) -> "Spinner":
        """Start the spinner"""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        exc_traceback: Optional[types.TracebackType] = None,
    ) -> None:
        """Stop the spinner

        Args:
            exc_type (type[BaseException]): The exception type.
            exc_value (BaseException): The exception value.
            exc_traceback (types.TracebackType): The exception traceback.
        """
        self.stop()
