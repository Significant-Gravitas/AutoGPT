"""A simple spinner module"""
import itertools
import sys
import threading
import time


class Spinner:
    """A simple spinner class"""

    def __init__(self, message: str = "Loading...", delay: float = 0.1) -> None:
        """Initialize the spinner class

        Args:
            message (str): The message to display.
            delay (float): The delay between each spinner update.
        """
        self.spinner = itertools.cycle(["-", "/", "|", "\\"])
        self.delay = delay
        self.message = message
        self.running = False
        self.spinner_thread = None

    def spin(self) -> None:
        """Spin the spinner"""
        while self.running:
            sys.stdout.write(f"{next(self.spinner)} {self.message}\r")
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write(f"\r{' ' * (len(self.message) + 2)}\r")

    def __enter__(self):
        """Start the spinner"""
        self.running = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.start()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """Stop the spinner

        Args:
            exc_type (Exception): The exception type.
            exc_value (Exception): The exception value.
            exc_traceback (Exception): The exception traceback.
        """
        self.running = False
        if self.spinner_thread is not None:
            self.spinner_thread.join()
        sys.stdout.write(f"\r{' ' * (len(self.message) + 2)}\r")
        sys.stdout.flush()

    def update_message(self, new_message, delay=0.1):
        """Update the spinner message
        Args:
            new_message (str): New message to display.
            delay (float): The delay in seconds between each spinner update.
        """
        time.sleep(delay)
        sys.stdout.write(
            f"\r{' ' * (len(self.message) + 2)}\r"
        )  # Clear the current message
        sys.stdout.flush()
        self.message = new_message
