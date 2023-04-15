import itertools
import sys
import threading
import time


class Spinner:
    """A simple spinner class"""

    def __init__(self, message="Loading...", delay=0.1):
        """Initialize the spinner class"""
        self.spinner = itertools.cycle(["-", "/", "|", "\\"])
        self.delay = delay
        self.message = message
        self.running = False
        self.spinner_thread = None

    def spin(self):
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

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Stop the spinner"""
        self.running = False
        if self.spinner_thread is not None:
            self.spinner_thread.join()
        sys.stdout.write(f"\r{' ' * (len(self.message) + 2)}\r")
        sys.stdout.flush()
