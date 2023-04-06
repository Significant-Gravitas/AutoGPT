import sys
import threading
import itertools
import time


class Spinner:
    def __init__(self, message="Loading...", delay=0.1):
        self.spinner = itertools.cycle(['-', '/', '|', '\\'])
        self.delay = delay
        self.message = message
        self.running = False
        self.spinner_thread = None

    def spin(self):
        while self.running:
            sys.stdout.write(next(self.spinner) + " " + self.message + "\r")
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b' * (len(self.message) + 2))

    def __enter__(self):
        self.running = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.running = False
        self.spinner_thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
        sys.stdout.flush()
