from sys import stdout 
from threading import Thread
from itertools import cycle
from time import sleep


class Spinner:
    def __init__(self, message="Loading...", delay=0.1):
        self.spinner = cycle(['-', '/', '|', '\\'])
        self.delay = delay
        self.message = message
        self.running = False
        self.spinner_thread = None

    def spin(self):
        while self.running:
            stdout.write(next(self.spinner) + " " + self.message + "\r")
            stdout.flush()
            sleep(self.delay)
            stdout.write('\b' * (len(self.message) + 2))

    def __enter__(self):
        self.running = True
        self.spinner_thread = Thread(target=self.spin)
        self.spinner_thread.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.running = False
        self.spinner_thread.join()
        stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
        stdout.flush()
