import sys
import threading
import itertools
import time


class Spinner:
    """一个简单的旋转图标类"""
    def __init__(self, message="Loading...", delay=0.1):
        """初始化旋转图标类"""
        self.spinner = itertools.cycle(['-', '/', '|', '\\'])
        self.delay = delay
        self.message = message
        self.running = False
        self.spinner_thread = None

    def spin(self):
        """旋转图标"""
        while self.running:
            sys.stdout.write(next(self.spinner) + " " + self.message + "\r")
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')

    def __enter__(self):
        """开始旋转图标"""
        self.running = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """停止旋转图标"""
        self.running = False
        self.spinner_thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
        sys.stdout.flush()
