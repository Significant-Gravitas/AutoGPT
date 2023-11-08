import platform
import queue
import select
import subprocess
import time
from threading import Thread
from typing import Any

import psutil


def run_linux_env(process: Any, start_time: float, timeout: float) -> None:
    while True:
        try:
            # This checks if there's data to be read from stdout without blocking.
            if process.stdout and select.select([process.stdout], [], [], 0)[0]:
                output = process.stdout.readline()
                print(output.strip())
        except Exception as e:
            continue

        # Check if process has ended, has no more output, or exceeded timeout
        if process.poll() is not None or (time.time() - start_time > timeout):
            break

    if time.time() - start_time > timeout:
        print("The Python function has exceeded the time limit and was terminated.")
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()

    else:
        print("The Python function has finished running.")


def enqueue_output(out: Any, my_queue: Any) -> None:
    for line in iter(out.readline, b""):
        my_queue.put(line)
    out.close()


def run_windows_env(process: Any, start_time: float, timeout: float) -> None:
    my_queue: Any = queue.Queue()
    thread = Thread(target=enqueue_output, args=(process.stdout, my_queue))
    thread.daemon = True
    thread.start()

    while True:
        try:
            output = my_queue.get_nowait().strip()
            print(output)
        except queue.Empty:
            pass

        if process.poll() is not None or (time.time() - start_time > timeout):
            break

    if time.time() - start_time > timeout:
        print("The Python function has exceeded the time limit and was terminated.")
        process.terminate()


def execute_subprocess(command, timeout):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    start_time = time.time()
    if platform.system() == "Windows":
        run_windows_env(process, start_time, timeout)
    else:
        run_linux_env(process, start_time, timeout)
    process.wait()
    if process.returncode != 0:
        print(f"The agent timed out")
