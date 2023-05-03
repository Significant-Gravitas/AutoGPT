"""
    https://stackoverflow.com/questions/49100806/
    pylint-and-subprocess-run-returning-exit-status-28
"""
import subprocess

cmd = " pylint src\\**\\*"
try:
    subprocComplete = subprocess.run(
        cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(subprocComplete.stdout.decode("utf-8"))
except subprocess.CalledProcessError as err:
    print(err.output.decode("utf-8"))
