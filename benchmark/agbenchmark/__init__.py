import pydevd_pycharm

pydevd_pycharm.settrace(
    "localhost", port=9739, stdoutToServer=True, stderrToServer=True
)
