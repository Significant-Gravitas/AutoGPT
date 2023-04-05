"""Exception instance for AXScript servers.

This module implements an exception instance that is raised by the core 
server scripting support.

When a script error occurs, it wraps the COM object that describes the
exception in a Python instance, which can then be raised and caught.
"""


class Exception:
    def __init__(self, activeScriptError):
        self.activeScriptError = activeScriptError

    def __getattr__(self, attr):
        return getattr(self.activeScriptError, attr)
