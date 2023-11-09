# General constants for the debugger

DBGSTATE_NOT_DEBUGGING = 0
DBGSTATE_RUNNING = 1
DBGSTATE_BREAK = 2
DBGSTATE_QUITTING = 3  # Attempting to back out of the debug session.

LINESTATE_CURRENT = 0x1  # This line is where we are stopped
LINESTATE_BREAKPOINT = 0x2  # This line is a breakpoint
LINESTATE_CALLSTACK = 0x4  # This line is in the callstack.

OPT_HIDE = "hide"
OPT_STOP_EXCEPTIONS = "stopatexceptions"

import win32api
import win32ui


def DoGetOption(optsDict, optName, default):
    optsDict[optName] = win32ui.GetProfileVal("Debugger Options", optName, default)


def LoadDebuggerOptions():
    opts = {}
    DoGetOption(opts, OPT_HIDE, 0)
    DoGetOption(opts, OPT_STOP_EXCEPTIONS, 1)
    return opts


def SaveDebuggerOptions(opts):
    for key, val in opts.items():
        win32ui.WriteProfileVal("Debugger Options", key, val)
