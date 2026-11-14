"""Windows job objects: a command and every process it starts, stopped
together.

TerminateProcess ends one process, not the ones it started, so a child that
outlived a command's CTRL_BREAK_EVENT ran on after the command was stopped.
A job holds the command and its descendants, those whose parent has exited
too, and TerminateJobObject ends them all. The command is created suspended
and runs only once it is in its job, so nothing it starts can get out by
starting first.

The job doesn't set JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. With it, closing the
job when a command is done would also end what the command started to
outlive it (`start "" server.exe`), which Linux and macOS leave running. And
it allows JOB_OBJECT_LIMIT_BREAKAWAY_OK: a program that asks for
CREATE_BREAKAWAY_FROM_JOB still starts, and leaves the job, as a POSIX one
may leave its process group.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# The CreateProcess flag that leaves a new process's first thread waiting to
# be resumed (ResumeThread). subprocess doesn't name it.
CREATE_SUSPENDED = 0x00000004

_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_THREAD_SUSPEND_RESUME = 0x0002
_TH32CS_SNAPTHREAD = 0x00000004
# JOBOBJECTINFOCLASS values.
_BASIC_ACCOUNTING_INFORMATION = 1
_BASIC_LIMIT_INFORMATION = 2
_JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
# ResumeThread's answer when it fails: (DWORD)-1.
_RESUME_FAILED = 0xFFFFFFFF
# The exit code TerminateJobObject gives each process, as Popen.kill() does.
_KILLED_EXIT_CODE = 1


class _ThreadEntry(ctypes.Structure):
    # THREADENTRY32 (tlhelp32.h).
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("cntUsage", ctypes.c_uint32),
        ("th32ThreadID", ctypes.c_uint32),
        ("th32OwnerProcessID", ctypes.c_uint32),
        ("tpBasePri", ctypes.c_int32),
        ("tpDeltaPri", ctypes.c_int32),
        ("dwFlags", ctypes.c_uint32),
    ]


class _BasicAccounting(ctypes.Structure):
    # JOBOBJECT_BASIC_ACCOUNTING_INFORMATION (winnt.h).
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    ]


class _BasicLimits(ctypes.Structure):
    # JOBOBJECT_BASIC_LIMIT_INFORMATION (winnt.h).
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


_kernel32_lib: Any = None


def _kernel32() -> Any:
    """kernel32, with the signatures of the calls used here."""
    if sys.platform != "win32":
        raise OSError("Windows only")
    global _kernel32_lib
    if _kernel32_lib is None:
        from ctypes import wintypes

        lib = ctypes.WinDLL("kernel32", use_last_error=True)
        handle, dword, boolean = wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL
        signatures: dict[str, tuple[list[Any], Any]] = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], handle),
            "SetInformationJobObject": ([handle, ctypes.c_int, ctypes.c_void_p, dword], boolean),
            "QueryInformationJobObject": (
                [handle, ctypes.c_int, ctypes.c_void_p, dword, ctypes.POINTER(dword)],
                boolean,
            ),
            "AssignProcessToJobObject": ([handle, handle], boolean),
            "TerminateJobObject": ([handle, wintypes.UINT], boolean),
            "OpenProcess": ([dword, boolean, dword], handle),
            "OpenThread": ([dword, boolean, dword], handle),
            "ResumeThread": ([handle], dword),
            "CreateToolhelp32Snapshot": ([dword, dword], handle),
            "Thread32First": ([handle, ctypes.c_void_p], boolean),
            "Thread32Next": ([handle, ctypes.c_void_p], boolean),
            "CloseHandle": ([handle], boolean),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(lib, name)
            function.argtypes = argtypes
            function.restype = restype
        _kernel32_lib = lib
    return _kernel32_lib


def _last_error() -> OSError:
    if sys.platform != "win32":
        return OSError("Windows only")
    return ctypes.WinError(ctypes.get_last_error())


class CommandJob:
    """The job object that holds one command's processes."""

    def __init__(self, handle: int) -> None:
        self._handle: int | None = handle

    def has_processes(self) -> bool:
        """Whether any process in the job is still running. True when that
        can't be read, so a stop goes ahead."""
        if self._handle is None:
            return False
        info = _BasicAccounting()
        if not _kernel32().QueryInformationJobObject(
            self._handle,
            _BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        ):
            return True
        return info.ActiveProcesses > 0

    def terminate(self) -> None:
        """End every process in the job."""
        if self._handle is None:
            return
        if not _kernel32().TerminateJobObject(self._handle, _KILLED_EXIT_CODE):
            raise _last_error()

    def close(self) -> None:
        """Let go of the job. Its processes keep running."""
        handle, self._handle = self._handle, None
        if handle is not None:
            _kernel32().CloseHandle(handle)


def run_in_job(pid: int) -> CommandJob | None:
    """Put process `pid`, created suspended (CREATE_SUSPENDED), in a job of
    its own, then let it run. None, with the process running, if the job
    couldn't be made. Raises OSError if the process couldn't be resumed: it
    is then still suspended, and the job, if one was made, is ended."""
    kernel32 = _kernel32()
    job: CommandJob | None = None
    try:
        job = _job_holding(kernel32, pid)
    except OSError as exc:
        logger.warning(
            "The command could not be put in a job of its own (%s): a stop ends it alone, "
            "without the processes it started",
            exc,
        )
    try:
        _resume(kernel32, pid)
    except OSError:
        if job is not None:
            try:
                job.terminate()
            finally:
                job.close()
        raise
    return job


def _job_holding(kernel32: Any, pid: int) -> CommandJob:
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise _last_error()
    job = CommandJob(handle)
    try:
        limits = _BasicLimits(LimitFlags=_JOB_OBJECT_LIMIT_BREAKAWAY_OK)
        if not kernel32.SetInformationJobObject(
            handle, _BASIC_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise _last_error()
        process = kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
        if not process:
            raise _last_error()
        try:
            if not kernel32.AssignProcessToJobObject(handle, process):
                raise _last_error()
        finally:
            kernel32.CloseHandle(process)
    except BaseException:
        job.close()
        raise
    return job


def _resume(kernel32: Any, pid: int) -> None:
    """Resume the threads of process `pid`: a process created suspended has
    one, which subprocess doesn't keep a handle to."""
    snapshot = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPTHREAD, 0)
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        raise _last_error()
    resumed = 0
    try:
        entry = _ThreadEntry()
        entry.dwSize = ctypes.sizeof(_ThreadEntry)
        more = kernel32.Thread32First(snapshot, ctypes.byref(entry))
        while more:
            if entry.th32OwnerProcessID == pid:
                thread = kernel32.OpenThread(_THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                if thread:
                    try:
                        if kernel32.ResumeThread(thread) != _RESUME_FAILED:
                            resumed += 1
                    finally:
                        kernel32.CloseHandle(thread)
            more = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    if not resumed:
        raise OSError(f"process {pid} could not be resumed")


__all__ = ["CREATE_SUSPENDED", "CommandJob", "run_in_job"]
