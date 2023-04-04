# Kills a process by process name
#
# Uses the Performance Data Helper to locate the PID, then kills it.
# Will only kill the process if there is only one process of that name
# (eg, attempting to kill "Python.exe" will only work if there is only
# one Python.exe running.  (Note that the current process does not
# count - ie, if Python.exe is hosting this script, you can still kill
# another Python.exe (as long as there is only one other Python.exe)

# Really just a demo for the win32pdh(util) module, which allows you
# to get all sorts of information about a running process and many
# other aspects of your system.

import sys

import win32api
import win32con
import win32pdhutil


def killProcName(procname):
    # Change suggested by Dan Knierim, who found that this performed a
    # "refresh", allowing us to kill processes created since this was run
    # for the first time.
    try:
        win32pdhutil.GetPerformanceAttributes("Process", "ID Process", procname)
    except:
        pass

    pids = win32pdhutil.FindPerformanceAttributesByName(procname)

    # If _my_ pid in there, remove it!
    try:
        pids.remove(win32api.GetCurrentProcessId())
    except ValueError:
        pass

    if len(pids) == 0:
        result = "Can't find %s" % procname
    elif len(pids) > 1:
        result = "Found too many %s's - pids=`%s`" % (procname, pids)
    else:
        handle = win32api.OpenProcess(win32con.PROCESS_TERMINATE, 0, pids[0])
        win32api.TerminateProcess(handle, 0)
        win32api.CloseHandle(handle)
        result = ""

    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for procname in sys.argv[1:]:
            result = killProcName(procname)
            if result:
                print(result)
                print("Dumping all processes...")
                win32pdhutil.ShowAllProcesses()
            else:
                print("Killed %s" % procname)
    else:
        print("Usage: killProcName.py procname ...")
