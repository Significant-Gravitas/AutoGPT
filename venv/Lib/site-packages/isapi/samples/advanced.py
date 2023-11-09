# This extension demonstrates some advanced features of the Python ISAPI
# framework.
# We demonstrate:
# * Reloading your Python module without shutting down IIS (eg, when your
#   .py implementation file changes.)
# * Custom command-line handling - both additional options and commands.
# * Using a query string - any part of the URL after a '?' is assumed to
#   be "variable names" separated by '&' - we will print the values of
#   these server variables.
# * If the tail portion of the URL is "ReportUnhealthy", IIS will be
#   notified we are unhealthy via a HSE_REQ_REPORT_UNHEALTHY request.
#   Whether this is acted upon depends on if the IIS health-checking
#   tools are installed, but you should always see the reason written
#   to the Windows event log - see the IIS documentation for more.

import os
import stat
import sys

from isapi import isapicon
from isapi.simple import SimpleExtension

if hasattr(sys, "isapidllhandle"):
    import win32traceutil

# Notes on reloading
# If your HttpFilterProc or HttpExtensionProc functions raises
# 'isapi.InternalReloadException', the framework will not treat it
# as an error but instead will terminate your extension, reload your
# extension module, re-initialize the instance, and re-issue the request.
# The Initialize functions are called with None as their param.  The
# return code from the terminate function is ignored.
#
# This is all the framework does to help you.  It is up to your code
# when you raise this exception.  This sample uses a Win32 "find
# notification".  Whenever windows tells us one of the files in the
# directory has changed, we check if the time of our source-file has
# changed, and set a flag.  Next imcoming request, we check the flag and
# raise the special exception if set.
#
# The end result is that the module is automatically reloaded whenever
# the source-file changes - you need take no further action to see your
# changes reflected in the running server.

# The framework only reloads your module - if you have libraries you
# depend on and also want reloaded, you must arrange for this yourself.
# One way of doing this would be to special case the import of these
# modules.  Eg:
# --
# try:
#    my_module = reload(my_module) # module already imported - reload it
# except NameError:
#    import my_module # first time around - import it.
# --
# When your module is imported for the first time, the NameError will
# be raised, and the module imported.  When the ISAPI framework reloads
# your module, the existing module will avoid the NameError, and allow
# you to reload that module.

import threading

import win32con
import win32event
import win32file
import winerror

from isapi import InternalReloadException

try:
    reload_counter += 1
except NameError:
    reload_counter = 0


# A watcher thread that checks for __file__ changing.
# When it detects it, it simply sets "change_detected" to true.
class ReloadWatcherThread(threading.Thread):
    def __init__(self):
        self.change_detected = False
        self.filename = __file__
        if self.filename.endswith("c") or self.filename.endswith("o"):
            self.filename = self.filename[:-1]
        self.handle = win32file.FindFirstChangeNotification(
            os.path.dirname(self.filename),
            False,  # watch tree?
            win32con.FILE_NOTIFY_CHANGE_LAST_WRITE,
        )
        threading.Thread.__init__(self)

    def run(self):
        last_time = os.stat(self.filename)[stat.ST_MTIME]
        while 1:
            try:
                rc = win32event.WaitForSingleObject(self.handle, win32event.INFINITE)
                win32file.FindNextChangeNotification(self.handle)
            except win32event.error as details:
                # handle closed - thread should terminate.
                if details.winerror != winerror.ERROR_INVALID_HANDLE:
                    raise
                break
            this_time = os.stat(self.filename)[stat.ST_MTIME]
            if this_time != last_time:
                print("Detected file change - flagging for reload.")
                self.change_detected = True
                last_time = this_time

    def stop(self):
        win32file.FindCloseChangeNotification(self.handle)


# The ISAPI extension - handles requests in our virtual dir, and sends the
# response to the client.
class Extension(SimpleExtension):
    "Python advanced sample Extension"

    def __init__(self):
        self.reload_watcher = ReloadWatcherThread()
        self.reload_watcher.start()

    def HttpExtensionProc(self, ecb):
        # NOTE: If you use a ThreadPoolExtension, you must still perform
        # this check in HttpExtensionProc - raising the exception from
        # The "Dispatch" method will just cause the exception to be
        # rendered to the browser.
        if self.reload_watcher.change_detected:
            print("Doing reload")
            raise InternalReloadException

        url = ecb.GetServerVariable("UNICODE_URL")
        if url.endswith("ReportUnhealthy"):
            ecb.ReportUnhealthy("I'm a little sick")

        ecb.SendResponseHeaders("200 OK", "Content-Type: text/html\r\n\r\n", 0)
        print("<HTML><BODY>", file=ecb)

        qs = ecb.GetServerVariable("QUERY_STRING")
        if qs:
            queries = qs.split("&")
            print("<PRE>", file=ecb)
            for q in queries:
                val = ecb.GetServerVariable(q, "&lt;no such variable&gt;")
                print("%s=%r" % (q, val), file=ecb)
            print("</PRE><P/>", file=ecb)

        print("This module has been imported", file=ecb)
        print("%d times" % (reload_counter,), file=ecb)
        print("</BODY></HTML>", file=ecb)
        ecb.close()
        return isapicon.HSE_STATUS_SUCCESS

    def TerminateExtension(self, status):
        self.reload_watcher.stop()


# The entry points for the ISAPI extension.
def __ExtensionFactory__():
    return Extension()


# Our special command line customization.
# Pre-install hook for our virtual directory.
def PreInstallDirectory(params, options):
    # If the user used our special '--description' option,
    # then we override our default.
    if options.description:
        params.Description = options.description


# Post install hook for our entire script
def PostInstall(params, options):
    print()
    print("The sample has been installed.")
    print("Point your browser to /AdvancedPythonSample")
    print("If you modify the source file and reload the page,")
    print("you should see the reload counter increment")


# Handler for our custom 'status' argument.
def status_handler(options, log, arg):
    "Query the status of something"
    print("Everything seems to be fine!")


custom_arg_handlers = {"status": status_handler}

if __name__ == "__main__":
    # If run from the command-line, install ourselves.
    from isapi.install import *

    params = ISAPIParameters(PostInstall=PostInstall)
    # Setup the virtual directories - this is a list of directories our
    # extension uses - in this case only 1.
    # Each extension has a "script map" - this is the mapping of ISAPI
    # extensions.
    sm = [ScriptMapParams(Extension="*", Flags=0)]
    vd = VirtualDirParameters(
        Name="AdvancedPythonSample",
        Description=Extension.__doc__,
        ScriptMaps=sm,
        ScriptMapUpdate="replace",
        # specify the pre-install hook.
        PreInstall=PreInstallDirectory,
    )
    params.VirtualDirs = [vd]
    # Setup our custom option parser.
    from optparse import OptionParser

    parser = OptionParser("")  # blank usage, so isapi sets it.
    parser.add_option(
        "",
        "--description",
        action="store",
        help="custom description to use for the virtual directory",
    )

    HandleCommandLine(
        params, opt_parser=parser, custom_arg_handlers=custom_arg_handlers
    )
