# This extension is used mainly for testing purposes - it is not
# designed to be a simple sample, but instead is a hotch-potch of things
# that attempts to exercise the framework.

import os
import stat
import sys

from isapi import isapicon
from isapi.simple import SimpleExtension

if hasattr(sys, "isapidllhandle"):
    import win32traceutil

# We use the same reload support as 'advanced.py' demonstrates.
import threading

import win32con
import win32event
import win32file
import winerror

from isapi import InternalReloadException


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


def TransmitFileCallback(ecb, hFile, cbIO, errCode):
    print("Transmit complete!")
    ecb.close()


# The ISAPI extension - handles requests in our virtual dir, and sends the
# response to the client.
class Extension(SimpleExtension):
    "Python test Extension"

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

        if ecb.GetServerVariable("UNICODE_URL").endswith("test.py"):
            file_flags = (
                win32con.FILE_FLAG_SEQUENTIAL_SCAN | win32con.FILE_FLAG_OVERLAPPED
            )
            hfile = win32file.CreateFile(
                __file__,
                win32con.GENERIC_READ,
                0,
                None,
                win32con.OPEN_EXISTING,
                file_flags,
                None,
            )
            flags = (
                isapicon.HSE_IO_ASYNC
                | isapicon.HSE_IO_DISCONNECT_AFTER_SEND
                | isapicon.HSE_IO_SEND_HEADERS
            )
            # We pass hFile to the callback simply as a way of keeping it alive
            # for the duration of the transmission
            try:
                ecb.TransmitFile(
                    TransmitFileCallback,
                    hfile,
                    int(hfile),
                    "200 OK",
                    0,
                    0,
                    None,
                    None,
                    flags,
                )
            except:
                # Errors keep this source file open!
                hfile.Close()
                raise
        else:
            # default response
            ecb.SendResponseHeaders("200 OK", "Content-Type: text/html\r\n\r\n", 0)
            print("<HTML><BODY>", file=ecb)
            print("The root of this site is at", ecb.MapURLToPath("/"), file=ecb)
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
    print("Point your browser to /PyISAPITest")


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
        Name="PyISAPITest",
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
