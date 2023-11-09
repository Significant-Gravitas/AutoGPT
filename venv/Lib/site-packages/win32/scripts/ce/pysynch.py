# Simple CE synchronisation utility with Python features.

import fnmatch
import getopt
import os
import string
import sys

import win32api
import win32con
import win32file
import wincerapi


class InvalidUsage(Exception):
    pass


def print_error(api_exc, msg):
    hr, fn, errmsg = api_exc
    print("%s - %s(%d)" % (msg, errmsg, hr))


def GetFileAttributes(file, local=1):
    if local:
        return win32api.GetFileAttributes(file)
    else:
        return wincerapi.CeGetFileAttributes(file)


def FindFiles(spec, local=1):
    if local:
        return win32api.FindFiles(spec)
    else:
        return wincerapi.CeFindFiles(spec)


def isdir(name, local=1):
    try:
        attr = GetFileAttributes(name, local)
        return attr & win32con.FILE_ATTRIBUTE_DIRECTORY
    except win32api.error:
        return 0


def CopyFileToCe(src_name, dest_name, progress=None):
    sh = win32file.CreateFile(
        src_name, win32con.GENERIC_READ, 0, None, win32con.OPEN_EXISTING, 0, None
    )
    bytes = 0
    try:
        dh = wincerapi.CeCreateFile(
            dest_name, win32con.GENERIC_WRITE, 0, None, win32con.OPEN_ALWAYS, 0, None
        )
        try:
            while 1:
                hr, data = win32file.ReadFile(sh, 2048)
                if not data:
                    break
                wincerapi.CeWriteFile(dh, data)
                bytes = bytes + len(data)
                if progress is not None:
                    progress(bytes)
        finally:
            pass
            dh.Close()
    finally:
        sh.Close()
    return bytes


def BuildFileList(spec, local, recurse, filter, filter_args, recursed_path=""):
    files = []
    if isdir(spec, local):
        path = spec
        raw_spec = "*"
    else:
        path, raw_spec = os.path.split(spec)
    if recurse:
        # Need full scan, to get sub-direcetories.
        infos = FindFiles(os.path.join(path, "*"), local)
    else:
        infos = FindFiles(os.path.join(path, raw_spec), local)
    for info in infos:
        src_name = str(info[8])
        full_src_name = os.path.join(path, src_name)
        if local:  # Can't do this for CE!
            full_src_name = win32api.GetFullPathName(full_src_name)
        if isdir(full_src_name, local):
            if recurse and src_name not in [".", ".."]:
                new_spec = os.path.join(full_src_name, raw_spec)
                files = files + BuildFileList(
                    new_spec,
                    local,
                    1,
                    filter,
                    filter_args,
                    os.path.join(recursed_path, src_name),
                )
        if fnmatch.fnmatch(src_name, raw_spec):
            rel_name = os.path.join(recursed_path, src_name)
            filter_data = filter(full_src_name, rel_name, info, local, filter_args)
            if filter_data is not None:
                files.append((full_src_name, info, filter_data))
    return files


def _copyfilter(full_name, rel_name, info, local, bMaintainDir):
    if isdir(full_name, local):
        return
    if bMaintainDir:
        return rel_name
    return os.path.split(rel_name)[1]


import pywin.dialogs.status
import win32ui


class FileCopyProgressDialog(pywin.dialogs.status.CStatusProgressDialog):
    def CopyProgress(self, bytes):
        self.Set(bytes / 1024)


def copy(args):
    """copy src [src ...],  dest
    Copy files to/from the CE device
    """
    bRecurse = bVerbose = 0
    bMaintainDir = 1
    try:
        opts, args = getopt.getopt(args, "rv")
    except getopt.error as details:
        raise InvalidUsage(details)
    for o, v in opts:
        if o == "-r":
            bRecuse = 1
        elif o == "-v":
            bVerbose = 1

    if len(args) < 2:
        raise InvalidUsage("Must specify a source and destination")

    src = args[:-1]
    dest = args[-1]
    # See if WCE: leading anywhere indicates a direction.
    if string.find(src[0], "WCE:") == 0:
        bToDevice = 0
    elif string.find(dest, "WCE:") == 0:
        bToDevice = 1
    else:
        # Assume copy to device.
        bToDevice = 1

    if not isdir(dest, not bToDevice):
        print("%s does not indicate a directory")

    files = []  # List of FQ (from_name, to_name)
    num_files = 0
    num_bytes = 0
    dialog = FileCopyProgressDialog("Copying files")
    dialog.CreateWindow(win32ui.GetMainFrame())
    if bToDevice:
        for spec in src:
            new = BuildFileList(spec, 1, bRecurse, _copyfilter, bMaintainDir)
            if not new:
                print("Warning: '%s' did not match any files" % (spec))
            files = files + new

        for full_src, src_info, dest_info in files:
            dest_name = os.path.join(dest, dest_info)
            size = src_info[5]
            print("Size=", size)
            if bVerbose:
                print(full_src, "->", dest_name, "- ", end=" ")
            dialog.SetText(dest_name)
            dialog.Set(0, size / 1024)
            bytes = CopyFileToCe(full_src, dest_name, dialog.CopyProgress)
            num_bytes = num_bytes + bytes
            if bVerbose:
                print(bytes, "bytes")
            num_files = num_files + 1
    dialog.Close()
    print("%d files copied (%d bytes)" % (num_files, num_bytes))


def _dirfilter(*args):
    return args[1]


def dir(args):
    """dir directory_name ...
    Perform a directory listing on the remote device
    """
    bRecurse = 0
    try:
        opts, args = getopt.getopt(args, "r")
    except getopt.error as details:
        raise InvalidUsage(details)
    for o, v in opts:
        if o == "-r":
            bRecurse = 1
    for arg in args:
        print("Directory of WCE:%s" % arg)
        files = BuildFileList(arg, 0, bRecurse, _dirfilter, None)
        total_size = 0
        for full_name, info, rel_name in files:
            date_str = info[3].Format("%d-%b-%Y %H:%M")
            attr_string = "     "
            if info[0] & win32con.FILE_ATTRIBUTE_DIRECTORY:
                attr_string = "<DIR>"
            print("%s  %s %10d %s" % (date_str, attr_string, info[5], rel_name))
            total_size = total_size + info[5]
        print(" " * 14 + "%3d files, %10d bytes" % (len(files), total_size))


def run(args):
    """run program [args]
    Starts the specified program on the remote device.
    """
    prog_args = []
    for arg in args:
        if " " in arg:
            prog_args.append('"' + arg + '"')
        else:
            prog_args.append(arg)
    prog_args = string.join(prog_args, " ")
    wincerapi.CeCreateProcess(prog_args, "", None, None, 0, 0, None, "", None)


def delete(args):
    """delete file, ...
    Delete one or more remote files
    """
    for arg in args:
        try:
            wincerapi.CeDeleteFile(arg)
            print("Deleted: %s" % arg)
        except win32api.error as details:
            print_error(details, "Error deleting '%s'" % arg)


def DumpCommands():
    print("%-10s - %s" % ("Command", "Description"))
    print("%-10s - %s" % ("-------", "-----------"))
    for name, item in list(globals().items()):
        if type(item) == type(DumpCommands):
            doc = getattr(item, "__doc__", "")
            if doc:
                lines = string.split(doc, "\n")
                print("%-10s - %s" % (name, lines[0]))
                for line in lines[1:]:
                    if line:
                        print(" " * 8, line)


def main():
    if len(sys.argv) < 2:
        print("You must specify a command!")
        DumpCommands()
        return
    command = sys.argv[1]
    fn = globals().get(command)
    if fn is None:
        print("Unknown command:", command)
        DumpCommands()
        return

    wincerapi.CeRapiInit()
    try:
        verinfo = wincerapi.CeGetVersionEx()
        print(
            "Connected to device, CE version %d.%d %s"
            % (verinfo[0], verinfo[1], verinfo[4])
        )
        try:
            fn(sys.argv[2:])
        except InvalidUsage as msg:
            print("Invalid syntax -", msg)
            print(fn.__doc__)

    finally:
        try:
            wincerapi.CeRapiUninit()
        except win32api.error as details:
            print_error(details, "Error disconnecting")


if __name__ == "__main__":
    main()
