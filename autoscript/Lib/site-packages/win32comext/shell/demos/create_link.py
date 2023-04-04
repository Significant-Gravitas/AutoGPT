# link.py
# From a demo by Mark Hammond, corrupted by Mike Fletcher
# (and re-corrupted by Mark Hammond :-)
import os

import pythoncom
from win32com.shell import shell


class PyShortcut:
    def __init__(self):
        self._base = pythoncom.CoCreateInstance(
            shell.CLSID_ShellLink,
            None,
            pythoncom.CLSCTX_INPROC_SERVER,
            shell.IID_IShellLink,
        )

    def load(self, filename):
        # Get an IPersist interface
        # which allows save/restore of object to/from files
        self._base.QueryInterface(pythoncom.IID_IPersistFile).Load(filename)

    def save(self, filename):
        self._base.QueryInterface(pythoncom.IID_IPersistFile).Save(filename, 0)

    def __getattr__(self, name):
        if name != "_base":
            return getattr(self._base, name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: %s LinkFile [path [, args[, description[, working_dir]]]]\n\nIf LinkFile does not exist, it will be created using the other args"
        )
        sys.exit(1)
    file = sys.argv[1]
    shortcut = PyShortcut()
    if os.path.exists(file):
        # load and dump info from file...
        shortcut.load(file)
        # now print data...
        print(
            "Shortcut in file %s to file:\n\t%s\nArguments:\n\t%s\nDescription:\n\t%s\nWorking Directory:\n\t%s\nItemIDs:\n\t<skipped>"
            % (
                file,
                shortcut.GetPath(shell.SLGP_SHORTPATH)[0],
                shortcut.GetArguments(),
                shortcut.GetDescription(),
                shortcut.GetWorkingDirectory(),
                # shortcut.GetIDList(),
            )
        )
    else:
        if len(sys.argv) < 3:
            print(
                "Link file does not exist\nYou must supply the path, args, description and working_dir as args"
            )
            sys.exit(1)
        # create the shortcut using rest of args...
        data = map(
            None,
            sys.argv[2:],
            ("SetPath", "SetArguments", "SetDescription", "SetWorkingDirectory"),
        )
        for value, function in data:
            if value and function:
                # call function on each non-null value
                getattr(shortcut, function)(value)
        shortcut.save(file)
