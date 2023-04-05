# A tool to setup the Python registry.


class error(Exception):
    pass


import sys  # at least we can count on this!


def FileExists(fname):
    """Check if a file exists.  Returns true or false."""
    import os

    try:
        os.stat(fname)
        return 1
    except os.error as details:
        return 0


def IsPackageDir(path, packageName, knownFileName):
    """Given a path, a ni package name, and possibly a known file name in
    the root of the package, see if this path is good.
    """
    import os

    if knownFileName is None:
        knownFileName = "."
    return FileExists(os.path.join(os.path.join(path, packageName), knownFileName))


def IsDebug():
    """Return "_d" if we're running a debug version.

    This is to be used within DLL names when locating them.
    """
    import importlib.machinery

    return "_d" if "_d.pyd" in importlib.machinery.EXTENSION_SUFFIXES else ""


def FindPackagePath(packageName, knownFileName, searchPaths):
    """Find a package.

    Given a ni style package name, check the package is registered.

    First place looked is the registry for an existing entry.  Then
    the searchPaths are searched.
    """
    import os

    import regutil

    pathLook = regutil.GetRegisteredNamedPath(packageName)
    if pathLook and IsPackageDir(pathLook, packageName, knownFileName):
        return pathLook, None  # The currently registered one is good.
    # Search down the search paths.
    for pathLook in searchPaths:
        if IsPackageDir(pathLook, packageName, knownFileName):
            # Found it
            ret = os.path.abspath(pathLook)
            return ret, ret
    raise error("The package %s can not be located" % packageName)


def FindHelpPath(helpFile, helpDesc, searchPaths):
    # See if the current registry entry is OK
    import os

    import win32api
    import win32con

    try:
        key = win32api.RegOpenKey(
            win32con.HKEY_LOCAL_MACHINE,
            "Software\\Microsoft\\Windows\\Help",
            0,
            win32con.KEY_ALL_ACCESS,
        )
        try:
            try:
                path = win32api.RegQueryValueEx(key, helpDesc)[0]
                if FileExists(os.path.join(path, helpFile)):
                    return os.path.abspath(path)
            except win32api.error:
                pass  # no registry entry.
        finally:
            key.Close()
    except win32api.error:
        pass
    for pathLook in searchPaths:
        if FileExists(os.path.join(pathLook, helpFile)):
            return os.path.abspath(pathLook)
        pathLook = os.path.join(pathLook, "Help")
        if FileExists(os.path.join(pathLook, helpFile)):
            return os.path.abspath(pathLook)
    raise error("The help file %s can not be located" % helpFile)


def FindAppPath(appName, knownFileName, searchPaths):
    """Find an application.

    First place looked is the registry for an existing entry.  Then
    the searchPaths are searched.
    """
    # Look in the first path.
    import os

    import regutil

    regPath = regutil.GetRegisteredNamedPath(appName)
    if regPath:
        pathLook = regPath.split(";")[0]
    if regPath and FileExists(os.path.join(pathLook, knownFileName)):
        return None  # The currently registered one is good.
    # Search down the search paths.
    for pathLook in searchPaths:
        if FileExists(os.path.join(pathLook, knownFileName)):
            # Found it
            return os.path.abspath(pathLook)
    raise error(
        "The file %s can not be located for application %s" % (knownFileName, appName)
    )


def FindPythonExe(exeAlias, possibleRealNames, searchPaths):
    """Find an exe.

    Returns the full path to the .exe, and a boolean indicating if the current
    registered entry is OK.  We don't trust the already registered version even
    if it exists - it may be wrong (ie, for a different Python version)
    """
    import os
    import sys

    import regutil
    import win32api

    if possibleRealNames is None:
        possibleRealNames = exeAlias
    # Look first in Python's home.
    found = os.path.join(sys.prefix, possibleRealNames)
    if not FileExists(found):  # for developers
        if "64 bit" in sys.version:
            found = os.path.join(sys.prefix, "PCBuild", "amd64", possibleRealNames)
        else:
            found = os.path.join(sys.prefix, "PCBuild", possibleRealNames)
    if not FileExists(found):
        found = LocateFileName(possibleRealNames, searchPaths)

    registered_ok = 0
    try:
        registered = win32api.RegQueryValue(
            regutil.GetRootKey(), regutil.GetAppPathsKey() + "\\" + exeAlias
        )
        registered_ok = found == registered
    except win32api.error:
        pass
    return found, registered_ok


def QuotedFileName(fname):
    """Given a filename, return a quoted version if necessary"""

    import regutil

    try:
        fname.index(" ")  # Other chars forcing quote?
        return '"%s"' % fname
    except ValueError:
        # No space in name.
        return fname


def LocateFileName(fileNamesString, searchPaths):
    """Locate a file name, anywhere on the search path.

    If the file can not be located, prompt the user to find it for us
    (using a common OpenFile dialog)

    Raises KeyboardInterrupt if the user cancels.
    """
    import os

    import regutil

    fileNames = fileNamesString.split(";")
    for path in searchPaths:
        for fileName in fileNames:
            try:
                retPath = os.path.join(path, fileName)
                os.stat(retPath)
                break
            except os.error:
                retPath = None
        if retPath:
            break
    else:
        fileName = fileNames[0]
        try:
            import win32con
            import win32ui
        except ImportError:
            raise error(
                "Need to locate the file %s, but the win32ui module is not available\nPlease run the program again, passing as a parameter the path to this file."
                % fileName
            )
        # Display a common dialog to locate the file.
        flags = win32con.OFN_FILEMUSTEXIST
        ext = os.path.splitext(fileName)[1]
        filter = "Files of requested type (*%s)|*%s||" % (ext, ext)
        dlg = win32ui.CreateFileDialog(1, None, fileName, flags, filter, None)
        dlg.SetOFNTitle("Locate " + fileName)
        if dlg.DoModal() != win32con.IDOK:
            raise KeyboardInterrupt("User cancelled the process")
        retPath = dlg.GetPathName()
    return os.path.abspath(retPath)


def LocatePath(fileName, searchPaths):
    """Like LocateFileName, but returns a directory only."""
    import os

    return os.path.abspath(os.path.split(LocateFileName(fileName, searchPaths))[0])


def LocateOptionalPath(fileName, searchPaths):
    """Like LocatePath, but returns None if the user cancels."""
    try:
        return LocatePath(fileName, searchPaths)
    except KeyboardInterrupt:
        return None


def LocateOptionalFileName(fileName, searchPaths=None):
    """Like LocateFileName, but returns None if the user cancels."""
    try:
        return LocateFileName(fileName, searchPaths)
    except KeyboardInterrupt:
        return None


def LocatePythonCore(searchPaths):
    """Locate and validate the core Python directories.  Returns a list
    of paths that should be used as the core (ie, un-named) portion of
    the Python path.
    """
    import os

    import regutil

    currentPath = regutil.GetRegisteredNamedPath(None)
    if currentPath:
        presearchPaths = currentPath.split(";")
    else:
        presearchPaths = [os.path.abspath(".")]
    libPath = None
    for path in presearchPaths:
        if FileExists(os.path.join(path, "os.py")):
            libPath = path
            break
    if libPath is None and searchPaths is not None:
        libPath = LocatePath("os.py", searchPaths)
    if libPath is None:
        raise error("The core Python library could not be located.")

    corePath = None
    suffix = IsDebug()
    for path in presearchPaths:
        if FileExists(os.path.join(path, "unicodedata%s.pyd" % suffix)):
            corePath = path
            break
    if corePath is None and searchPaths is not None:
        corePath = LocatePath("unicodedata%s.pyd" % suffix, searchPaths)
    if corePath is None:
        raise error("The core Python path could not be located.")

    installPath = os.path.abspath(os.path.join(libPath, ".."))
    return installPath, [libPath, corePath]


def FindRegisterPackage(packageName, knownFile, searchPaths, registryAppName=None):
    """Find and Register a package.

    Assumes the core registry setup correctly.

    In addition, if the location located by the package is already
    in the **core** path, then an entry is registered, but no path.
    (no other paths are checked, as the application whose path was used
    may later be uninstalled.  This should not happen with the core)
    """

    import regutil

    if not packageName:
        raise error("A package name must be supplied")
    corePaths = regutil.GetRegisteredNamedPath(None).split(";")
    if not searchPaths:
        searchPaths = corePaths
    registryAppName = registryAppName or packageName
    try:
        pathLook, pathAdd = FindPackagePath(packageName, knownFile, searchPaths)
        if pathAdd is not None:
            if pathAdd in corePaths:
                pathAdd = ""
            regutil.RegisterNamedPath(registryAppName, pathAdd)
        return pathLook
    except error as details:
        print(
            "*** The %s package could not be registered - %s" % (packageName, details)
        )
        print(
            "*** Please ensure you have passed the correct paths on the command line."
        )
        print(
            "*** - For packages, you should pass a path to the packages parent directory,"
        )
        print("*** - and not the package directory itself...")


def FindRegisterApp(appName, knownFiles, searchPaths):
    """Find and Register a package.

    Assumes the core registry setup correctly.

    """

    import regutil

    if type(knownFiles) == type(""):
        knownFiles = [knownFiles]
    paths = []
    try:
        for knownFile in knownFiles:
            pathLook = FindAppPath(appName, knownFile, searchPaths)
            if pathLook:
                paths.append(pathLook)
    except error as details:
        print("*** ", details)
        return

    regutil.RegisterNamedPath(appName, ";".join(paths))


def FindRegisterPythonExe(exeAlias, searchPaths, actualFileNames=None):
    """Find and Register a Python exe (not necessarily *the* python.exe)

    Assumes the core registry setup correctly.
    """

    import regutil

    fname, ok = FindPythonExe(exeAlias, actualFileNames, searchPaths)
    if not ok:
        regutil.RegisterPythonExe(fname, exeAlias)
    return fname


def FindRegisterHelpFile(helpFile, searchPaths, helpDesc=None):
    import regutil

    try:
        pathLook = FindHelpPath(helpFile, helpDesc, searchPaths)
    except error as details:
        print("*** ", details)
        return
    #       print "%s found at %s" % (helpFile, pathLook)
    regutil.RegisterHelpFile(helpFile, pathLook, helpDesc)


def SetupCore(searchPaths):
    """Setup the core Python information in the registry.

    This function makes no assumptions about the current state of sys.path.

    After this function has completed, you should have access to the standard
    Python library, and the standard Win32 extensions
    """

    import sys

    for path in searchPaths:
        sys.path.append(path)

    import os

    import regutil
    import win32api
    import win32con

    installPath, corePaths = LocatePythonCore(searchPaths)
    # Register the core Pythonpath.
    print(corePaths)
    regutil.RegisterNamedPath(None, ";".join(corePaths))

    # Register the install path.
    hKey = win32api.RegCreateKey(regutil.GetRootKey(), regutil.BuildDefaultPythonKey())
    try:
        # Core Paths.
        win32api.RegSetValue(hKey, "InstallPath", win32con.REG_SZ, installPath)
    finally:
        win32api.RegCloseKey(hKey)

    # Register the win32 core paths.
    win32paths = (
        os.path.abspath(os.path.split(win32api.__file__)[0])
        + ";"
        + os.path.abspath(
            os.path.split(LocateFileName("win32con.py;win32con.pyc", sys.path))[0]
        )
    )

    # Python has builtin support for finding a "DLLs" directory, but
    # not a PCBuild.  Having it in the core paths means it is ignored when
    # an EXE not in the Python dir is hosting us - so we add it as a named
    # value
    check = os.path.join(sys.prefix, "PCBuild")
    if "64 bit" in sys.version:
        check = os.path.join(check, "amd64")
    if os.path.isdir(check):
        regutil.RegisterNamedPath("PCBuild", check)


def RegisterShellInfo(searchPaths):
    """Registers key parts of the Python installation with the Windows Shell.

    Assumes a valid, minimal Python installation exists
    (ie, SetupCore() has been previously successfully run)
    """
    import regutil
    import win32con

    suffix = IsDebug()
    # Set up a pointer to the .exe's
    exePath = FindRegisterPythonExe("Python%s.exe" % suffix, searchPaths)
    regutil.SetRegistryDefaultValue(".py", "Python.File", win32con.HKEY_CLASSES_ROOT)
    regutil.RegisterShellCommand("Open", QuotedFileName(exePath) + ' "%1" %*', "&Run")
    regutil.SetRegistryDefaultValue(
        "Python.File\\DefaultIcon", "%s,0" % exePath, win32con.HKEY_CLASSES_ROOT
    )

    FindRegisterHelpFile("Python.hlp", searchPaths, "Main Python Documentation")
    FindRegisterHelpFile("ActivePython.chm", searchPaths, "Main Python Documentation")

    # We consider the win32 core, as it contains all the win32 api type
    # stuff we need.


#       FindRegisterApp("win32", ["win32con.pyc", "win32api%s.pyd" % suffix], searchPaths)

usage = (
    """\
regsetup.py - Setup/maintain the registry for Python apps.

Run without options, (but possibly search paths) to repair a totally broken
python registry setup.  This should allow other options to work.

Usage:   %s [options ...] paths ...
-p packageName  -- Find and register a package.  Looks in the paths for
                   a sub-directory with the name of the package, and
                   adds a path entry for the package.
-a appName      -- Unconditionally add an application name to the path.
                   A new path entry is create with the app name, and the
                   paths specified are added to the registry.
-c              -- Add the specified paths to the core Pythonpath.
                   If a path appears on the core path, and a package also
                   needs that same path, the package will not bother
                   registering it.  Therefore, By adding paths to the
                   core path, you can avoid packages re-registering the same path.
-m filename     -- Find and register the specific file name as a module.
                   Do not include a path on the filename!
--shell         -- Register everything with the Win95/NT shell.
--upackage name -- Unregister the package
--uapp name     -- Unregister the app (identical to --upackage)
--umodule name  -- Unregister the module

--description   -- Print a description of the usage.
--examples      -- Print examples of usage.
"""
    % sys.argv[0]
)

description = """\
If no options are processed, the program attempts to validate and set
the standard Python path to the point where the standard library is
available.  This can be handy if you move Python to a new drive/sub-directory,
in which case most of the options would fail (as they need at least string.py,
os.py etc to function.)
Running without options should repair Python well enough to run with
the other options.

paths are search paths that the program will use to seek out a file.
For example, when registering the core Python, you may wish to
provide paths to non-standard places to look for the Python help files,
library files, etc.

See also the "regcheck.py" utility which will check and dump the contents
of the registry.
"""

examples = """\
Examples:
"regsetup c:\\wierd\\spot\\1 c:\\wierd\\spot\\2"
Attempts to setup the core Python.  Looks in some standard places,
as well as the 2 wierd spots to locate the core Python files (eg, Python.exe,
python14.dll, the standard library and Win32 Extensions.

"regsetup -a myappname . .\subdir"
Registers a new Pythonpath entry named myappname, with "C:\\I\\AM\\HERE" and
"C:\\I\\AM\\HERE\subdir" added to the path (ie, all args are converted to
absolute paths)

"regsetup -c c:\\my\\python\\files"
Unconditionally add "c:\\my\\python\\files" to the 'core' Python path.

"regsetup -m some.pyd \\windows\\system"
Register the module some.pyd in \\windows\\system as a registered
module.  This will allow some.pyd to be imported, even though the
windows system directory is not (usually!) on the Python Path.

"regsetup --umodule some"
Unregister the module "some".  This means normal import rules then apply
for that module.
"""

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["/?", "-?", "-help", "-h"]:
        print(usage)
    elif len(sys.argv) == 1 or not sys.argv[1][0] in ["/", "-"]:
        # No args, or useful args.
        searchPath = sys.path[:]
        for arg in sys.argv[1:]:
            searchPath.append(arg)
        # Good chance we are being run from the "regsetup.py" directory.
        # Typically this will be "\somewhere\win32\Scripts" and the
        # "somewhere" and "..\Lib" should also be searched.
        searchPath.append("..\\Build")
        searchPath.append("..\\Lib")
        searchPath.append("..")
        searchPath.append("..\\..")

        # for developers:
        # also search somewhere\lib, ..\build, and ..\..\build
        searchPath.append("..\\..\\lib")
        searchPath.append("..\\build")
        if "64 bit" in sys.version:
            searchPath.append("..\\..\\pcbuild\\amd64")
        else:
            searchPath.append("..\\..\\pcbuild")

        print("Attempting to setup/repair the Python core")

        SetupCore(searchPath)
        RegisterShellInfo(searchPath)
        FindRegisterHelpFile("PyWin32.chm", searchPath, "Pythonwin Reference")
        # Check the registry.
        print("Registration complete - checking the registry...")
        import regcheck

        regcheck.CheckRegistry()
    else:
        searchPaths = []
        import getopt

        opts, args = getopt.getopt(
            sys.argv[1:],
            "p:a:m:c",
            ["shell", "upackage=", "uapp=", "umodule=", "description", "examples"],
        )
        for arg in args:
            searchPaths.append(arg)
        for o, a in opts:
            if o == "--description":
                print(description)
            if o == "--examples":
                print(examples)
            if o == "--shell":
                print("Registering the Python core.")
                RegisterShellInfo(searchPaths)
            if o == "-p":
                print("Registering package", a)
                FindRegisterPackage(a, None, searchPaths)
            if o in ["--upackage", "--uapp"]:
                import regutil

                print("Unregistering application/package", a)
                regutil.UnregisterNamedPath(a)
            if o == "-a":
                import regutil

                path = ";".join(searchPaths)
                print("Registering application", a, "to path", path)
                regutil.RegisterNamedPath(a, path)
            if o == "-c":
                if not len(searchPaths):
                    raise error("-c option must provide at least one additional path")
                import regutil
                import win32api

                currentPaths = regutil.GetRegisteredNamedPath(None).split(";")
                oldLen = len(currentPaths)
                for newPath in searchPaths:
                    if newPath not in currentPaths:
                        currentPaths.append(newPath)
                if len(currentPaths) != oldLen:
                    print(
                        "Registering %d new core paths" % (len(currentPaths) - oldLen)
                    )
                    regutil.RegisterNamedPath(None, ";".join(currentPaths))
                else:
                    print("All specified paths are already registered.")
