# Some registry helpers.
import os
import sys

import win32api
import win32con

error = "Registry utility error"

# A .py file has a CLSID associated with it (why? - dunno!)
CLSIDPyFile = "{b51df050-06ae-11cf-ad3b-524153480001}"

RegistryIDPyFile = "Python.File"  # The registry "file type" of a .py file
RegistryIDPycFile = "Python.CompiledFile"  # The registry "file type" of a .pyc file


def BuildDefaultPythonKey():
    """Builds a string containing the path to the current registry key.

    The Python registry key contains the Python version.  This function
    uses the version of the DLL used by the current process to get the
    registry key currently in use.
    """
    return "Software\\Python\\PythonCore\\" + sys.winver


def GetRootKey():
    """Retrieves the Registry root in use by Python."""
    keyname = BuildDefaultPythonKey()
    try:
        k = win32api.RegOpenKey(win32con.HKEY_CURRENT_USER, keyname)
        k.close()
        return win32con.HKEY_CURRENT_USER
    except win32api.error:
        return win32con.HKEY_LOCAL_MACHINE


def GetRegistryDefaultValue(subkey, rootkey=None):
    """A helper to return the default value for a key in the registry."""
    if rootkey is None:
        rootkey = GetRootKey()
    return win32api.RegQueryValue(rootkey, subkey)


def SetRegistryDefaultValue(subKey, value, rootkey=None):
    """A helper to set the default value for a key in the registry"""
    if rootkey is None:
        rootkey = GetRootKey()
    if type(value) == str:
        typeId = win32con.REG_SZ
    elif type(value) == int:
        typeId = win32con.REG_DWORD
    else:
        raise TypeError("Value must be string or integer - was passed " + repr(value))

    win32api.RegSetValue(rootkey, subKey, typeId, value)


def GetAppPathsKey():
    return "Software\\Microsoft\\Windows\\CurrentVersion\\App Paths"


def RegisterPythonExe(exeFullPath, exeAlias=None, exeAppPath=None):
    """Register a .exe file that uses Python.

    Registers the .exe with the OS.  This allows the specified .exe to
    be run from the command-line or start button without using the full path,
    and also to setup application specific path (ie, os.environ['PATH']).

    Currently the exeAppPath is not supported, so this function is general
    purpose, and not specific to Python at all.  Later, exeAppPath may provide
    a reasonable default that is used.

    exeFullPath -- The full path to the .exe
    exeAlias = None -- An alias for the exe - if none, the base portion
              of the filename is used.
    exeAppPath -- Not supported.
    """
    # Note - Dont work on win32s (but we dont care anymore!)
    if exeAppPath:
        raise error("Do not support exeAppPath argument currently")
    if exeAlias is None:
        exeAlias = os.path.basename(exeFullPath)
    win32api.RegSetValue(
        GetRootKey(), GetAppPathsKey() + "\\" + exeAlias, win32con.REG_SZ, exeFullPath
    )


def GetRegisteredExe(exeAlias):
    """Get a registered .exe"""
    return win32api.RegQueryValue(GetRootKey(), GetAppPathsKey() + "\\" + exeAlias)


def UnregisterPythonExe(exeAlias):
    """Unregister a .exe file that uses Python."""
    try:
        win32api.RegDeleteKey(GetRootKey(), GetAppPathsKey() + "\\" + exeAlias)
    except win32api.error as exc:
        import winerror

        if exc.winerror != winerror.ERROR_FILE_NOT_FOUND:
            raise
        return


def RegisterNamedPath(name, path):
    """Register a named path - ie, a named PythonPath entry."""
    keyStr = BuildDefaultPythonKey() + "\\PythonPath"
    if name:
        keyStr = keyStr + "\\" + name
    win32api.RegSetValue(GetRootKey(), keyStr, win32con.REG_SZ, path)


def UnregisterNamedPath(name):
    """Unregister a named path - ie, a named PythonPath entry."""
    keyStr = BuildDefaultPythonKey() + "\\PythonPath\\" + name
    try:
        win32api.RegDeleteKey(GetRootKey(), keyStr)
    except win32api.error as exc:
        import winerror

        if exc.winerror != winerror.ERROR_FILE_NOT_FOUND:
            raise
        return


def GetRegisteredNamedPath(name):
    """Get a registered named path, or None if it doesnt exist."""
    keyStr = BuildDefaultPythonKey() + "\\PythonPath"
    if name:
        keyStr = keyStr + "\\" + name
    try:
        return win32api.RegQueryValue(GetRootKey(), keyStr)
    except win32api.error as exc:
        import winerror

        if exc.winerror != winerror.ERROR_FILE_NOT_FOUND:
            raise
        return None


def RegisterModule(modName, modPath):
    """Register an explicit module in the registry.  This forces the Python import
    mechanism to locate this module directly, without a sys.path search.  Thus
    a registered module need not appear in sys.path at all.

    modName -- The name of the module, as used by import.
    modPath -- The full path and file name of the module.
    """
    try:
        import os

        os.stat(modPath)
    except os.error:
        print("Warning: Registering non-existant module %s" % modPath)
    win32api.RegSetValue(
        GetRootKey(),
        BuildDefaultPythonKey() + "\\Modules\\%s" % modName,
        win32con.REG_SZ,
        modPath,
    )


def UnregisterModule(modName):
    """Unregister an explicit module in the registry.

    modName -- The name of the module, as used by import.
    """
    try:
        win32api.RegDeleteKey(
            GetRootKey(), BuildDefaultPythonKey() + "\\Modules\\%s" % modName
        )
    except win32api.error as exc:
        import winerror

        if exc.winerror != winerror.ERROR_FILE_NOT_FOUND:
            raise


def GetRegisteredHelpFile(helpDesc):
    """Given a description, return the registered entry."""
    try:
        return GetRegistryDefaultValue(BuildDefaultPythonKey() + "\\Help\\" + helpDesc)
    except win32api.error:
        try:
            return GetRegistryDefaultValue(
                BuildDefaultPythonKey() + "\\Help\\" + helpDesc,
                win32con.HKEY_CURRENT_USER,
            )
        except win32api.error:
            pass
    return None


def RegisterHelpFile(helpFile, helpPath, helpDesc=None, bCheckFile=1):
    """Register a help file in the registry.

      Note that this used to support writing to the Windows Help
      key, however this is no longer done, as it seems to be incompatible.

    helpFile -- the base name of the help file.
    helpPath -- the path to the help file
    helpDesc -- A description for the help file.  If None, the helpFile param is used.
    bCheckFile -- A flag indicating if the file existence should be checked.
    """
    if helpDesc is None:
        helpDesc = helpFile
    fullHelpFile = os.path.join(helpPath, helpFile)
    try:
        if bCheckFile:
            os.stat(fullHelpFile)
    except os.error:
        raise ValueError("Help file does not exist")
    # Now register with Python itself.
    win32api.RegSetValue(
        GetRootKey(),
        BuildDefaultPythonKey() + "\\Help\\%s" % helpDesc,
        win32con.REG_SZ,
        fullHelpFile,
    )


def UnregisterHelpFile(helpFile, helpDesc=None):
    """Unregister a help file in the registry.

    helpFile -- the base name of the help file.
    helpDesc -- A description for the help file.  If None, the helpFile param is used.
    """
    key = win32api.RegOpenKey(
        win32con.HKEY_LOCAL_MACHINE,
        "Software\\Microsoft\\Windows\\Help",
        0,
        win32con.KEY_ALL_ACCESS,
    )
    try:
        try:
            win32api.RegDeleteValue(key, helpFile)
        except win32api.error as exc:
            import winerror

            if exc.winerror != winerror.ERROR_FILE_NOT_FOUND:
                raise
    finally:
        win32api.RegCloseKey(key)

    # Now de-register with Python itself.
    if helpDesc is None:
        helpDesc = helpFile
    try:
        win32api.RegDeleteKey(
            GetRootKey(), BuildDefaultPythonKey() + "\\Help\\%s" % helpDesc
        )
    except win32api.error as exc:
        import winerror

        if exc.winerror != winerror.ERROR_FILE_NOT_FOUND:
            raise


def RegisterCoreDLL(coredllName=None):
    """Registers the core DLL in the registry.

    If no params are passed, the name of the Python DLL used in
    the current process is used and registered.
    """
    if coredllName is None:
        coredllName = win32api.GetModuleFileName(sys.dllhandle)
        # must exist!
    else:
        try:
            os.stat(coredllName)
        except os.error:
            print("Warning: Registering non-existant core DLL %s" % coredllName)

    hKey = win32api.RegCreateKey(GetRootKey(), BuildDefaultPythonKey())
    try:
        win32api.RegSetValue(hKey, "Dll", win32con.REG_SZ, coredllName)
    finally:
        win32api.RegCloseKey(hKey)
    # Lastly, setup the current version to point to me.
    win32api.RegSetValue(
        GetRootKey(),
        "Software\\Python\\PythonCore\\CurrentVersion",
        win32con.REG_SZ,
        sys.winver,
    )


def RegisterFileExtensions(defPyIcon, defPycIcon, runCommand):
    """Register the core Python file extensions.

    defPyIcon -- The default icon to use for .py files, in 'fname,offset' format.
    defPycIcon -- The default icon to use for .pyc files, in 'fname,offset' format.
    runCommand -- The command line to use for running .py files
    """
    # Register the file extensions.
    pythonFileId = RegistryIDPyFile
    win32api.RegSetValue(
        win32con.HKEY_CLASSES_ROOT, ".py", win32con.REG_SZ, pythonFileId
    )
    win32api.RegSetValue(
        win32con.HKEY_CLASSES_ROOT, pythonFileId, win32con.REG_SZ, "Python File"
    )
    win32api.RegSetValue(
        win32con.HKEY_CLASSES_ROOT,
        "%s\\CLSID" % pythonFileId,
        win32con.REG_SZ,
        CLSIDPyFile,
    )
    win32api.RegSetValue(
        win32con.HKEY_CLASSES_ROOT,
        "%s\\DefaultIcon" % pythonFileId,
        win32con.REG_SZ,
        defPyIcon,
    )
    base = "%s\\Shell" % RegistryIDPyFile
    win32api.RegSetValue(
        win32con.HKEY_CLASSES_ROOT, base + "\\Open", win32con.REG_SZ, "Run"
    )
    win32api.RegSetValue(
        win32con.HKEY_CLASSES_ROOT,
        base + "\\Open\\Command",
        win32con.REG_SZ,
        runCommand,
    )

    # Register the .PYC.
    pythonFileId = RegistryIDPycFile
    win32api.RegSetValue(
        win32con.HKEY_CLASSES_ROOT, ".pyc", win32con.REG_SZ, pythonFileId
    )
    win32api.RegSetValue(
        win32con.HKEY_CLASSES_ROOT,
        pythonFileId,
        win32con.REG_SZ,
        "Compiled Python File",
    )
    win32api.RegSetValue(
        win32con.HKEY_CLASSES_ROOT,
        "%s\\DefaultIcon" % pythonFileId,
        win32con.REG_SZ,
        defPycIcon,
    )
    base = "%s\\Shell" % pythonFileId
    win32api.RegSetValue(
        win32con.HKEY_CLASSES_ROOT, base + "\\Open", win32con.REG_SZ, "Run"
    )
    win32api.RegSetValue(
        win32con.HKEY_CLASSES_ROOT,
        base + "\\Open\\Command",
        win32con.REG_SZ,
        runCommand,
    )


def RegisterShellCommand(shellCommand, exeCommand, shellUserCommand=None):
    # Last param for "Open" - for a .py file to be executed by the command line
    # or shell execute (eg, just entering "foo.py"), the Command must be "Open",
    # but you may associate a different name for the right-click menu.
    # In our case, normally we have "Open=Run"
    base = "%s\\Shell" % RegistryIDPyFile
    if shellUserCommand:
        win32api.RegSetValue(
            win32con.HKEY_CLASSES_ROOT,
            base + "\\%s" % (shellCommand),
            win32con.REG_SZ,
            shellUserCommand,
        )

    win32api.RegSetValue(
        win32con.HKEY_CLASSES_ROOT,
        base + "\\%s\\Command" % (shellCommand),
        win32con.REG_SZ,
        exeCommand,
    )


def RegisterDDECommand(shellCommand, ddeApp, ddeTopic, ddeCommand):
    base = "%s\\Shell" % RegistryIDPyFile
    win32api.RegSetValue(
        win32con.HKEY_CLASSES_ROOT,
        base + "\\%s\\ddeexec" % (shellCommand),
        win32con.REG_SZ,
        ddeCommand,
    )
    win32api.RegSetValue(
        win32con.HKEY_CLASSES_ROOT,
        base + "\\%s\\ddeexec\\Application" % (shellCommand),
        win32con.REG_SZ,
        ddeApp,
    )
    win32api.RegSetValue(
        win32con.HKEY_CLASSES_ROOT,
        base + "\\%s\\ddeexec\\Topic" % (shellCommand),
        win32con.REG_SZ,
        ddeTopic,
    )
