"""Utilities for registering objects.

This module contains utility functions to register Python objects as
valid COM Servers.  The RegisterServer function provides all information
necessary to allow the COM framework to respond to a request for a COM object,
construct the necessary Python object, and dispatch COM events.

"""
import os
import sys

import pythoncom
import win32api
import win32con
import winerror

CATID_PythonCOMServer = "{B3EF80D0-68E2-11D0-A689-00C04FD658FF}"


def _set_subkeys(keyName, valueDict, base=win32con.HKEY_CLASSES_ROOT):
    hkey = win32api.RegCreateKey(base, keyName)
    try:
        for key, value in valueDict.items():
            win32api.RegSetValueEx(hkey, key, None, win32con.REG_SZ, value)
    finally:
        win32api.RegCloseKey(hkey)


def _set_string(path, value, base=win32con.HKEY_CLASSES_ROOT):
    "Set a string value in the registry."

    win32api.RegSetValue(base, path, win32con.REG_SZ, value)


def _get_string(path, base=win32con.HKEY_CLASSES_ROOT):
    "Get a string value from the registry."

    try:
        return win32api.RegQueryValue(base, path)
    except win32api.error:
        return None


def _remove_key(path, base=win32con.HKEY_CLASSES_ROOT):
    "Remove a string from the registry."

    try:
        win32api.RegDeleteKey(base, path)
    except win32api.error as xxx_todo_changeme1:
        (code, fn, msg) = xxx_todo_changeme1.args
        if code != winerror.ERROR_FILE_NOT_FOUND:
            raise win32api.error(code, fn, msg)


def recurse_delete_key(path, base=win32con.HKEY_CLASSES_ROOT):
    """Recursively delete registry keys.

    This is needed since you can't blast a key when subkeys exist.
    """
    try:
        h = win32api.RegOpenKey(base, path)
    except win32api.error as xxx_todo_changeme2:
        (code, fn, msg) = xxx_todo_changeme2.args
        if code != winerror.ERROR_FILE_NOT_FOUND:
            raise win32api.error(code, fn, msg)
    else:
        # parent key found and opened successfully. do some work, making sure
        # to always close the thing (error or no).
        try:
            # remove all of the subkeys
            while 1:
                try:
                    subkeyname = win32api.RegEnumKey(h, 0)
                except win32api.error as xxx_todo_changeme:
                    (code, fn, msg) = xxx_todo_changeme.args
                    if code != winerror.ERROR_NO_MORE_ITEMS:
                        raise win32api.error(code, fn, msg)
                    break
                recurse_delete_key(path + "\\" + subkeyname, base)

            # remove the parent key
            _remove_key(path, base)
        finally:
            win32api.RegCloseKey(h)


def _cat_registrar():
    return pythoncom.CoCreateInstance(
        pythoncom.CLSID_StdComponentCategoriesMgr,
        None,
        pythoncom.CLSCTX_INPROC_SERVER,
        pythoncom.IID_ICatRegister,
    )


def _find_localserver_exe(mustfind):
    if not sys.platform.startswith("win32"):
        return sys.executable
    if pythoncom.__file__.find("_d") < 0:
        exeBaseName = "pythonw.exe"
    else:
        exeBaseName = "pythonw_d.exe"
    # First see if in the same directory as this .EXE
    exeName = os.path.join(os.path.split(sys.executable)[0], exeBaseName)
    if not os.path.exists(exeName):
        # See if in our sys.prefix directory
        exeName = os.path.join(sys.prefix, exeBaseName)
    if not os.path.exists(exeName):
        # See if in our sys.prefix/pcbuild directory (for developers)
        if "64 bit" in sys.version:
            exeName = os.path.join(sys.prefix, "PCbuild", "amd64", exeBaseName)
        else:
            exeName = os.path.join(sys.prefix, "PCbuild", exeBaseName)
    if not os.path.exists(exeName):
        # See if the registry has some info.
        try:
            key = "SOFTWARE\\Python\\PythonCore\\%s\\InstallPath" % sys.winver
            path = win32api.RegQueryValue(win32con.HKEY_LOCAL_MACHINE, key)
            exeName = os.path.join(path, exeBaseName)
        except (AttributeError, win32api.error):
            pass
    if not os.path.exists(exeName):
        if mustfind:
            raise RuntimeError("Can not locate the program '%s'" % exeBaseName)
        return None
    return exeName


def _find_localserver_module():
    import win32com.server

    path = win32com.server.__path__[0]
    baseName = "localserver"
    pyfile = os.path.join(path, baseName + ".py")
    try:
        os.stat(pyfile)
    except os.error:
        # See if we have a compiled extension
        if __debug__:
            ext = ".pyc"
        else:
            ext = ".pyo"
        pyfile = os.path.join(path, baseName + ext)
        try:
            os.stat(pyfile)
        except os.error:
            raise RuntimeError(
                "Can not locate the Python module 'win32com.server.%s'" % baseName
            )
    return pyfile


def RegisterServer(
    clsid,
    pythonInstString=None,
    desc=None,
    progID=None,
    verProgID=None,
    defIcon=None,
    threadingModel="both",
    policy=None,
    catids=[],
    other={},
    addPyComCat=None,
    dispatcher=None,
    clsctx=None,
    addnPath=None,
):
    """Registers a Python object as a COM Server.  This enters almost all necessary
    information in the system registry, allowing COM to use the object.

    clsid -- The (unique) CLSID of the server.
    pythonInstString -- A string holding the instance name that will be created
                  whenever COM requests a new object.
    desc -- The description of the COM object.
    progID -- The user name of this object (eg, Word.Document)
    verProgId -- The user name of this version's implementation (eg Word.6.Document)
    defIcon -- The default icon for the object.
    threadingModel -- The threading model this object supports.
    policy -- The policy to use when creating this object.
    catids -- A list of category ID's this object belongs in.
    other -- A dictionary of extra items to be registered.
    addPyComCat -- A flag indicating if the object should be added to the list
             of Python servers installed on the machine.  If None (the default)
             then it will be registered when running from python source, but
             not registered if running in a frozen environment.
    dispatcher -- The dispatcher to use when creating this object.
    clsctx -- One of the CLSCTX_* constants.
    addnPath -- An additional path the COM framework will add to sys.path
                before attempting to create the object.
    """

    ### backwards-compat check
    ### Certain policies do not require a "class name", just the policy itself.
    if not pythonInstString and not policy:
        raise TypeError(
            "You must specify either the Python Class or Python Policy which implement the COM object."
        )

    keyNameRoot = "CLSID\\%s" % str(clsid)
    _set_string(keyNameRoot, desc)

    # Also register as an "Application" so DCOM etc all see us.
    _set_string("AppID\\%s" % clsid, progID)
    # Depending on contexts requested, register the specified server type.
    # Set default clsctx.
    if not clsctx:
        clsctx = pythoncom.CLSCTX_INPROC_SERVER | pythoncom.CLSCTX_LOCAL_SERVER
    # And if we are frozen, ignore the ones that don't make sense in this
    # context.
    if pythoncom.frozen:
        assert (
            sys.frozen
        ), "pythoncom is frozen, but sys.frozen is not set - don't know the context!"
        if sys.frozen == "dll":
            clsctx = clsctx & pythoncom.CLSCTX_INPROC_SERVER
        else:
            clsctx = clsctx & pythoncom.CLSCTX_LOCAL_SERVER
    # Now setup based on the clsctx left over.
    if clsctx & pythoncom.CLSCTX_INPROC_SERVER:
        # get the module to use for registration.
        # nod to Gordon's installer - if sys.frozen and sys.frozendllhandle
        # exist, then we are being registered via a DLL - use this DLL as the
        # file name.
        if pythoncom.frozen:
            if hasattr(sys, "frozendllhandle"):
                dllName = win32api.GetModuleFileName(sys.frozendllhandle)
            else:
                raise RuntimeError(
                    "We appear to have a frozen DLL, but I don't know the DLL to use"
                )
        else:
            # Normal case - running from .py file, so register pythoncom's DLL.
            # Although now we prefer a 'loader' DLL if it exists to avoid some
            # manifest issues (the 'loader' DLL has a manifest, but pythoncom does not)
            pythoncom_dir = os.path.dirname(pythoncom.__file__)
            suffix = "_d" if "_d" in pythoncom.__file__ else ""
            # Always register with the full path to the DLLs.
            loadername = os.path.join(
                pythoncom_dir,
                "pythoncomloader%d%d%s.dll"
                % (sys.version_info[0], sys.version_info[1], suffix),
            )
            dllName = loadername if os.path.isfile(loadername) else pythoncom.__file__

        _set_subkeys(
            keyNameRoot + "\\InprocServer32",
            {
                None: dllName,
                "ThreadingModel": threadingModel,
            },
        )
    else:  # Remove any old InProcServer32 registrations
        _remove_key(keyNameRoot + "\\InprocServer32")

    if clsctx & pythoncom.CLSCTX_LOCAL_SERVER:
        if pythoncom.frozen:
            # If we are frozen, we write "{exe} /Automate", just
            # like "normal" .EXEs do
            exeName = win32api.GetShortPathName(sys.executable)
            command = "%s /Automate" % (exeName,)
        else:
            # Running from .py sources - we need to write
            # 'python.exe win32com\server\localserver.py {clsid}"
            exeName = _find_localserver_exe(1)
            exeName = win32api.GetShortPathName(exeName)
            pyfile = _find_localserver_module()
            command = '%s "%s" %s' % (exeName, pyfile, str(clsid))
        _set_string(keyNameRoot + "\\LocalServer32", command)
    else:  # Remove any old LocalServer32 registrations
        _remove_key(keyNameRoot + "\\LocalServer32")

    if pythonInstString:
        _set_string(keyNameRoot + "\\PythonCOM", pythonInstString)
    else:
        _remove_key(keyNameRoot + "\\PythonCOM")
    if policy:
        _set_string(keyNameRoot + "\\PythonCOMPolicy", policy)
    else:
        _remove_key(keyNameRoot + "\\PythonCOMPolicy")

    if dispatcher:
        _set_string(keyNameRoot + "\\PythonCOMDispatcher", dispatcher)
    else:
        _remove_key(keyNameRoot + "\\PythonCOMDispatcher")

    if defIcon:
        _set_string(keyNameRoot + "\\DefaultIcon", defIcon)
    else:
        _remove_key(keyNameRoot + "\\DefaultIcon")

    if addnPath:
        _set_string(keyNameRoot + "\\PythonCOMPath", addnPath)
    else:
        _remove_key(keyNameRoot + "\\PythonCOMPath")

    if addPyComCat is None:
        addPyComCat = pythoncom.frozen == 0
    if addPyComCat:
        catids = catids + [CATID_PythonCOMServer]

    # Set up the implemented categories
    if catids:
        regCat = _cat_registrar()
        regCat.RegisterClassImplCategories(clsid, catids)

    # set up any other reg values they might have
    if other:
        for key, value in other.items():
            _set_string(keyNameRoot + "\\" + key, value)

    if progID:
        # set the progID as the most specific that was given to us
        if verProgID:
            _set_string(keyNameRoot + "\\ProgID", verProgID)
        else:
            _set_string(keyNameRoot + "\\ProgID", progID)

        # Set up the root entries - version independent.
        if desc:
            _set_string(progID, desc)
        _set_string(progID + "\\CLSID", str(clsid))

        # Set up the root entries - version dependent.
        if verProgID:
            # point from independent to the current version
            _set_string(progID + "\\CurVer", verProgID)

            # point to the version-independent one
            _set_string(keyNameRoot + "\\VersionIndependentProgID", progID)

            # set up the versioned progID
            if desc:
                _set_string(verProgID, desc)
            _set_string(verProgID + "\\CLSID", str(clsid))


def GetUnregisterServerKeys(clsid, progID=None, verProgID=None, customKeys=None):
    """Given a server, return a list of of ("key", root), which are keys recursively
    and uncondtionally deleted at unregister or uninstall time.
    """
    # remove the main CLSID registration
    ret = [("CLSID\\%s" % str(clsid), win32con.HKEY_CLASSES_ROOT)]
    # remove the versioned ProgID registration
    if verProgID:
        ret.append((verProgID, win32con.HKEY_CLASSES_ROOT))
    # blow away the independent ProgID. we can't leave it since we just
    # torched the class.
    ### could potentially check the CLSID... ?
    if progID:
        ret.append((progID, win32con.HKEY_CLASSES_ROOT))
    # The DCOM config tool may write settings to the AppID key for our CLSID
    ret.append(("AppID\\%s" % str(clsid), win32con.HKEY_CLASSES_ROOT))
    # Any custom keys?
    if customKeys:
        ret = ret + customKeys

    return ret


def UnregisterServer(clsid, progID=None, verProgID=None, customKeys=None):
    """Unregisters a Python COM server."""

    for args in GetUnregisterServerKeys(clsid, progID, verProgID, customKeys):
        recurse_delete_key(*args)

    ### it might be nice at some point to "roll back" the independent ProgID
    ### to an earlier version if one exists, and just blowing away the
    ### specified version of the ProgID (and its corresponding CLSID)
    ### another time, though...

    ### NOTE: ATL simply blows away the above three keys without the
    ### potential checks that I describe.  Assuming that defines the
    ### "standard" then we have no additional changes necessary.


def GetRegisteredServerOption(clsid, optionName):
    """Given a CLSID for a server and option name, return the option value"""
    keyNameRoot = "CLSID\\%s\\%s" % (str(clsid), str(optionName))
    return _get_string(keyNameRoot)


def _get(ob, attr, default=None):
    try:
        return getattr(ob, attr)
    except AttributeError:
        pass
    # look down sub-classes
    try:
        bases = ob.__bases__
    except AttributeError:
        # ob is not a class - no probs.
        return default
    for base in bases:
        val = _get(base, attr, None)
        if val is not None:
            return val
    return default


def RegisterClasses(*classes, **flags):
    quiet = "quiet" in flags and flags["quiet"]
    debugging = "debug" in flags and flags["debug"]
    for cls in classes:
        clsid = cls._reg_clsid_
        progID = _get(cls, "_reg_progid_")
        desc = _get(cls, "_reg_desc_", progID)
        spec = _get(cls, "_reg_class_spec_")
        verProgID = _get(cls, "_reg_verprogid_")
        defIcon = _get(cls, "_reg_icon_")
        threadingModel = _get(cls, "_reg_threading_", "both")
        catids = _get(cls, "_reg_catids_", [])
        options = _get(cls, "_reg_options_", {})
        policySpec = _get(cls, "_reg_policy_spec_")
        clsctx = _get(cls, "_reg_clsctx_")
        tlb_filename = _get(cls, "_reg_typelib_filename_")
        # default to being a COM category only when not frozen.
        addPyComCat = not _get(cls, "_reg_disable_pycomcat_", pythoncom.frozen != 0)
        addnPath = None
        if debugging:
            # If the class has a debugging dispatcher specified, use it, otherwise
            # use our default dispatcher.
            dispatcherSpec = _get(cls, "_reg_debug_dispatcher_spec_")
            if dispatcherSpec is None:
                dispatcherSpec = "win32com.server.dispatcher.DefaultDebugDispatcher"
            # And remember the debugging flag as servers may wish to use it at runtime.
            debuggingDesc = "(for debugging)"
            options["Debugging"] = "1"
        else:
            dispatcherSpec = _get(cls, "_reg_dispatcher_spec_")
            debuggingDesc = ""
            options["Debugging"] = "0"

        if spec is None:
            moduleName = cls.__module__
            if moduleName == "__main__":
                # Use argv[0] to determine the module name.
                try:
                    # Use the win32api to find the case-sensitive name
                    moduleName = os.path.splitext(
                        win32api.FindFiles(sys.argv[0])[0][8]
                    )[0]
                except (IndexError, win32api.error):
                    # Can't find the script file - the user must explicitely set the _reg_... attribute.
                    raise TypeError(
                        "Can't locate the script hosting the COM object - please set _reg_class_spec_ in your object"
                    )

            spec = moduleName + "." + cls.__name__
            # Frozen apps don't need their directory on sys.path
            if not pythoncom.frozen:
                scriptDir = os.path.split(sys.argv[0])[0]
                if not scriptDir:
                    scriptDir = "."
                addnPath = win32api.GetFullPathName(scriptDir)

        RegisterServer(
            clsid,
            spec,
            desc,
            progID,
            verProgID,
            defIcon,
            threadingModel,
            policySpec,
            catids,
            options,
            addPyComCat,
            dispatcherSpec,
            clsctx,
            addnPath,
        )
        if not quiet:
            print("Registered:", progID or spec, debuggingDesc)
        # Register the typelibrary
        if tlb_filename:
            tlb_filename = os.path.abspath(tlb_filename)
            typelib = pythoncom.LoadTypeLib(tlb_filename)
            pythoncom.RegisterTypeLib(typelib, tlb_filename)
            if not quiet:
                print("Registered type library:", tlb_filename)
    extra = flags.get("finalize_register")
    if extra:
        extra()


def UnregisterClasses(*classes, **flags):
    quiet = "quiet" in flags and flags["quiet"]
    for cls in classes:
        clsid = cls._reg_clsid_
        progID = _get(cls, "_reg_progid_")
        verProgID = _get(cls, "_reg_verprogid_")
        customKeys = _get(cls, "_reg_remove_keys_")
        unregister_typelib = _get(cls, "_reg_typelib_filename_") is not None

        UnregisterServer(clsid, progID, verProgID, customKeys)
        if not quiet:
            print("Unregistered:", progID or str(clsid))
        if unregister_typelib:
            tlb_guid = _get(cls, "_typelib_guid_")
            if tlb_guid is None:
                # I guess I could load the typelib, but they need the GUID anyway.
                print("Have typelib filename, but no GUID - can't unregister")
            else:
                major, minor = _get(cls, "_typelib_version_", (1, 0))
                lcid = _get(cls, "_typelib_lcid_", 0)
                try:
                    pythoncom.UnRegisterTypeLib(tlb_guid, major, minor, lcid)
                    if not quiet:
                        print("Unregistered type library")
                except pythoncom.com_error:
                    pass

    extra = flags.get("finalize_unregister")
    if extra:
        extra()


#
# Unregister info is for installers or external uninstallers.
# The WISE installer, for example firstly registers the COM server,
# then queries for the Unregister info, appending it to its
# install log.  Uninstalling the package will the uninstall the server
def UnregisterInfoClasses(*classes, **flags):
    ret = []
    for cls in classes:
        clsid = cls._reg_clsid_
        progID = _get(cls, "_reg_progid_")
        verProgID = _get(cls, "_reg_verprogid_")
        customKeys = _get(cls, "_reg_remove_keys_")

        ret = ret + GetUnregisterServerKeys(clsid, progID, verProgID, customKeys)
    return ret


# Attempt to 're-execute' our current process with elevation.
def ReExecuteElevated(flags):
    import tempfile

    import win32event  # we've already checked we are running XP above
    import win32process
    import winxpgui
    from win32com.shell import shellcon
    from win32com.shell.shell import ShellExecuteEx

    if not flags["quiet"]:
        print("Requesting elevation and retrying...")
    new_params = " ".join(['"' + a + '"' for a in sys.argv])
    # If we aren't already in unattended mode, we want our sub-process to
    # be.
    if not flags["unattended"]:
        new_params += " --unattended"
    # specifying the parent means the dialog is centered over our window,
    # which is a good usability clue.
    # hwnd is unlikely on the command-line, but flags may come from elsewhere
    hwnd = flags.get("hwnd", None)
    if hwnd is None:
        try:
            hwnd = winxpgui.GetConsoleWindow()
        except winxpgui.error:
            hwnd = 0
    # Redirect output so we give the user some clue what went wrong.  This
    # also means we need to use COMSPEC.  However, the "current directory"
    # appears to end up ignored - so we execute things via a temp batch file.
    tempbase = tempfile.mktemp("pycomserverreg")
    outfile = tempbase + ".out"
    batfile = tempbase + ".bat"

    # If registering from pythonwin, need to run python console instead since
    #  pythonwin will just open script for editting
    current_exe = os.path.split(sys.executable)[1].lower()
    exe_to_run = None
    if current_exe == "pythonwin.exe":
        exe_to_run = os.path.join(sys.prefix, "python.exe")
    elif current_exe == "pythonwin_d.exe":
        exe_to_run = os.path.join(sys.prefix, "python_d.exe")
    if not exe_to_run or not os.path.exists(exe_to_run):
        exe_to_run = sys.executable

    try:
        batf = open(batfile, "w")
        try:
            cwd = os.getcwd()
            print("@echo off", file=batf)
            # nothing is 'inherited' by the elevated process, including the
            # environment.  I wonder if we need to set more?
            print("set PYTHONPATH=%s" % os.environ.get("PYTHONPATH", ""), file=batf)
            # may be on a different drive - select that before attempting to CD.
            print(os.path.splitdrive(cwd)[0], file=batf)
            print('cd "%s"' % os.getcwd(), file=batf)
            print(
                '%s %s > "%s" 2>&1'
                % (win32api.GetShortPathName(exe_to_run), new_params, outfile),
                file=batf,
            )
        finally:
            batf.close()
        executable = os.environ.get("COMSPEC", "cmd.exe")
        rc = ShellExecuteEx(
            hwnd=hwnd,
            fMask=shellcon.SEE_MASK_NOCLOSEPROCESS,
            lpVerb="runas",
            lpFile=executable,
            lpParameters='/C "%s"' % batfile,
            nShow=win32con.SW_SHOW,
        )
        hproc = rc["hProcess"]
        win32event.WaitForSingleObject(hproc, win32event.INFINITE)
        exit_code = win32process.GetExitCodeProcess(hproc)
        outf = open(outfile)
        try:
            output = outf.read()
        finally:
            outf.close()

        if exit_code:
            # Even if quiet you get to see this message.
            print("Error: registration failed (exit code %s)." % exit_code)
        # if we are quiet then the output if likely to already be nearly
        # empty, so always print it.
        print(output, end=" ")
    finally:
        for f in (outfile, batfile):
            try:
                os.unlink(f)
            except os.error as exc:
                print("Failed to remove tempfile '%s': %s" % (f, exc))


def UseCommandLine(*classes, **flags):
    unregisterInfo = "--unregister_info" in sys.argv
    unregister = "--unregister" in sys.argv
    flags["quiet"] = flags.get("quiet", 0) or "--quiet" in sys.argv
    flags["debug"] = flags.get("debug", 0) or "--debug" in sys.argv
    flags["unattended"] = flags.get("unattended", 0) or "--unattended" in sys.argv
    if unregisterInfo:
        return UnregisterInfoClasses(*classes, **flags)
    try:
        if unregister:
            UnregisterClasses(*classes, **flags)
        else:
            RegisterClasses(*classes, **flags)
    except win32api.error as exc:
        # If we are on xp+ and have "access denied", retry using
        # ShellExecuteEx with 'runas' verb to force elevation (vista) and/or
        # admin login dialog (vista/xp)
        if (
            flags["unattended"]
            or exc.winerror != winerror.ERROR_ACCESS_DENIED
            or sys.getwindowsversion()[0] < 5
        ):
            raise
        ReExecuteElevated(flags)


def RegisterPyComCategory():
    """Register the Python COM Server component category."""
    regCat = _cat_registrar()
    regCat.RegisterCategories([(CATID_PythonCOMServer, 0x0409, "Python COM Server")])


if not pythoncom.frozen:
    try:
        win32api.RegQueryValue(
            win32con.HKEY_CLASSES_ROOT,
            "Component Categories\\%s" % CATID_PythonCOMServer,
        )
    except win32api.error:
        try:
            RegisterPyComCategory()
        except pythoncom.error:  # Error with the COM category manager - oh well.
            pass
