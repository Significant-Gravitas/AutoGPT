# General purpose service utilities, both for standard Python scripts,
# and for for Python programs which run as services...
#
# Note that most utility functions here will raise win32api.error's
# (which is win32service.error, pywintypes.error, etc)
# when things go wrong - eg, not enough permissions to hit the
# registry etc.

import importlib
import os
import sys
import warnings

import pywintypes
import win32api
import win32con
import win32service
import winerror

_d = "_d" if "_d.pyd" in importlib.machinery.EXTENSION_SUFFIXES else ""
error = RuntimeError


# Returns the full path to an executable for hosting a Python service - typically
# 'pythonservice.exe'
# * If you pass a param and it exists as a file, you'll get the abs path back
# * Otherwise we'll use the param instead of 'pythonservice.exe', and we will
#   look for it.
def LocatePythonServiceExe(exe=None):
    if not exe and hasattr(sys, "frozen"):
        # If py2exe etc calls this with no exe, default is current exe,
        # and all setup is their problem :)
        return sys.executable

    if exe and os.path.isfile(exe):
        return win32api.GetFullPathName(exe)

    # We are confused if we aren't now looking for our default. But if that
    # exists as specified we assume it's good.
    exe = f"pythonservice{_d}.exe"
    if os.path.isfile(exe):
        return win32api.GetFullPathName(exe)

    # Now we are searching for the .exe
    # We are going to want it here.
    correct = os.path.join(sys.exec_prefix, exe)
    # Even if that file already exists, we copy the one installed by pywin32
    # in-case it was upgraded.
    # pywin32 installed it next to win32service.pyd (but we can't run it from there)
    maybe = os.path.join(os.path.dirname(win32service.__file__), exe)
    if os.path.exists(maybe):
        print(f"copying host exe '{maybe}' -> '{correct}'")
        win32api.CopyFile(maybe, correct)

    if not os.path.exists(correct):
        raise error(f"Can't find '{correct}'")

    # If pywintypes.dll isn't next to us, or at least next to pythonXX.dll,
    # there's a good chance the service will not run. That's usually copied by
    # `pywin32_postinstall`, but putting it next to the python DLL seems
    # reasonable.
    # (Unlike the .exe above, we don't unconditionally copy this, and possibly
    # copy it to a different place. Doesn't seem a good reason for that!?)
    python_dll = win32api.GetModuleFileName(sys.dllhandle)
    pyw = f"pywintypes{sys.version_info[0]}{sys.version_info[1]}{_d}.dll"
    correct_pyw = os.path.join(os.path.dirname(python_dll), pyw)

    if not os.path.exists(correct_pyw):
        print(f"copying helper dll '{pywintypes.__file__}' -> '{correct_pyw}'")
        win32api.CopyFile(pywintypes.__file__, correct_pyw)

    return correct


def _GetServiceShortName(longName):
    # looks up a services name
    # from the display name
    # Thanks to Andy McKay for this code.
    access = (
        win32con.KEY_READ | win32con.KEY_ENUMERATE_SUB_KEYS | win32con.KEY_QUERY_VALUE
    )
    hkey = win32api.RegOpenKey(
        win32con.HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Services", 0, access
    )
    num = win32api.RegQueryInfoKey(hkey)[0]
    longName = longName.lower()
    # loop through number of subkeys
    for x in range(0, num):
        # find service name, open subkey
        svc = win32api.RegEnumKey(hkey, x)
        skey = win32api.RegOpenKey(hkey, svc, 0, access)
        try:
            # find display name
            thisName = str(win32api.RegQueryValueEx(skey, "DisplayName")[0])
            if thisName.lower() == longName:
                return svc
        except win32api.error:
            # in case there is no key called DisplayName
            pass
    return None


# Open a service given either it's long or short name.
def SmartOpenService(hscm, name, access):
    try:
        return win32service.OpenService(hscm, name, access)
    except win32api.error as details:
        if details.winerror not in [
            winerror.ERROR_SERVICE_DOES_NOT_EXIST,
            winerror.ERROR_INVALID_NAME,
        ]:
            raise
    name = win32service.GetServiceKeyName(hscm, name)
    return win32service.OpenService(hscm, name, access)


def LocateSpecificServiceExe(serviceName):
    # Return the .exe name of any service.
    hkey = win32api.RegOpenKey(
        win32con.HKEY_LOCAL_MACHINE,
        "SYSTEM\\CurrentControlSet\\Services\\%s" % (serviceName),
        0,
        win32con.KEY_ALL_ACCESS,
    )
    try:
        return win32api.RegQueryValueEx(hkey, "ImagePath")[0]
    finally:
        hkey.Close()


def InstallPerfmonForService(serviceName, iniName, dllName=None):
    # If no DLL name, look it up in the INI file name
    if not dllName:  # May be empty string!
        dllName = win32api.GetProfileVal("Python", "dll", "", iniName)
    # Still not found - look for the standard one in the same dir as win32service.pyd
    if not dllName:
        try:
            tryName = os.path.join(
                os.path.split(win32service.__file__)[0], "perfmondata.dll"
            )
            if os.path.isfile(tryName):
                dllName = tryName
        except AttributeError:
            # Frozen app? - anyway, can't find it!
            pass
    if not dllName:
        raise ValueError("The name of the performance DLL must be available")
    dllName = win32api.GetFullPathName(dllName)
    # Now setup all the required "Performance" entries.
    hkey = win32api.RegOpenKey(
        win32con.HKEY_LOCAL_MACHINE,
        "SYSTEM\\CurrentControlSet\\Services\\%s" % (serviceName),
        0,
        win32con.KEY_ALL_ACCESS,
    )
    try:
        subKey = win32api.RegCreateKey(hkey, "Performance")
        try:
            win32api.RegSetValueEx(subKey, "Library", 0, win32con.REG_SZ, dllName)
            win32api.RegSetValueEx(
                subKey, "Open", 0, win32con.REG_SZ, "OpenPerformanceData"
            )
            win32api.RegSetValueEx(
                subKey, "Close", 0, win32con.REG_SZ, "ClosePerformanceData"
            )
            win32api.RegSetValueEx(
                subKey, "Collect", 0, win32con.REG_SZ, "CollectPerformanceData"
            )
        finally:
            win32api.RegCloseKey(subKey)
    finally:
        win32api.RegCloseKey(hkey)
    # Now do the "Lodctr" thang...

    try:
        import perfmon

        path, fname = os.path.split(iniName)
        oldPath = os.getcwd()
        if path:
            os.chdir(path)
        try:
            perfmon.LoadPerfCounterTextStrings("python.exe " + fname)
        finally:
            os.chdir(oldPath)
    except win32api.error as details:
        print("The service was installed OK, but the performance monitor")
        print("data could not be loaded.", details)


def _GetCommandLine(exeName, exeArgs):
    if exeArgs is not None:
        return exeName + " " + exeArgs
    else:
        return exeName


def InstallService(
    pythonClassString,
    serviceName,
    displayName,
    startType=None,
    errorControl=None,
    bRunInteractive=0,
    serviceDeps=None,
    userName=None,
    password=None,
    exeName=None,
    perfMonIni=None,
    perfMonDll=None,
    exeArgs=None,
    description=None,
    delayedstart=None,
):
    # Handle the default arguments.
    if startType is None:
        startType = win32service.SERVICE_DEMAND_START
    serviceType = win32service.SERVICE_WIN32_OWN_PROCESS
    if bRunInteractive:
        serviceType = serviceType | win32service.SERVICE_INTERACTIVE_PROCESS
    if errorControl is None:
        errorControl = win32service.SERVICE_ERROR_NORMAL

    exeName = '"%s"' % LocatePythonServiceExe(exeName)
    commandLine = _GetCommandLine(exeName, exeArgs)
    hscm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ALL_ACCESS)
    try:
        hs = win32service.CreateService(
            hscm,
            serviceName,
            displayName,
            win32service.SERVICE_ALL_ACCESS,  # desired access
            serviceType,  # service type
            startType,
            errorControl,  # error control type
            commandLine,
            None,
            0,
            serviceDeps,
            userName,
            password,
        )
        if description is not None:
            try:
                win32service.ChangeServiceConfig2(
                    hs, win32service.SERVICE_CONFIG_DESCRIPTION, description
                )
            except NotImplementedError:
                pass  ## ChangeServiceConfig2 and description do not exist on NT
        if delayedstart is not None:
            try:
                win32service.ChangeServiceConfig2(
                    hs,
                    win32service.SERVICE_CONFIG_DELAYED_AUTO_START_INFO,
                    delayedstart,
                )
            except (win32service.error, NotImplementedError):
                ## delayed start only exists on Vista and later - warn only when trying to set delayed to True
                warnings.warn("Delayed Start not available on this system")
        win32service.CloseServiceHandle(hs)
    finally:
        win32service.CloseServiceHandle(hscm)
    InstallPythonClassString(pythonClassString, serviceName)
    # If I have performance monitor info to install, do that.
    if perfMonIni is not None:
        InstallPerfmonForService(serviceName, perfMonIni, perfMonDll)


def ChangeServiceConfig(
    pythonClassString,
    serviceName,
    startType=None,
    errorControl=None,
    bRunInteractive=0,
    serviceDeps=None,
    userName=None,
    password=None,
    exeName=None,
    displayName=None,
    perfMonIni=None,
    perfMonDll=None,
    exeArgs=None,
    description=None,
    delayedstart=None,
):
    # Before doing anything, remove any perfmon counters.
    try:
        import perfmon

        perfmon.UnloadPerfCounterTextStrings("python.exe " + serviceName)
    except (ImportError, win32api.error):
        pass

    # The EXE location may have changed
    exeName = '"%s"' % LocatePythonServiceExe(exeName)

    # Handle the default arguments.
    if startType is None:
        startType = win32service.SERVICE_NO_CHANGE
    if errorControl is None:
        errorControl = win32service.SERVICE_NO_CHANGE

    hscm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ALL_ACCESS)
    serviceType = win32service.SERVICE_WIN32_OWN_PROCESS
    if bRunInteractive:
        serviceType = serviceType | win32service.SERVICE_INTERACTIVE_PROCESS
    commandLine = _GetCommandLine(exeName, exeArgs)
    try:
        hs = SmartOpenService(hscm, serviceName, win32service.SERVICE_ALL_ACCESS)
        try:
            win32service.ChangeServiceConfig(
                hs,
                serviceType,  # service type
                startType,
                errorControl,  # error control type
                commandLine,
                None,
                0,
                serviceDeps,
                userName,
                password,
                displayName,
            )
            if description is not None:
                try:
                    win32service.ChangeServiceConfig2(
                        hs, win32service.SERVICE_CONFIG_DESCRIPTION, description
                    )
                except NotImplementedError:
                    pass  ## ChangeServiceConfig2 and description do not exist on NT
            if delayedstart is not None:
                try:
                    win32service.ChangeServiceConfig2(
                        hs,
                        win32service.SERVICE_CONFIG_DELAYED_AUTO_START_INFO,
                        delayedstart,
                    )
                except (win32service.error, NotImplementedError):
                    ## Delayed start only exists on Vista and later.  On Nt, will raise NotImplementedError since ChangeServiceConfig2
                    ## doensn't exist.  On Win2k and XP, will fail with ERROR_INVALID_LEVEL
                    ## Warn only if trying to set delayed to True
                    if delayedstart:
                        warnings.warn("Delayed Start not available on this system")
        finally:
            win32service.CloseServiceHandle(hs)
    finally:
        win32service.CloseServiceHandle(hscm)
    InstallPythonClassString(pythonClassString, serviceName)
    # If I have performance monitor info to install, do that.
    if perfMonIni is not None:
        InstallPerfmonForService(serviceName, perfMonIni, perfMonDll)


def InstallPythonClassString(pythonClassString, serviceName):
    # Now setup our Python specific entries.
    if pythonClassString:
        key = win32api.RegCreateKey(
            win32con.HKEY_LOCAL_MACHINE,
            "System\\CurrentControlSet\\Services\\%s\\PythonClass" % serviceName,
        )
        try:
            win32api.RegSetValue(key, None, win32con.REG_SZ, pythonClassString)
        finally:
            win32api.RegCloseKey(key)


# Utility functions for Services, to allow persistant properties.
def SetServiceCustomOption(serviceName, option, value):
    try:
        serviceName = serviceName._svc_name_
    except AttributeError:
        pass
    key = win32api.RegCreateKey(
        win32con.HKEY_LOCAL_MACHINE,
        "System\\CurrentControlSet\\Services\\%s\\Parameters" % serviceName,
    )
    try:
        if type(value) == type(0):
            win32api.RegSetValueEx(key, option, 0, win32con.REG_DWORD, value)
        else:
            win32api.RegSetValueEx(key, option, 0, win32con.REG_SZ, value)
    finally:
        win32api.RegCloseKey(key)


def GetServiceCustomOption(serviceName, option, defaultValue=None):
    # First param may also be a service class/instance.
    # This allows services to pass "self"
    try:
        serviceName = serviceName._svc_name_
    except AttributeError:
        pass
    key = win32api.RegCreateKey(
        win32con.HKEY_LOCAL_MACHINE,
        "System\\CurrentControlSet\\Services\\%s\\Parameters" % serviceName,
    )
    try:
        try:
            return win32api.RegQueryValueEx(key, option)[0]
        except win32api.error:  # No value.
            return defaultValue
    finally:
        win32api.RegCloseKey(key)


def RemoveService(serviceName):
    try:
        import perfmon

        perfmon.UnloadPerfCounterTextStrings("python.exe " + serviceName)
    except (ImportError, win32api.error):
        pass

    hscm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ALL_ACCESS)
    try:
        hs = SmartOpenService(hscm, serviceName, win32service.SERVICE_ALL_ACCESS)
        win32service.DeleteService(hs)
        win32service.CloseServiceHandle(hs)
    finally:
        win32service.CloseServiceHandle(hscm)

    import win32evtlogutil

    try:
        win32evtlogutil.RemoveSourceFromRegistry(serviceName)
    except win32api.error:
        pass


def ControlService(serviceName, code, machine=None):
    hscm = win32service.OpenSCManager(machine, None, win32service.SC_MANAGER_ALL_ACCESS)
    try:
        hs = SmartOpenService(hscm, serviceName, win32service.SERVICE_ALL_ACCESS)
        try:
            status = win32service.ControlService(hs, code)
        finally:
            win32service.CloseServiceHandle(hs)
    finally:
        win32service.CloseServiceHandle(hscm)
    return status


def __FindSvcDeps(findName):
    if type(findName) is pywintypes.UnicodeType:
        findName = str(findName)
    dict = {}
    k = win32api.RegOpenKey(
        win32con.HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Services"
    )
    num = 0
    while 1:
        try:
            svc = win32api.RegEnumKey(k, num)
        except win32api.error:
            break
        num = num + 1
        sk = win32api.RegOpenKey(k, svc)
        try:
            deps, typ = win32api.RegQueryValueEx(sk, "DependOnService")
        except win32api.error:
            deps = ()
        for dep in deps:
            dep = dep.lower()
            dep_on = dict.get(dep, [])
            dep_on.append(svc)
            dict[dep] = dep_on

    return __ResolveDeps(findName, dict)


def __ResolveDeps(findName, dict):
    items = dict.get(findName.lower(), [])
    retList = []
    for svc in items:
        retList.insert(0, svc)
        retList = __ResolveDeps(svc, dict) + retList
    return retList


def WaitForServiceStatus(serviceName, status, waitSecs, machine=None):
    """Waits for the service to return the specified status.  You
    should have already requested the service to enter that state"""
    for i in range(waitSecs * 4):
        now_status = QueryServiceStatus(serviceName, machine)[1]
        if now_status == status:
            break
        win32api.Sleep(250)
    else:
        raise pywintypes.error(
            winerror.ERROR_SERVICE_REQUEST_TIMEOUT,
            "QueryServiceStatus",
            win32api.FormatMessage(winerror.ERROR_SERVICE_REQUEST_TIMEOUT)[:-2],
        )


def __StopServiceWithTimeout(hs, waitSecs=30):
    try:
        status = win32service.ControlService(hs, win32service.SERVICE_CONTROL_STOP)
    except pywintypes.error as exc:
        if exc.winerror != winerror.ERROR_SERVICE_NOT_ACTIVE:
            raise
    for i in range(waitSecs):
        status = win32service.QueryServiceStatus(hs)
        if status[1] == win32service.SERVICE_STOPPED:
            break
        win32api.Sleep(1000)
    else:
        raise pywintypes.error(
            winerror.ERROR_SERVICE_REQUEST_TIMEOUT,
            "ControlService",
            win32api.FormatMessage(winerror.ERROR_SERVICE_REQUEST_TIMEOUT)[:-2],
        )


def StopServiceWithDeps(serviceName, machine=None, waitSecs=30):
    # Stop a service recursively looking for dependant services
    hscm = win32service.OpenSCManager(machine, None, win32service.SC_MANAGER_ALL_ACCESS)
    try:
        deps = __FindSvcDeps(serviceName)
        for dep in deps:
            hs = win32service.OpenService(hscm, dep, win32service.SERVICE_ALL_ACCESS)
            try:
                __StopServiceWithTimeout(hs, waitSecs)
            finally:
                win32service.CloseServiceHandle(hs)
        # Now my service!
        hs = win32service.OpenService(
            hscm, serviceName, win32service.SERVICE_ALL_ACCESS
        )
        try:
            __StopServiceWithTimeout(hs, waitSecs)
        finally:
            win32service.CloseServiceHandle(hs)

    finally:
        win32service.CloseServiceHandle(hscm)


def StopService(serviceName, machine=None):
    return ControlService(serviceName, win32service.SERVICE_CONTROL_STOP, machine)


def StartService(serviceName, args=None, machine=None):
    hscm = win32service.OpenSCManager(machine, None, win32service.SC_MANAGER_ALL_ACCESS)
    try:
        hs = SmartOpenService(hscm, serviceName, win32service.SERVICE_ALL_ACCESS)
        try:
            win32service.StartService(hs, args)
        finally:
            win32service.CloseServiceHandle(hs)
    finally:
        win32service.CloseServiceHandle(hscm)


def RestartService(serviceName, args=None, waitSeconds=30, machine=None):
    "Stop the service, and then start it again (with some tolerance for allowing it to stop.)"
    try:
        StopService(serviceName, machine)
    except pywintypes.error as exc:
        # Allow only "service not running" error
        if exc.winerror != winerror.ERROR_SERVICE_NOT_ACTIVE:
            raise
    # Give it a few goes, as the service may take time to stop
    for i in range(waitSeconds):
        try:
            StartService(serviceName, args, machine)
            break
        except pywintypes.error as exc:
            if exc.winerror != winerror.ERROR_SERVICE_ALREADY_RUNNING:
                raise
            win32api.Sleep(1000)
    else:
        print("Gave up waiting for the old service to stop!")


def _DebugCtrlHandler(evt):
    if evt in (win32con.CTRL_C_EVENT, win32con.CTRL_BREAK_EVENT):
        assert g_debugService
        print("Stopping debug service.")
        g_debugService.SvcStop()
        return True
    return False


def DebugService(cls, argv=[]):
    # Run a service in "debug" mode.  Re-implements what pythonservice.exe
    # does when it sees a "-debug" param.
    # Currently only used by "frozen" (ie, py2exe) programs (but later may
    # end up being used for all services should we ever remove
    # pythonservice.exe)
    import servicemanager

    global g_debugService

    print("Debugging service %s - press Ctrl+C to stop." % (cls._svc_name_,))
    servicemanager.Debugging(True)
    servicemanager.PrepareToHostSingle(cls)
    g_debugService = cls(argv)
    # Setup a ctrl+c handler to simulate a "stop"
    win32api.SetConsoleCtrlHandler(_DebugCtrlHandler, True)
    try:
        g_debugService.SvcRun()
    finally:
        win32api.SetConsoleCtrlHandler(_DebugCtrlHandler, False)
        servicemanager.Debugging(False)
        g_debugService = None


def GetServiceClassString(cls, argv=None):
    if argv is None:
        argv = sys.argv
    import pickle

    modName = pickle.whichmodule(cls, cls.__name__)
    if modName == "__main__":
        try:
            fname = win32api.GetFullPathName(argv[0])
            path = os.path.split(fname)[0]
            # Eaaaahhhh - sometimes this will be a short filename, which causes
            # problems with 1.5.1 and the silly filename case rule.
            filelist = win32api.FindFiles(fname)
            # win32api.FindFiles will not detect files in a zip or exe. If list is empty,
            # skip the test and hope the file really exists.
            if len(filelist) != 0:
                # Get the long name
                fname = os.path.join(path, filelist[0][8])
        except win32api.error:
            raise error(
                "Could not resolve the path name '%s' to a full path" % (argv[0])
            )
        modName = os.path.splitext(fname)[0]
    return modName + "." + cls.__name__


def QueryServiceStatus(serviceName, machine=None):
    hscm = win32service.OpenSCManager(machine, None, win32service.SC_MANAGER_CONNECT)
    try:
        hs = SmartOpenService(hscm, serviceName, win32service.SERVICE_QUERY_STATUS)
        try:
            status = win32service.QueryServiceStatus(hs)
        finally:
            win32service.CloseServiceHandle(hs)
    finally:
        win32service.CloseServiceHandle(hscm)
    return status


def usage():
    try:
        fname = os.path.split(sys.argv[0])[1]
    except:
        fname = sys.argv[0]
    print(
        "Usage: '%s [options] install|update|remove|start [...]|stop|restart [...]|debug [...]'"
        % fname
    )
    print("Options for 'install' and 'update' commands only:")
    print(" --username domain\\username : The Username the service is to run under")
    print(" --password password : The password for the username")
    print(
        " --startup [manual|auto|disabled|delayed] : How the service starts, default = manual"
    )
    print(" --interactive : Allow the service to interact with the desktop.")
    print(
        " --perfmonini file: .ini file to use for registering performance monitor data"
    )
    print(" --perfmondll file: .dll file to use when querying the service for")
    print("   performance data, default = perfmondata.dll")
    print("Options for 'start' and 'stop' commands only:")
    print(" --wait seconds: Wait for the service to actually start or stop.")
    print("                 If you specify --wait with the 'stop' option, the service")
    print("                 and all dependent services will be stopped, each waiting")
    print("                 the specified period.")
    sys.exit(1)


def HandleCommandLine(
    cls,
    serviceClassString=None,
    argv=None,
    customInstallOptions="",
    customOptionHandler=None,
):
    """Utility function allowing services to process the command line.

    Allows standard commands such as 'start', 'stop', 'debug', 'install' etc.

    Install supports 'standard' command line options prefixed with '--', such as
    --username, --password, etc.  In addition,
    the function allows custom command line options to be handled by the calling function.
    """
    err = 0

    if argv is None:
        argv = sys.argv

    if len(argv) <= 1:
        usage()

    serviceName = cls._svc_name_
    serviceDisplayName = cls._svc_display_name_
    if serviceClassString is None:
        serviceClassString = GetServiceClassString(cls)

    # Pull apart the command line
    import getopt

    try:
        opts, args = getopt.getopt(
            argv[1:],
            customInstallOptions,
            [
                "password=",
                "username=",
                "startup=",
                "perfmonini=",
                "perfmondll=",
                "interactive",
                "wait=",
            ],
        )
    except getopt.error as details:
        print(details)
        usage()
    userName = None
    password = None
    perfMonIni = perfMonDll = None
    startup = None
    delayedstart = None
    interactive = None
    waitSecs = 0
    for opt, val in opts:
        if opt == "--username":
            userName = val
        elif opt == "--password":
            password = val
        elif opt == "--perfmonini":
            perfMonIni = val
        elif opt == "--perfmondll":
            perfMonDll = val
        elif opt == "--interactive":
            interactive = 1
        elif opt == "--startup":
            map = {
                "manual": win32service.SERVICE_DEMAND_START,
                "auto": win32service.SERVICE_AUTO_START,
                "delayed": win32service.SERVICE_AUTO_START,  ## ChangeServiceConfig2 called later
                "disabled": win32service.SERVICE_DISABLED,
            }
            try:
                startup = map[val.lower()]
            except KeyError:
                print("'%s' is not a valid startup option" % val)
            if val.lower() == "delayed":
                delayedstart = True
            elif val.lower() == "auto":
                delayedstart = False
            ## else no change
        elif opt == "--wait":
            try:
                waitSecs = int(val)
            except ValueError:
                print("--wait must specify an integer number of seconds.")
                usage()

    arg = args[0]
    knownArg = 0
    # First we process all arguments which pass additional args on
    if arg == "start":
        knownArg = 1
        print("Starting service %s" % (serviceName))
        try:
            StartService(serviceName, args[1:])
            if waitSecs:
                WaitForServiceStatus(
                    serviceName, win32service.SERVICE_RUNNING, waitSecs
                )
        except win32service.error as exc:
            print("Error starting service: %s" % exc.strerror)
            err = exc.winerror

    elif arg == "restart":
        knownArg = 1
        print("Restarting service %s" % (serviceName))
        RestartService(serviceName, args[1:])
        if waitSecs:
            WaitForServiceStatus(serviceName, win32service.SERVICE_RUNNING, waitSecs)

    elif arg == "debug":
        knownArg = 1
        if not hasattr(sys, "frozen"):
            # non-frozen services use pythonservice.exe which handles a
            # -debug option
            svcArgs = " ".join(args[1:])
            try:
                exeName = LocateSpecificServiceExe(serviceName)
            except win32api.error as exc:
                if exc.winerror == winerror.ERROR_FILE_NOT_FOUND:
                    print("The service does not appear to be installed.")
                    print("Please install the service before debugging it.")
                    sys.exit(1)
                raise
            try:
                os.system("%s -debug %s %s" % (exeName, serviceName, svcArgs))
            # ^C is used to kill the debug service.  Sometimes Python also gets
            # interrupted - ignore it...
            except KeyboardInterrupt:
                pass
        else:
            # py2exe services don't use pythonservice - so we simulate
            # debugging here.
            DebugService(cls, args)

    if not knownArg and len(args) != 1:
        usage()  # the rest of the cmds don't take addn args

    if arg == "install":
        knownArg = 1
        try:
            serviceDeps = cls._svc_deps_
        except AttributeError:
            serviceDeps = None
        try:
            exeName = cls._exe_name_
        except AttributeError:
            exeName = None  # Default to PythonService.exe
        try:
            exeArgs = cls._exe_args_
        except AttributeError:
            exeArgs = None
        try:
            description = cls._svc_description_
        except AttributeError:
            description = None
        print("Installing service %s" % (serviceName,))
        # Note that we install the service before calling the custom option
        # handler, so if the custom handler fails, we have an installed service (from NT's POV)
        # but is unlikely to work, as the Python code controlling it failed.  Therefore
        # we remove the service if the first bit works, but the second doesnt!
        try:
            InstallService(
                serviceClassString,
                serviceName,
                serviceDisplayName,
                serviceDeps=serviceDeps,
                startType=startup,
                bRunInteractive=interactive,
                userName=userName,
                password=password,
                exeName=exeName,
                perfMonIni=perfMonIni,
                perfMonDll=perfMonDll,
                exeArgs=exeArgs,
                description=description,
                delayedstart=delayedstart,
            )
            if customOptionHandler:
                customOptionHandler(*(opts,))
            print("Service installed")
        except win32service.error as exc:
            if exc.winerror == winerror.ERROR_SERVICE_EXISTS:
                arg = "update"  # Fall through to the "update" param!
            else:
                print(
                    "Error installing service: %s (%d)" % (exc.strerror, exc.winerror)
                )
                err = exc.winerror
        except ValueError as msg:  # Can be raised by custom option handler.
            print("Error installing service: %s" % str(msg))
            err = -1
            # xxx - maybe I should remove after _any_ failed install - however,
            # xxx - it may be useful to help debug to leave the service as it failed.
            # xxx - We really _must_ remove as per the comments above...
            # As we failed here, remove the service, so the next installation
            # attempt works.
            try:
                RemoveService(serviceName)
            except win32api.error:
                print("Warning - could not remove the partially installed service.")

    if arg == "update":
        knownArg = 1
        try:
            serviceDeps = cls._svc_deps_
        except AttributeError:
            serviceDeps = None
        try:
            exeName = cls._exe_name_
        except AttributeError:
            exeName = None  # Default to PythonService.exe
        try:
            exeArgs = cls._exe_args_
        except AttributeError:
            exeArgs = None
        try:
            description = cls._svc_description_
        except AttributeError:
            description = None
        print("Changing service configuration")
        try:
            ChangeServiceConfig(
                serviceClassString,
                serviceName,
                serviceDeps=serviceDeps,
                startType=startup,
                bRunInteractive=interactive,
                userName=userName,
                password=password,
                exeName=exeName,
                displayName=serviceDisplayName,
                perfMonIni=perfMonIni,
                perfMonDll=perfMonDll,
                exeArgs=exeArgs,
                description=description,
                delayedstart=delayedstart,
            )
            if customOptionHandler:
                customOptionHandler(*(opts,))
            print("Service updated")
        except win32service.error as exc:
            print(
                "Error changing service configuration: %s (%d)"
                % (exc.strerror, exc.winerror)
            )
            err = exc.winerror

    elif arg == "remove":
        knownArg = 1
        print("Removing service %s" % (serviceName))
        try:
            RemoveService(serviceName)
            print("Service removed")
        except win32service.error as exc:
            print("Error removing service: %s (%d)" % (exc.strerror, exc.winerror))
            err = exc.winerror
    elif arg == "stop":
        knownArg = 1
        print("Stopping service %s" % (serviceName))
        try:
            if waitSecs:
                StopServiceWithDeps(serviceName, waitSecs=waitSecs)
            else:
                StopService(serviceName)
        except win32service.error as exc:
            print("Error stopping service: %s (%d)" % (exc.strerror, exc.winerror))
            err = exc.winerror
    if not knownArg:
        err = -1
        print("Unknown command - '%s'" % arg)
        usage()
    return err


#
# Useful base class to build services from.
#
class ServiceFramework:
    # Required Attributes:
    # _svc_name_ = The service name
    # _svc_display_name_ = The service display name

    # Optional Attributes:
    _svc_deps_ = None  # sequence of service names on which this depends
    _exe_name_ = None  # Default to PythonService.exe
    _exe_args_ = None  # Default to no arguments
    _svc_description_ = (
        None  # Only exists on Windows 2000 or later, ignored on windows NT
    )

    def __init__(self, args):
        import servicemanager

        self.ssh = servicemanager.RegisterServiceCtrlHandler(
            args[0], self.ServiceCtrlHandlerEx, True
        )
        servicemanager.SetEventSourceName(self._svc_name_)
        self.checkPoint = 0

    def GetAcceptedControls(self):
        # Setup the service controls we accept based on our attributes. Note
        # that if you need to handle controls via SvcOther[Ex](), you must
        # override this.
        accepted = 0
        if hasattr(self, "SvcStop"):
            accepted = accepted | win32service.SERVICE_ACCEPT_STOP
        if hasattr(self, "SvcPause") and hasattr(self, "SvcContinue"):
            accepted = accepted | win32service.SERVICE_ACCEPT_PAUSE_CONTINUE
        if hasattr(self, "SvcShutdown"):
            accepted = accepted | win32service.SERVICE_ACCEPT_SHUTDOWN
        return accepted

    def ReportServiceStatus(
        self, serviceStatus, waitHint=5000, win32ExitCode=0, svcExitCode=0
    ):
        if self.ssh is None:  # Debugging!
            return
        if serviceStatus == win32service.SERVICE_START_PENDING:
            accepted = 0
        else:
            accepted = self.GetAcceptedControls()

        if serviceStatus in [
            win32service.SERVICE_RUNNING,
            win32service.SERVICE_STOPPED,
        ]:
            checkPoint = 0
        else:
            self.checkPoint = self.checkPoint + 1
            checkPoint = self.checkPoint

        # Now report the status to the control manager
        status = (
            win32service.SERVICE_WIN32_OWN_PROCESS,
            serviceStatus,
            accepted,  # dwControlsAccepted,
            win32ExitCode,  # dwWin32ExitCode;
            svcExitCode,  # dwServiceSpecificExitCode;
            checkPoint,  # dwCheckPoint;
            waitHint,
        )
        win32service.SetServiceStatus(self.ssh, status)

    def SvcInterrogate(self):
        # Assume we are running, and everyone is happy.
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)

    def SvcOther(self, control):
        try:
            print("Unknown control status - %d" % control)
        except IOError:
            # services may not have a valid stdout!
            pass

    def ServiceCtrlHandler(self, control):
        return self.ServiceCtrlHandlerEx(control, 0, None)

    # The 'Ex' functions, which take additional params
    def SvcOtherEx(self, control, event_type, data):
        # The default here is to call self.SvcOther as that is the old behaviour.
        # If you want to take advantage of the extra data, override this method
        return self.SvcOther(control)

    def ServiceCtrlHandlerEx(self, control, event_type, data):
        if control == win32service.SERVICE_CONTROL_STOP:
            return self.SvcStop()
        elif control == win32service.SERVICE_CONTROL_PAUSE:
            return self.SvcPause()
        elif control == win32service.SERVICE_CONTROL_CONTINUE:
            return self.SvcContinue()
        elif control == win32service.SERVICE_CONTROL_INTERROGATE:
            return self.SvcInterrogate()
        elif control == win32service.SERVICE_CONTROL_SHUTDOWN:
            return self.SvcShutdown()
        else:
            return self.SvcOtherEx(control, event_type, data)

    def SvcRun(self):
        # This is the entry point the C framework calls when the Service is
        # started. Your Service class should implement SvcDoRun().
        # Or you can override this method for more control over the Service
        # statuses reported to the SCM.

        # If this method raises an exception, the C framework will detect this
        # and report a SERVICE_STOPPED status with a non-zero error code.

        self.ReportServiceStatus(win32service.SERVICE_RUNNING)
        self.SvcDoRun()
        # Once SvcDoRun terminates, the service has stopped.
        # We tell the SCM the service is still stopping - the C framework
        # will automatically tell the SCM it has stopped when this returns.
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
