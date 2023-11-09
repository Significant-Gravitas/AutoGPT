# A sample implementation of IEmptyVolumeCache - see
# http://msdn2.microsoft.com/en-us/library/aa969271.aspx for an overview.
#
# * Execute this script to register the handler
# * Start the "disk cleanup" tool - look for "pywin32 compiled files"
import os
import stat
import sys

import pythoncom
import win32gui
import winerror
from win32com.server.exception import COMException
from win32com.shell import shell, shellcon

# Our shell extension.
IEmptyVolumeCache_Methods = (
    "Initialize GetSpaceUsed Purge ShowProperties Deactivate".split()
)
IEmptyVolumeCache2_Methods = "InitializeEx".split()

ico = os.path.join(sys.prefix, "py.ico")
if not os.path.isfile(ico):
    ico = os.path.join(sys.prefix, "PC", "py.ico")
if not os.path.isfile(ico):
    ico = None
    print("Can't find python.ico - no icon will be installed")


class EmptyVolumeCache:
    _reg_progid_ = "Python.ShellExtension.EmptyVolumeCache"
    _reg_desc_ = "Python Sample Shell Extension (disk cleanup)"
    _reg_clsid_ = "{EADD0777-2968-4c72-A999-2BF5F756259C}"
    _reg_icon_ = ico
    _com_interfaces_ = [shell.IID_IEmptyVolumeCache, shell.IID_IEmptyVolumeCache2]
    _public_methods_ = IEmptyVolumeCache_Methods + IEmptyVolumeCache2_Methods

    def Initialize(self, hkey, volume, flags):
        # This should never be called, except on win98.
        print("Unless we are on 98, Initialize call is unexpected!")
        raise COMException(hresult=winerror.E_NOTIMPL)

    def InitializeEx(self, hkey, volume, key_name, flags):
        # Must return a tuple of:
        # (display_name, description, button_name, flags)
        print("InitializeEx called with", hkey, volume, key_name, flags)
        self.volume = volume
        if flags & shellcon.EVCF_SETTINGSMODE:
            print("We are being run on a schedule")
            # In this case, "because there is no opportunity for user
            # feedback, only those files that are extremely safe to clean up
            # should be touched. You should ignore the initialization
            # method's pcwszVolume parameter and clean unneeded files
            # regardless of what drive they are on."
            self.volume = None  # flag as 'any disk will do'
        elif flags & shellcon.EVCF_OUTOFDISKSPACE:
            # In this case, "the handler should be aggressive about deleting
            # files, even if it results in a performance loss. However, the
            # handler obviously should not delete files that would cause an
            # application to fail or the user to lose data."
            print("We are being run as we are out of disk-space")
        else:
            # This case is not documented - we are guessing :)
            print("We are being run because the user asked")

        # For the sake of demo etc, we tell the shell to only show us when
        # there are > 0 bytes available.  Our GetSpaceUsed will check the
        # volume, so will return 0 when we are on a different disk
        flags = shellcon.EVCF_DONTSHOWIFZERO | shellcon.EVCF_ENABLEBYDEFAULT

        return (
            "pywin32 compiled files",
            "Removes all .pyc and .pyo files in the pywin32 directories",
            "click me!",
            flags,
        )

    def _GetDirectories(self):
        root_dir = os.path.abspath(os.path.dirname(os.path.dirname(win32gui.__file__)))
        if self.volume is not None and not root_dir.lower().startswith(
            self.volume.lower()
        ):
            return []
        return [
            os.path.join(root_dir, p)
            for p in ("win32", "win32com", "win32comext", "isapi")
        ]

    def _WalkCallback(self, arg, directory, files):
        # callback function for os.path.walk - no need to be member, but its
        # close to the callers :)
        callback, total_list = arg
        for file in files:
            fqn = os.path.join(directory, file).lower()
            if file.endswith(".pyc") or file.endswith(".pyo"):
                # See below - total_list == None means delete files,
                # otherwise it is a list where the result is stored. Its a
                # list simply due to the way os.walk works - only [0] is
                # referenced
                if total_list is None:
                    print("Deleting file", fqn)
                    # Should do callback.PurgeProcess - left as an exercise :)
                    os.remove(fqn)
                else:
                    total_list[0] += os.stat(fqn)[stat.ST_SIZE]
                    # and callback to the tool
                    if callback:
                        # for the sake of seeing the progress bar do its thing,
                        # we take longer than we need to...
                        # ACK - for some bizarre reason this screws up the XP
                        # cleanup manager - clues welcome!! :)
                        ## print "Looking in", directory, ", but waiting a while..."
                        ## time.sleep(3)
                        # now do it
                        used = total_list[0]
                        callback.ScanProgress(used, 0, "Looking at " + fqn)

    def GetSpaceUsed(self, callback):
        total = [0]  # See _WalkCallback above
        try:
            for d in self._GetDirectories():
                os.path.walk(d, self._WalkCallback, (callback, total))
                print("After looking in", d, "we have", total[0], "bytes")
        except pythoncom.error as exc:
            # This will be raised by the callback when the user selects 'cancel'.
            if exc.hresult != winerror.E_ABORT:
                raise  # that's the documented error code!
            print("User cancelled the operation")
        return total[0]

    def Purge(self, amt_to_free, callback):
        print("Purging", amt_to_free, "bytes...")
        # we ignore amt_to_free - it is generally what we returned for
        # GetSpaceUsed
        try:
            for d in self._GetDirectories():
                os.path.walk(d, self._WalkCallback, (callback, None))
        except pythoncom.error as exc:
            # This will be raised by the callback when the user selects 'cancel'.
            if exc.hresult != winerror.E_ABORT:
                raise  # that's the documented error code!
            print("User cancelled the operation")

    def ShowProperties(self, hwnd):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def Deactivate(self):
        print("Deactivate called")
        return 0


def DllRegisterServer():
    # Also need to register specially in:
    # HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Explorer\VolumeCaches
    # See link at top of file.
    import winreg

    kn = r"Software\Microsoft\Windows\CurrentVersion\Explorer\VolumeCaches\%s" % (
        EmptyVolumeCache._reg_desc_,
    )
    key = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, kn)
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, EmptyVolumeCache._reg_clsid_)


def DllUnregisterServer():
    import winreg

    kn = r"Software\Microsoft\Windows\CurrentVersion\Explorer\VolumeCaches\%s" % (
        EmptyVolumeCache._reg_desc_,
    )
    try:
        key = winreg.DeleteKey(winreg.HKEY_LOCAL_MACHINE, kn)
    except WindowsError as details:
        import errno

        if details.errno != errno.ENOENT:
            raise
    print(EmptyVolumeCache._reg_desc_, "unregistration complete.")


if __name__ == "__main__":
    from win32com.server import register

    register.UseCommandLine(
        EmptyVolumeCache,
        finalize_register=DllRegisterServer,
        finalize_unregister=DllUnregisterServer,
    )
