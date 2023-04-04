"""General utility functions common to client and server.

  This module contains a collection of general purpose utility functions.
"""
import pythoncom
import win32api
import win32con


def IIDToInterfaceName(iid):
    """Converts an IID to a string interface name.

    Used primarily for debugging purposes, this allows a cryptic IID to
    be converted to a useful string name.  This will firstly look for interfaces
    known (ie, registered) by pythoncom.  If not known, it will look in the
    registry for a registered interface.

    iid -- An IID object.

    Result -- Always a string - either an interface name, or '<Unregistered interface>'
    """
    try:
        return pythoncom.ServerInterfaces[iid]
    except KeyError:
        try:
            try:
                return win32api.RegQueryValue(
                    win32con.HKEY_CLASSES_ROOT, "Interface\\%s" % iid
                )
            except win32api.error:
                pass
        except ImportError:
            pass
        return str(iid)
