# TestExchange = Exchange Server Dump
# Note that this code uses "CDO", which is unlikely to get the best choice.
# You should use the Outlook object model, or
# the win32com.mapi examples for a low-level interface.

import os

import pythoncom
from win32com.client import constants, gencache

ammodule = None  # was the generated module!


def GetDefaultProfileName():
    import win32api
    import win32con

    try:
        key = win32api.RegOpenKey(
            win32con.HKEY_CURRENT_USER,
            "Software\\Microsoft\\Windows NT\\CurrentVersion\\Windows Messaging Subsystem\\Profiles",
        )
        try:
            return win32api.RegQueryValueEx(key, "DefaultProfile")[0]
        finally:
            key.Close()
    except win32api.error:
        return None


#
# Recursive dump of folders.
#
def DumpFolder(folder, indent=0):
    print(" " * indent, folder.Name)
    folders = folder.Folders
    folder = folders.GetFirst()
    while folder:
        DumpFolder(folder, indent + 1)
        folder = folders.GetNext()


def DumpFolders(session):
    try:
        infostores = session.InfoStores
    except AttributeError:
        # later outlook?
        store = session.DefaultStore
        folder = store.GetRootFolder()
        DumpFolder(folder)
        return

    print(infostores)
    print("There are %d infostores" % infostores.Count)
    for i in range(infostores.Count):
        infostore = infostores[i + 1]
        print("Infostore = ", infostore.Name)
        try:
            folder = infostore.RootFolder
        except pythoncom.com_error as details:
            hr, msg, exc, arg = details
            # -2147221219 == MAPI_E_FAILONEPROVIDER - a single provider temporarily not available.
            if exc and exc[-1] == -2147221219:
                print("This info store is currently not available")
                continue
        DumpFolder(folder)


# Build a dictionary of property tags, so I can reverse look-up
#
PropTagsById = {}
if ammodule:
    for name, val in ammodule.constants.__dict__.items():
        PropTagsById[val] = name


def TestAddress(session):
    #       entry = session.GetAddressEntry("Skip")
    #       print entry
    pass


def TestUser(session):
    ae = session.CurrentUser
    fields = getattr(ae, "Fields", [])
    print("User has %d fields" % len(fields))
    for f in range(len(fields)):
        field = fields[f + 1]
        try:
            id = PropTagsById[field.ID]
        except KeyError:
            id = field.ID
        print("%s/%s=%s" % (field.Name, id, field.Value))


def test():
    import win32com.client

    oldcwd = os.getcwd()
    try:
        session = gencache.EnsureDispatch("MAPI.Session")
        try:
            session.Logon(GetDefaultProfileName())
        except pythoncom.com_error as details:
            print("Could not log on to MAPI:", details)
            return
    except pythoncom.error:
        # no mapi.session - let's try outlook
        app = gencache.EnsureDispatch("Outlook.Application")
        session = app.Session

    try:
        TestUser(session)
        TestAddress(session)
        DumpFolders(session)
    finally:
        session.Logoff()
        # It appears Exchange will change the cwd on us :(
        os.chdir(oldcwd)


if __name__ == "__main__":
    from .util import CheckClean

    test()
    CheckClean()
