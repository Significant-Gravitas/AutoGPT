import os

import pythoncom
import pywintypes
import win32api
import win32com
import win32com.client
import win32com.client.dynamic
import win32com.server.util
import win32ui
from pywin32_testutil import str2bytes
from pywintypes import Unicode
from win32com import storagecon
from win32com.axcontrol import axcontrol
from win32com.test.util import CheckClean

S_OK = 0


import win32timezone

now = win32timezone.now()


class LockBytes:
    _public_methods_ = [
        "ReadAt",
        "WriteAt",
        "Flush",
        "SetSize",
        "LockRegion",
        "UnlockRegion",
        "Stat",
    ]
    _com_interfaces_ = [pythoncom.IID_ILockBytes]

    def __init__(self, data=""):
        self.data = str2bytes(data)
        self.ctime = now
        self.mtime = now
        self.atime = now

    def ReadAt(self, offset, cb):
        print("ReadAt")
        result = self.data[offset : offset + cb]
        return result

    def WriteAt(self, offset, data):
        print("WriteAt " + str(offset))
        print("len " + str(len(data)))
        print("data:")
        # print data
        if len(self.data) >= offset:
            newdata = self.data[0:offset] + data
        print(len(newdata))
        if len(self.data) >= offset + len(data):
            newdata = newdata + self.data[offset + len(data) :]
        print(len(newdata))
        self.data = newdata
        return len(data)

    def Flush(self, whatsthis=0):
        print("Flush" + str(whatsthis))
        fname = os.path.join(win32api.GetTempPath(), "persist.doc")
        open(fname, "wb").write(self.data)
        return S_OK

    def SetSize(self, size):
        print("Set Size" + str(size))
        if size > len(self.data):
            self.data = self.data + str2bytes("\000" * (size - len(self.data)))
        else:
            self.data = self.data[0:size]
        return S_OK

    def LockRegion(self, offset, size, locktype):
        print("LockRegion")

    def UnlockRegion(self, offset, size, locktype):
        print("UnlockRegion")

    def Stat(self, statflag):
        print("returning Stat " + str(statflag))
        return (
            "PyMemBytes",
            storagecon.STGTY_LOCKBYTES,
            len(self.data),
            self.mtime,
            self.ctime,
            self.atime,
            storagecon.STGM_DIRECT | storagecon.STGM_READWRITE | storagecon.STGM_CREATE,
            storagecon.STGM_SHARE_EXCLUSIVE,
            "{00020905-0000-0000-C000-000000000046}",
            0,  # statebits ?
            0,
        )


class OleClientSite:
    _public_methods_ = [
        "SaveObject",
        "GetMoniker",
        "GetContainer",
        "ShowObject",
        "OnShowWindow",
        "RequestNewObjectLayout",
    ]
    _com_interfaces_ = [axcontrol.IID_IOleClientSite]

    def __init__(self, data=""):
        self.IPersistStorage = None
        self.IStorage = None

    def SetIPersistStorage(self, IPersistStorage):
        self.IPersistStorage = IPersistStorage

    def SetIStorage(self, IStorage):
        self.IStorage = IStorage

    def SaveObject(self):
        print("SaveObject")
        if self.IPersistStorage != None and self.IStorage != None:
            self.IPersistStorage.Save(self.IStorage, 1)
            self.IStorage.Commit(0)
        return S_OK

    def GetMoniker(self, dwAssign, dwWhichMoniker):
        print("GetMoniker " + str(dwAssign) + " " + str(dwWhichMoniker))

    def GetContainer(self):
        print("GetContainer")

    def ShowObject(self):
        print("ShowObject")

    def OnShowWindow(self, fShow):
        print("ShowObject" + str(fShow))

    def RequestNewObjectLayout(self):
        print("RequestNewObjectLayout")


def test():
    # create a LockBytes object and
    # wrap it as a COM object
    #       import win32com.server.dispatcher
    lbcom = win32com.server.util.wrap(
        LockBytes(), pythoncom.IID_ILockBytes
    )  # , useDispatcher=win32com.server.dispatcher.DispatcherWin32trace)

    # create a structured storage on the ILockBytes object
    stcom = pythoncom.StgCreateDocfileOnILockBytes(
        lbcom,
        storagecon.STGM_DIRECT
        | storagecon.STGM_CREATE
        | storagecon.STGM_READWRITE
        | storagecon.STGM_SHARE_EXCLUSIVE,
        0,
    )

    # create our ClientSite
    ocs = OleClientSite()
    # wrap it as a COM object
    ocscom = win32com.server.util.wrap(ocs, axcontrol.IID_IOleClientSite)

    # create a Word OLE Document, connect it to our site and our storage
    oocom = axcontrol.OleCreate(
        "{00020906-0000-0000-C000-000000000046}",
        axcontrol.IID_IOleObject,
        0,
        (0,),
        ocscom,
        stcom,
    )

    mf = win32ui.GetMainFrame()
    hwnd = mf.GetSafeHwnd()

    # Set the host and document name
    # for unknown reason document name becomes hostname, and document name
    # is not set, debugged it, but don't know where the problem is?
    oocom.SetHostNames("OTPython", "This is Cool")

    # activate the OLE document
    oocom.DoVerb(-1, ocscom, 0, hwnd, mf.GetWindowRect())

    # set the hostnames again
    oocom.SetHostNames("OTPython2", "ThisisCool2")

    # get IDispatch of Word
    doc = win32com.client.Dispatch(oocom.QueryInterface(pythoncom.IID_IDispatch))

    # get IPersistStorage of Word
    dpcom = oocom.QueryInterface(pythoncom.IID_IPersistStorage)

    # let our ClientSite know the interfaces
    ocs.SetIPersistStorage(dpcom)
    ocs.SetIStorage(stcom)

    # use IDispatch to do the Office Word test
    # pasted from TestOffice.py

    wrange = doc.Range()
    for i in range(10):
        wrange.InsertAfter("Hello from Python %d\n" % i)
    paras = doc.Paragraphs
    for i in range(len(paras)):
        paras[i]().Font.ColorIndex = i + 1
        paras[i]().Font.Size = 12 + (4 * i)
    # XXX - note that
    # for para in paras:
    #       para().Font...
    # doesnt seem to work - no error, just doesnt work
    # Should check if it works for VB!

    dpcom.Save(stcom, 0)
    dpcom.HandsOffStorage()
    #       oocom.Close(axcontrol.OLECLOSE_NOSAVE) # or OLECLOSE_SAVEIFDIRTY, but it fails???

    # Save the ILockBytes data to "persist2.doc"
    lbcom.Flush()

    # exiting Winword will automatically update the ILockBytes data
    # and flush it to "%TEMP%\persist.doc"
    doc.Application.Quit()


if __name__ == "__main__":
    test()
    pythoncom.CoUninitialize()
    CheckClean()
