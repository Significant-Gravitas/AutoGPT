import unittest

import netbios
import win32api
import win32wnet
from pywin32_testutil import str2bytes

RESOURCE_CONNECTED = 0x00000001
RESOURCE_GLOBALNET = 0x00000002
RESOURCE_REMEMBERED = 0x00000003
RESOURCE_RECENT = 0x00000004
RESOURCE_CONTEXT = 0x00000005
RESOURCETYPE_ANY = 0x00000000
RESOURCETYPE_DISK = 0x00000001
RESOURCETYPE_PRINT = 0x00000002
RESOURCETYPE_RESERVED = 0x00000008
RESOURCETYPE_UNKNOWN = 0xFFFFFFFF
RESOURCEUSAGE_CONNECTABLE = 0x00000001
RESOURCEUSAGE_CONTAINER = 0x00000002
RESOURCEDISPLAYTYPE_GENERIC = 0x00000000
RESOURCEDISPLAYTYPE_DOMAIN = 0x00000001
RESOURCEDISPLAYTYPE_SERVER = 0x00000002
RESOURCEDISPLAYTYPE_SHARE = 0x00000003


NETRESOURCE_attributes = [
    ("dwScope", int),
    ("dwType", int),
    ("dwDisplayType", int),
    ("dwUsage", int),
    ("lpLocalName", str),
    ("lpRemoteName", str),
    ("lpComment", str),
    ("lpProvider", str),
]

NCB_attributes = [
    ("Command", int),
    ("Retcode", int),
    ("Lsn", int),
    ("Num", int),
    #    ("Bufflen", int), - read-only
    ("Callname", str),
    ("Name", str),
    ("Rto", int),
    ("Sto", int),
    ("Lana_num", int),
    ("Cmd_cplt", int),
    ("Event", int),
    ("Post", int),
]


class TestCase(unittest.TestCase):
    def testGetUser(self):
        self.assertEqual(win32api.GetUserName(), win32wnet.WNetGetUser())

    def _checkItemAttributes(self, item, attrs):
        for attr, typ in attrs:
            val = getattr(item, attr)
            if typ is int:
                self.assertTrue(
                    type(val) in (int,), "Attr %r has value %r" % (attr, val)
                )
                new_val = val + 1
            elif typ is str:
                if val is not None:
                    # on py2k, must be string or unicode.  py3k must be string or bytes.
                    self.assertTrue(
                        type(val) in (str, str), "Attr %r has value %r" % (attr, val)
                    )
                    new_val = val + " new value"
                else:
                    new_val = "new value"
            else:
                self.fail("Don't know what %s is" % (typ,))
            # set the attribute just to make sure we can.
            setattr(item, attr, new_val)

    def testNETRESOURCE(self):
        nr = win32wnet.NETRESOURCE()
        self._checkItemAttributes(nr, NETRESOURCE_attributes)

    def testWNetEnumResource(self):
        handle = win32wnet.WNetOpenEnum(RESOURCE_GLOBALNET, RESOURCETYPE_ANY, 0, None)
        try:
            while 1:
                items = win32wnet.WNetEnumResource(handle, 0)
                if len(items) == 0:
                    break
                for item in items:
                    self._checkItemAttributes(item, NETRESOURCE_attributes)
        finally:
            handle.Close()

    def testNCB(self):
        ncb = win32wnet.NCB()
        self._checkItemAttributes(ncb, NCB_attributes)

    def testNetbios(self):
        # taken from the demo code in netbios.py
        ncb = win32wnet.NCB()
        ncb.Command = netbios.NCBENUM
        la_enum = netbios.LANA_ENUM()
        ncb.Buffer = la_enum
        rc = win32wnet.Netbios(ncb)
        self.assertEqual(rc, 0)
        for i in range(la_enum.length):
            ncb.Reset()
            ncb.Command = netbios.NCBRESET
            ncb.Lana_num = netbios.byte_to_int(la_enum.lana[i])
            rc = Netbios(ncb)
            self.assertEqual(rc, 0)
            ncb.Reset()
            ncb.Command = netbios.NCBASTAT
            ncb.Lana_num = byte_to_int(la_enum.lana[i])
            ncb.Callname = str2bytes("*               ")  # ensure bytes on py2x and 3k
            adapter = netbios.ADAPTER_STATUS()
            ncb.Buffer = adapter
            Netbios(ncb)
            # expect 6 bytes in the mac address.
            self.assertTrue(len(adapter.adapter_address), 6)

    def iterConnectableShares(self):
        nr = win32wnet.NETRESOURCE()
        nr.dwScope = RESOURCE_GLOBALNET
        nr.dwUsage = RESOURCEUSAGE_CONTAINER
        nr.lpRemoteName = "\\\\" + win32api.GetComputerName()

        handle = win32wnet.WNetOpenEnum(RESOURCE_GLOBALNET, RESOURCETYPE_ANY, 0, nr)
        while 1:
            items = win32wnet.WNetEnumResource(handle, 0)
            if len(items) == 0:
                break
            for item in items:
                if item.dwDisplayType == RESOURCEDISPLAYTYPE_SHARE:
                    yield item

    def findUnusedDriveLetter(self):
        existing = [
            x[0].lower() for x in win32api.GetLogicalDriveStrings().split("\0") if x
        ]
        handle = win32wnet.WNetOpenEnum(RESOURCE_REMEMBERED, RESOURCETYPE_DISK, 0, None)
        try:
            while 1:
                items = win32wnet.WNetEnumResource(handle, 0)
                if len(items) == 0:
                    break
                xtra = [i.lpLocalName[0].lower() for i in items if i.lpLocalName]
                existing.extend(xtra)
        finally:
            handle.Close()
        for maybe in "defghijklmnopqrstuvwxyz":
            if maybe not in existing:
                return maybe
        self.fail("All drive mappings are taken?")

    def testAddConnection(self):
        localName = self.findUnusedDriveLetter() + ":"
        for share in self.iterConnectableShares():
            share.lpLocalName = localName
            win32wnet.WNetAddConnection2(share)
            win32wnet.WNetCancelConnection2(localName, 0, 0)
            break

    def testAddConnectionOld(self):
        localName = self.findUnusedDriveLetter() + ":"
        for share in self.iterConnectableShares():
            win32wnet.WNetAddConnection2(share.dwType, localName, share.lpRemoteName)
            win32wnet.WNetCancelConnection2(localName, 0, 0)
            break


if __name__ == "__main__":
    unittest.main()
