# Tests for the win32security module.
import unittest

import ntsecuritycon
import pywintypes
import win32api
import win32con
import win32security
import winerror
from pywin32_testutil import TestSkipped, ob2memory, testmain


class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.pwr_sid = win32security.LookupAccountName("", "Power Users")[0]
        try:
            self.admin_sid = win32security.LookupAccountName("", "Administrator")[0]
        except pywintypes.error as exc:
            # in automation we see:
            # pywintypes.error: (1332, 'LookupAccountName', 'No mapping between account names and security IDs was done.')
            if exc.winerror != winerror.ERROR_NONE_MAPPED:
                raise
            self.admin_sid = None

    def tearDown(self):
        pass

    def testEqual(self):
        if self.admin_sid is None:
            raise TestSkipped("No 'Administrator' account is available")
        self.assertEqual(
            win32security.LookupAccountName("", "Administrator")[0],
            win32security.LookupAccountName("", "Administrator")[0],
        )

    def testNESID(self):
        self.assertTrue(self.pwr_sid == self.pwr_sid)
        if self.admin_sid:
            self.assertTrue(self.pwr_sid != self.admin_sid)

    def testNEOther(self):
        self.assertTrue(self.pwr_sid != None)
        self.assertTrue(None != self.pwr_sid)
        self.assertFalse(self.pwr_sid == None)
        self.assertFalse(None == self.pwr_sid)
        self.assertNotEqual(None, self.pwr_sid)

    def testSIDInDict(self):
        d = dict(foo=self.pwr_sid)
        self.assertEqual(d["foo"], self.pwr_sid)

    def testBuffer(self):
        if self.admin_sid is None:
            raise TestSkipped("No 'Administrator' account is available")
        self.assertEqual(
            ob2memory(win32security.LookupAccountName("", "Administrator")[0]),
            ob2memory(win32security.LookupAccountName("", "Administrator")[0]),
        )

    def testMemory(self):
        pwr_sid = self.pwr_sid
        admin_sid = self.admin_sid
        sd1 = win32security.SECURITY_DESCRIPTOR()
        sd2 = win32security.SECURITY_DESCRIPTOR()
        sd3 = win32security.SECURITY_DESCRIPTOR()
        dacl = win32security.ACL()
        dacl.AddAccessAllowedAce(
            win32security.ACL_REVISION, win32con.GENERIC_READ, pwr_sid
        )
        if admin_sid is not None:
            dacl.AddAccessAllowedAce(
                win32security.ACL_REVISION, win32con.GENERIC_ALL, admin_sid
            )
        sd4 = win32security.SECURITY_DESCRIPTOR()
        sacl = win32security.ACL()
        if admin_sid is not None:
            sacl.AddAuditAccessAce(
                win32security.ACL_REVISION, win32con.DELETE, admin_sid, 1, 1
            )
        sacl.AddAuditAccessAce(
            win32security.ACL_REVISION, win32con.GENERIC_ALL, pwr_sid, 1, 1
        )
        for x in range(0, 200000):
            if admin_sid is not None:
                sd1.SetSecurityDescriptorOwner(admin_sid, 0)
            sd2.SetSecurityDescriptorGroup(pwr_sid, 0)
            sd3.SetSecurityDescriptorDacl(1, dacl, 0)
            sd4.SetSecurityDescriptorSacl(1, sacl, 0)


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.ds_handle = None
        try:
            # saving the handle means the other test itself should bind faster.
            self.ds_handle = win32security.DsBind()
        except win32security.error as exc:
            if exc.winerror != winerror.ERROR_NO_SUCH_DOMAIN:
                raise
            raise TestSkipped(exc)

    def tearDown(self):
        if self.ds_handle is not None:
            self.ds_handle.close()


class TestDS(DomainTests):
    def testDsGetDcName(self):
        # Not sure what we can actually test here!  At least calling it
        # does something :)
        win32security.DsGetDcName()

    def testDsListServerInfo(self):
        # again, not checking much, just exercising the code.
        h = win32security.DsBind()
        for status, ignore, site in win32security.DsListSites(h):
            for status, ignore, server in win32security.DsListServersInSite(h, site):
                info = win32security.DsListInfoForServer(h, server)
            for status, ignore, domain in win32security.DsListDomainsInSite(h, site):
                pass

    def testDsCrackNames(self):
        h = win32security.DsBind()
        fmt_offered = ntsecuritycon.DS_FQDN_1779_NAME
        name = win32api.GetUserNameEx(fmt_offered)
        result = win32security.DsCrackNames(h, 0, fmt_offered, fmt_offered, (name,))
        self.assertEqual(name, result[0][2])

    def testDsCrackNamesSyntax(self):
        # Do a syntax check only - that allows us to avoid binding.
        # But must use DS_CANONICAL_NAME (or _EX)
        expected = win32api.GetUserNameEx(win32api.NameCanonical)
        fmt_offered = ntsecuritycon.DS_FQDN_1779_NAME
        name = win32api.GetUserNameEx(fmt_offered)
        result = win32security.DsCrackNames(
            None,
            ntsecuritycon.DS_NAME_FLAG_SYNTACTICAL_ONLY,
            fmt_offered,
            ntsecuritycon.DS_CANONICAL_NAME,
            (name,),
        )
        self.assertEqual(expected, result[0][2])


class TestTranslate(DomainTests):
    def _testTranslate(self, fmt_from, fmt_to):
        name = win32api.GetUserNameEx(fmt_from)
        expected = win32api.GetUserNameEx(fmt_to)
        got = win32security.TranslateName(name, fmt_from, fmt_to)
        self.assertEqual(got, expected)

    def testTranslate1(self):
        self._testTranslate(win32api.NameFullyQualifiedDN, win32api.NameSamCompatible)

    def testTranslate2(self):
        self._testTranslate(win32api.NameSamCompatible, win32api.NameFullyQualifiedDN)

    def testTranslate3(self):
        self._testTranslate(win32api.NameFullyQualifiedDN, win32api.NameUniqueId)

    def testTranslate4(self):
        self._testTranslate(win32api.NameUniqueId, win32api.NameFullyQualifiedDN)


if __name__ == "__main__":
    testmain()
