import unittest

import pythoncom
import win32api
import win32com.test.util
from win32com import storagecon


class TestEnum(win32com.test.util.TestCase):
    def testit(self):
        fname, tmp = win32api.GetTempFileName(win32api.GetTempPath(), "stg")
        m = storagecon.STGM_READWRITE | storagecon.STGM_SHARE_EXCLUSIVE
        ##  file, mode, format, attrs (always 0), IID (IStorage or IPropertySetStorage, storage options(only used with STGFMT_DOCFILE)
        pss = pythoncom.StgOpenStorageEx(
            fname, m, storagecon.STGFMT_FILE, 0, pythoncom.IID_IPropertySetStorage
        )
        ###                               {"Version":2,"reserved":0,"SectorSize":512,"TemplateFile":u'somefilename'})

        ## FMTID_SummaryInformation FMTID_DocSummaryInformation FMTID_UserDefinedProperties
        psuser = pss.Create(
            pythoncom.FMTID_UserDefinedProperties,
            pythoncom.IID_IPropertySetStorage,
            storagecon.PROPSETFLAG_DEFAULT,
            storagecon.STGM_READWRITE
            | storagecon.STGM_CREATE
            | storagecon.STGM_SHARE_EXCLUSIVE,
        )  ## its very picky about flag combinations!
        psuser.WriteMultiple((3, 4), ("hey", "bubba"))
        psuser.WritePropertyNames((3, 4), ("property3", "property4"))
        expected_summaries = []
        expected_summaries.append(("property3", 3, pythoncom.VT_BSTR))
        expected_summaries.append(("property4", 4, pythoncom.VT_BSTR))
        psuser = None

        pssum = pss.Create(
            pythoncom.FMTID_SummaryInformation,
            pythoncom.IID_IPropertySetStorage,
            storagecon.PROPSETFLAG_DEFAULT,
            storagecon.STGM_READWRITE
            | storagecon.STGM_CREATE
            | storagecon.STGM_SHARE_EXCLUSIVE,
        )
        pssum.WriteMultiple(
            (storagecon.PIDSI_AUTHOR, storagecon.PIDSI_COMMENTS), ("me", "comment")
        )

        pssum = None
        pss = None  ## doesn't seem to be a close or release method, and you can't even reopen it from the same process until previous object is gone

        pssread = pythoncom.StgOpenStorageEx(
            fname,
            storagecon.STGM_READ | storagecon.STGM_SHARE_EXCLUSIVE,
            storagecon.STGFMT_FILE,
            0,
            pythoncom.IID_IPropertySetStorage,
        )
        found_summaries = []
        for psstat in pssread:
            ps = pssread.Open(
                psstat[0], storagecon.STGM_READ | storagecon.STGM_SHARE_EXCLUSIVE
            )
            for p in ps:
                p_val = ps.ReadMultiple((p[1],))[0]
                if (p[1] == storagecon.PIDSI_AUTHOR and p_val == "me") or (
                    p[1] == storagecon.PIDSI_COMMENTS and p_val == "comment"
                ):
                    pass
                else:
                    self.fail("Uxexpected property %s/%s" % (p, p_val))
            ps = None
            ## FMTID_UserDefinedProperties can't exist without FMTID_DocSummaryInformation, and isn't returned independently from Enum
            ## also can't be open at same time
            if psstat[0] == pythoncom.FMTID_DocSummaryInformation:
                ps = pssread.Open(
                    pythoncom.FMTID_UserDefinedProperties,
                    storagecon.STGM_READ | storagecon.STGM_SHARE_EXCLUSIVE,
                )
                for p in ps:
                    found_summaries.append(p)
                ps = None
        psread = None
        expected_summaries.sort()
        found_summaries.sort()
        self.assertEqual(expected_summaries, found_summaries)


if __name__ == "__main__":
    unittest.main()
