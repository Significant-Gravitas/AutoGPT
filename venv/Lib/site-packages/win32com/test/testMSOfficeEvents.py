# OfficeEvents - test/demonstrate events with Word and Excel.
import msvcrt
import sys
import threading
import time
import types

import pythoncom
from win32com.client import Dispatch, DispatchWithEvents

stopEvent = threading.Event()


def TestExcel():
    class ExcelEvents:
        def OnNewWorkbook(self, wb):
            if type(wb) != types.InstanceType:
                raise RuntimeError(
                    "The transformer doesnt appear to have translated this for us!"
                )
            self.seen_events["OnNewWorkbook"] = None

        def OnWindowActivate(self, wb, wn):
            if type(wb) != types.InstanceType or type(wn) != types.InstanceType:
                raise RuntimeError(
                    "The transformer doesnt appear to have translated this for us!"
                )
            self.seen_events["OnWindowActivate"] = None

        def OnWindowDeactivate(self, wb, wn):
            self.seen_events["OnWindowDeactivate"] = None

        def OnSheetDeactivate(self, sh):
            self.seen_events["OnSheetDeactivate"] = None

        def OnSheetBeforeDoubleClick(self, Sh, Target, Cancel):
            if Target.Column % 2 == 0:
                print("You can double-click there...")
            else:
                print("You can not double-click there...")
                # This function is a void, so the result ends up in
                # the only ByRef - Cancel.
                return 1

    class WorkbookEvents:
        def OnActivate(self):
            print("workbook OnActivate")

        def OnBeforeRightClick(self, Target, Cancel):
            print("It's a Worksheet Event")

    e = DispatchWithEvents("Excel.Application", ExcelEvents)
    e.seen_events = {}
    e.Visible = 1
    book = e.Workbooks.Add()
    book = DispatchWithEvents(book, WorkbookEvents)
    print("Have book", book)
    #    sheet = e.Worksheets(1)
    #    sheet = DispatchWithEvents(sheet, WorksheetEvents)

    print("Double-click in a few of the Excel cells...")
    print("Press any key when finished with Excel, or wait 10 seconds...")
    if not _WaitForFinish(e, 10):
        e.Quit()
    if not _CheckSeenEvents(e, ["OnNewWorkbook", "OnWindowActivate"]):
        sys.exit(1)


def TestWord():
    class WordEvents:
        def OnDocumentChange(self):
            self.seen_events["OnDocumentChange"] = None

        def OnWindowActivate(self, doc, wn):
            self.seen_events["OnWindowActivate"] = None

        def OnQuit(self):
            self.seen_events["OnQuit"] = None
            stopEvent.set()

    w = DispatchWithEvents("Word.Application", WordEvents)
    w.seen_events = {}
    w.Visible = 1
    w.Documents.Add()
    print("Press any key when finished with Word, or wait 10 seconds...")
    if not _WaitForFinish(w, 10):
        w.Quit()
    if not _CheckSeenEvents(w, ["OnDocumentChange", "OnWindowActivate"]):
        sys.exit(1)


def _WaitForFinish(ob, timeout):
    end = time.time() + timeout
    while 1:
        if msvcrt.kbhit():
            msvcrt.getch()
            break
        pythoncom.PumpWaitingMessages()
        stopEvent.wait(0.2)
        if stopEvent.isSet():
            stopEvent.clear()
            break
        try:
            if not ob.Visible:
                # Gone invisible - we need to pretend we timed
                # out, so the app is quit.
                return 0
        except pythoncom.com_error:
            # Excel is busy (eg, editing the cell) - ignore
            pass
        if time.time() > end:
            return 0
    return 1


def _CheckSeenEvents(o, events):
    rc = 1
    for e in events:
        if e not in o.seen_events:
            print("ERROR: Expected event did not trigger", e)
            rc = 0
    return rc


def test():
    import sys

    if "noword" not in sys.argv[1:]:
        TestWord()
    if "noexcel" not in sys.argv[1:]:
        TestExcel()
    print("Word and Excel event tests passed.")


if __name__ == "__main__":
    test()
