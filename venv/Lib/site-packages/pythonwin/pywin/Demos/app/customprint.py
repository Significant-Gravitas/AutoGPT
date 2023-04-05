# A demo of an Application object that has some custom print functionality.

# If you desire, you can also run this from inside Pythonwin, in which
# case it will do the demo inside the Pythonwin environment.

# This sample was contributed by Roger Burnham.

import win32api
import win32con
import win32ui
from pywin.framework import app
from pywin.mfc import afxres, dialog, docview

PRINTDLGORD = 1538
IDC_PRINT_MAG_EDIT = 1010


class PrintDemoTemplate(docview.DocTemplate):
    def _SetupSharedMenu_(self):
        pass


class PrintDemoView(docview.ScrollView):
    def OnInitialUpdate(self):
        ret = self._obj_.OnInitialUpdate()
        self.colors = {
            "Black": (0x00 << 0) + (0x00 << 8) + (0x00 << 16),
            "Red": (0xFF << 0) + (0x00 << 8) + (0x00 << 16),
            "Green": (0x00 << 0) + (0xFF << 8) + (0x00 << 16),
            "Blue": (0x00 << 0) + (0x00 << 8) + (0xFF << 16),
            "Cyan": (0x00 << 0) + (0xFF << 8) + (0xFF << 16),
            "Magenta": (0xFF << 0) + (0x00 << 8) + (0xFF << 16),
            "Yellow": (0xFF << 0) + (0xFF << 8) + (0x00 << 16),
        }
        self.pens = {}
        for name, color in self.colors.items():
            self.pens[name] = win32ui.CreatePen(win32con.PS_SOLID, 5, color)
        self.pen = None
        self.size = (128, 128)
        self.SetScaleToFitSize(self.size)
        self.HookCommand(self.OnFilePrint, afxres.ID_FILE_PRINT)
        self.HookCommand(self.OnFilePrintPreview, win32ui.ID_FILE_PRINT_PREVIEW)
        return ret

    def OnDraw(self, dc):
        oldPen = None
        x, y = self.size
        delta = 2
        colors = list(self.colors.keys())
        colors.sort()
        colors = colors * 2
        for color in colors:
            if oldPen is None:
                oldPen = dc.SelectObject(self.pens[color])
            else:
                dc.SelectObject(self.pens[color])
            dc.MoveTo((delta, delta))
            dc.LineTo((x - delta, delta))
            dc.LineTo((x - delta, y - delta))
            dc.LineTo((delta, y - delta))
            dc.LineTo((delta, delta))
            delta = delta + 4
            if x - delta <= 0 or y - delta <= 0:
                break
        dc.SelectObject(oldPen)

    def OnPrepareDC(self, dc, pInfo):
        if dc.IsPrinting():
            mag = self.prtDlg["mag"]
            dc.SetMapMode(win32con.MM_ANISOTROPIC)
            dc.SetWindowOrg((0, 0))
            dc.SetWindowExt((1, 1))
            dc.SetViewportOrg((0, 0))
            dc.SetViewportExt((mag, mag))

    def OnPreparePrinting(self, pInfo):
        flags = (
            win32ui.PD_USEDEVMODECOPIES
            | win32ui.PD_PAGENUMS
            | win32ui.PD_NOPAGENUMS
            | win32ui.PD_NOSELECTION
        )
        self.prtDlg = ImagePrintDialog(pInfo, PRINTDLGORD, flags)
        pInfo.SetPrintDialog(self.prtDlg)
        pInfo.SetMinPage(1)
        pInfo.SetMaxPage(1)
        pInfo.SetFromPage(1)
        pInfo.SetToPage(1)
        ret = self.DoPreparePrinting(pInfo)
        return ret

    def OnBeginPrinting(self, dc, pInfo):
        return self._obj_.OnBeginPrinting(dc, pInfo)

    def OnEndPrinting(self, dc, pInfo):
        del self.prtDlg
        return self._obj_.OnEndPrinting(dc, pInfo)

    def OnFilePrintPreview(self, *arg):
        self._obj_.OnFilePrintPreview()

    def OnFilePrint(self, *arg):
        self._obj_.OnFilePrint()

    def OnPrint(self, dc, pInfo):
        doc = self.GetDocument()
        metrics = dc.GetTextMetrics()
        cxChar = metrics["tmAveCharWidth"]
        cyChar = metrics["tmHeight"]
        left, top, right, bottom = pInfo.GetDraw()
        dc.TextOut(0, 2 * cyChar, doc.GetTitle())
        top = top + (7 * cyChar) / 2
        dc.MoveTo(left, top)
        dc.LineTo(right, top)
        top = top + cyChar
        # this seems to have not effect...
        # get what I want with the dc.SetWindowOrg calls
        pInfo.SetDraw((left, top, right, bottom))
        dc.SetWindowOrg((0, -top))

        self.OnDraw(dc)
        dc.SetTextAlign(win32con.TA_LEFT | win32con.TA_BOTTOM)

        rect = self.GetWindowRect()
        rect = self.ScreenToClient(rect)
        height = rect[3] - rect[1]
        dc.SetWindowOrg((0, -(top + height + cyChar)))
        dc.MoveTo(left, 0)
        dc.LineTo(right, 0)

        x = 0
        y = (3 * cyChar) / 2

        dc.TextOut(x, y, doc.GetTitle())
        y = y + cyChar


class PrintDemoApp(app.CApp):
    def __init__(self):
        app.CApp.__init__(self)

    def InitInstance(self):
        template = PrintDemoTemplate(None, None, None, PrintDemoView)
        self.AddDocTemplate(template)
        self._obj_.InitMDIInstance()
        self.LoadMainFrame()
        doc = template.OpenDocumentFile(None)
        doc.SetTitle("Custom Print Document")


class ImagePrintDialog(dialog.PrintDialog):
    sectionPos = "Image Print Demo"

    def __init__(self, pInfo, dlgID, flags=win32ui.PD_USEDEVMODECOPIES):
        dialog.PrintDialog.__init__(self, pInfo, dlgID, flags=flags)
        mag = win32ui.GetProfileVal(self.sectionPos, "Document Magnification", 0)
        if mag <= 0:
            mag = 2
            win32ui.WriteProfileVal(self.sectionPos, "Document Magnification", mag)

        self["mag"] = mag

    def OnInitDialog(self):
        self.magCtl = self.GetDlgItem(IDC_PRINT_MAG_EDIT)
        self.magCtl.SetWindowText(repr(self["mag"]))
        return dialog.PrintDialog.OnInitDialog(self)

    def OnOK(self):
        dialog.PrintDialog.OnOK(self)
        strMag = self.magCtl.GetWindowText()
        try:
            self["mag"] = int(strMag)
        except:
            pass
        win32ui.WriteProfileVal(self.sectionPos, "Document Magnification", self["mag"])


if __name__ == "__main__":
    # Running under Pythonwin
    def test():
        template = PrintDemoTemplate(None, None, None, PrintDemoView)
        template.OpenDocumentFile(None)

    test()
else:
    app = PrintDemoApp()
