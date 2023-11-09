# Demo of Generic document windows, DC, and Font usage
# by Dave Brennan (brennan@hal.com)

# usage examples:

# >>> from fontdemo import *
# >>> d = FontDemo('Hello, Python')
# >>> f1 = { 'name':'Arial', 'height':36, 'weight':win32con.FW_BOLD}
# >>> d.SetFont(f1)
# >>> f2 = {'name':'Courier New', 'height':24, 'italic':1}
# >>> d.SetFont (f2)

import win32api
import win32con
import win32ui
from pywin.mfc import docview

# font is a dictionary in which the following elements matter:
# (the best matching font to supplied parameters is returned)
#   name		string name of the font as known by Windows
#   size		point size of font in logical units
#   weight		weight of font (win32con.FW_NORMAL, win32con.FW_BOLD)
#   italic		boolean; true if set to anything but None
#   underline	boolean; true if set to anything but None


class FontView(docview.ScrollView):
    def __init__(
        self, doc, text="Python Rules!", font_spec={"name": "Arial", "height": 42}
    ):
        docview.ScrollView.__init__(self, doc)
        self.font = win32ui.CreateFont(font_spec)
        self.text = text
        self.width = self.height = 0
        # set up message handlers
        self.HookMessage(self.OnSize, win32con.WM_SIZE)

    def OnAttachedObjectDeath(self):
        docview.ScrollView.OnAttachedObjectDeath(self)
        del self.font

    def SetFont(self, new_font):
        # Change font on the fly
        self.font = win32ui.CreateFont(new_font)
        # redraw the entire client window
        selfInvalidateRect(None)

    def OnSize(self, params):
        lParam = params[3]
        self.width = win32api.LOWORD(lParam)
        self.height = win32api.HIWORD(lParam)

    def OnPrepareDC(self, dc, printinfo):
        # Set up the DC for forthcoming OnDraw call
        self.SetScrollSizes(win32con.MM_TEXT, (100, 100))
        dc.SetTextColor(win32api.RGB(0, 0, 255))
        dc.SetBkColor(win32api.GetSysColor(win32con.COLOR_WINDOW))
        dc.SelectObject(self.font)
        dc.SetTextAlign(win32con.TA_CENTER | win32con.TA_BASELINE)

    def OnDraw(self, dc):
        if self.width == 0 and self.height == 0:
            left, top, right, bottom = self.GetClientRect()
            self.width = right - left
            self.height = bottom - top
        x, y = self.width // 2, self.height // 2
        dc.TextOut(x, y, self.text)


def FontDemo():
    # create doc/view
    template = docview.DocTemplate(win32ui.IDR_PYTHONTYPE, None, None, FontView)
    doc = template.OpenDocumentFile(None)
    doc.SetTitle("Font Demo")
    # 	print "template is ", template, "obj is", template._obj_
    template.close()


# 	print "closed"
# 	del template

if __name__ == "__main__":
    import demoutils

    if demoutils.NeedGoodGUI():
        FontDemo()
