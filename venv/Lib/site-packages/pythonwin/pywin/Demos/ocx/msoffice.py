# This demo uses some of the Microsoft Office components.
#
# It was taken from an MSDN article showing how to embed excel.
# It is not comlpete yet, but it _does_ show an Excel spreadsheet in a frame!
#

import regutil
import win32con
import win32ui
import win32uiole
from pywin.mfc import activex, docview, object, window
from win32com.client import gencache

# WordModule = gencache.EnsureModule('{00020905-0000-0000-C000-000000000046}', 1033, 8, 0)
# if WordModule is None:
# 	raise ImportError, "Microsoft Word version 8 does not appear to be installed."


class OleClientItem(object.CmdTarget):
    def __init__(self, doc):
        object.CmdTarget.__init__(self, win32uiole.CreateOleClientItem(doc))

    def OnGetItemPosition(self):
        # For now return a hard-coded rect.
        return (10, 10, 210, 210)

    def OnActivate(self):
        # Allow only one inplace activate item per frame
        view = self.GetActiveView()
        item = self.GetDocument().GetInPlaceActiveItem(view)
        if item is not None and item._obj_ != self._obj_:
            item.Close()
        self._obj_.OnActivate()

    def OnChange(self, oleNotification, dwParam):
        self._obj_.OnChange(oleNotification, dwParam)
        self.GetDocument().UpdateAllViews(None)

    def OnChangeItemPosition(self, rect):
        # During in-place activation CEmbed_ExcelCntrItem::OnChangeItemPosition
        #  is called by the server to change the position of the in-place
        #  window.  Usually, this is a result of the data in the server
        #  document changing such that the extent has changed or as a result
        #  of in-place resizing.
        #
        # The default here is to call the base class, which will call
        #  COleClientItem::SetItemRects to move the item
        #  to the new position.
        if not self._obj_.OnChangeItemPosition(self, rect):
            return 0

        # TODO: update any cache you may have of the item's rectangle/extent
        return 1


class OleDocument(object.CmdTarget):
    def __init__(self, template):
        object.CmdTarget.__init__(self, win32uiole.CreateOleDocument(template))
        self.EnableCompoundFile()


class ExcelView(docview.ScrollView):
    def OnInitialUpdate(self):
        self.HookMessage(self.OnSetFocus, win32con.WM_SETFOCUS)
        self.HookMessage(self.OnSize, win32con.WM_SIZE)

        self.SetScrollSizes(win32con.MM_TEXT, (100, 100))
        rc = self._obj_.OnInitialUpdate()
        self.EmbedExcel()
        return rc

    def EmbedExcel(self):
        doc = self.GetDocument()
        self.clientItem = OleClientItem(doc)
        self.clientItem.CreateNewItem("Excel.Sheet")
        self.clientItem.DoVerb(-1, self)
        doc.UpdateAllViews(None)

    def OnDraw(self, dc):
        doc = self.GetDocument()
        pos = doc.GetStartPosition()
        clientItem, pos = doc.GetNextItem(pos)
        clientItem.Draw(dc, (10, 10, 210, 210))

    # Special handling of OnSetFocus and OnSize are required for a container
    #  when an object is being edited in-place.
    def OnSetFocus(self, msg):
        item = self.GetDocument().GetInPlaceActiveItem(self)
        if (
            item is not None
            and item.GetItemState() == win32uiole.COleClientItem_activeUIState
        ):
            wnd = item.GetInPlaceWindow()
            if wnd is not None:
                wnd.SetFocus()
            return 0  # Dont get the base version called.
        return 1  # Call the base version.

    def OnSize(self, params):
        item = self.GetDocument().GetInPlaceActiveItem(self)
        if item is not None:
            item.SetItemRects()
        return 1  # do call the base!


class OleTemplate(docview.DocTemplate):
    def __init__(
        self, resourceId=None, MakeDocument=None, MakeFrame=None, MakeView=None
    ):
        if MakeDocument is None:
            MakeDocument = OleDocument
        if MakeView is None:
            MakeView = ExcelView
        docview.DocTemplate.__init__(
            self, resourceId, MakeDocument, MakeFrame, MakeView
        )


class WordFrame(window.MDIChildWnd):
    def __init__(self, doc=None):
        self._obj_ = win32ui.CreateMDIChild()
        self._obj_.AttachObject(self)
        # Dont call base class doc/view version...

    def Create(self, title, rect=None, parent=None):
        style = win32con.WS_CHILD | win32con.WS_VISIBLE | win32con.WS_OVERLAPPEDWINDOW
        self._obj_.CreateWindow(None, title, style, rect, parent)

        rect = self.GetClientRect()
        rect = (0, 0, rect[2] - rect[0], rect[3] - rect[1])
        self.ocx = MyWordControl()
        self.ocx.CreateControl(
            "Microsoft Word", win32con.WS_VISIBLE | win32con.WS_CHILD, rect, self, 20000
        )


def Demo():
    import sys

    import win32api

    docName = None
    if len(sys.argv) > 1:
        docName = win32api.GetFullPathName(sys.argv[1])
    OleTemplate().OpenDocumentFile(None)


# 	f = WordFrame(docName)
# 	f.Create("Microsoft Office")

if __name__ == "__main__":
    Demo()
