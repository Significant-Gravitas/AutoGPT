import commctrl
import fontdemo
import win32ui
from pywin.mfc import docview, window

# derive from CMDIChild.  This does much work for us.


class SplitterFrame(window.MDIChildWnd):
    def __init__(self):
        # call base CreateFrame
        self.images = None
        window.MDIChildWnd.__init__(self)

    def OnCreateClient(self, cp, context):
        splitter = win32ui.CreateSplitter()
        doc = context.doc
        frame_rect = self.GetWindowRect()
        size = ((frame_rect[2] - frame_rect[0]), (frame_rect[3] - frame_rect[1]) // 2)
        sub_size = (size[0] // 2, size[1])
        splitter.CreateStatic(self, 2, 1)
        self.v1 = win32ui.CreateEditView(doc)
        self.v2 = fontdemo.FontView(doc)
        # CListControl view
        self.v3 = win32ui.CreateListView(doc)
        sub_splitter = win32ui.CreateSplitter()
        # pass "splitter" so each view knows how to get to the others
        sub_splitter.CreateStatic(splitter, 1, 2)
        sub_splitter.CreateView(self.v1, 0, 0, (sub_size))
        sub_splitter.CreateView(self.v2, 0, 1, (0, 0))  # size ignored.
        splitter.SetRowInfo(0, size[1], 0)
        splitter.CreateView(self.v3, 1, 0, (0, 0))  # size ignored.
        # Setup items in the imagelist
        self.images = win32ui.CreateImageList(32, 32, 1, 5, 5)
        self.images.Add(win32ui.GetApp().LoadIcon(win32ui.IDR_MAINFRAME))
        self.images.Add(win32ui.GetApp().LoadIcon(win32ui.IDR_PYTHONCONTYPE))
        self.images.Add(win32ui.GetApp().LoadIcon(win32ui.IDR_TEXTTYPE))
        self.v3.SetImageList(self.images, commctrl.LVSIL_NORMAL)
        self.v3.InsertItem(0, "Icon 1", 0)
        self.v3.InsertItem(0, "Icon 2", 1)
        self.v3.InsertItem(0, "Icon 3", 2)
        # 		self.v3.Arrange(commctrl.LVA_DEFAULT) Hmmm - win95 aligns left always???
        return 1

    def OnDestroy(self, msg):
        window.MDIChildWnd.OnDestroy(self, msg)
        if self.images:
            self.images.DeleteImageList()
            self.images = None

    def InitialUpdateFrame(self, doc, makeVisible):
        self.v1.ReplaceSel("Hello from Edit Window 1")
        self.v1.SetModifiedFlag(0)


class SampleTemplate(docview.DocTemplate):
    def __init__(self):
        docview.DocTemplate.__init__(
            self, win32ui.IDR_PYTHONTYPE, None, SplitterFrame, None
        )

    def InitialUpdateFrame(self, frame, doc, makeVisible):
        # 		print "frame is ", frame, frame._obj_
        # 		print "doc is ", doc, doc._obj_
        self._obj_.InitialUpdateFrame(frame, doc, makeVisible)  # call default handler.
        frame.InitialUpdateFrame(doc, makeVisible)


def demo():
    template = SampleTemplate()
    doc = template.OpenDocumentFile(None)
    doc.SetTitle("Splitter Demo")


if __name__ == "__main__":
    import demoutils

    if demoutils.NeedGoodGUI():
        demo()
