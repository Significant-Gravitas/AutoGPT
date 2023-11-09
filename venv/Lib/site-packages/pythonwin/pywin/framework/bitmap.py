import os

import win32api
import win32con
import win32ui
from pywin.mfc import docview, window

from . import app

bStretch = 1


class BitmapDocument(docview.Document):
    "A bitmap document.  Holds the bitmap data itself."

    def __init__(self, template):
        docview.Document.__init__(self, template)
        self.bitmap = None

    def OnNewDocument(self):
        # I can not create new bitmaps.
        win32ui.MessageBox("Bitmaps can not be created.")

    def OnOpenDocument(self, filename):
        self.bitmap = win32ui.CreateBitmap()
        # init data members
        f = open(filename, "rb")
        try:
            try:
                self.bitmap.LoadBitmapFile(f)
            except IOError:
                win32ui.MessageBox("Could not load the bitmap from %s" % filename)
                return 0
        finally:
            f.close()
        self.size = self.bitmap.GetSize()
        return 1

    def DeleteContents(self):
        self.bitmap = None


class BitmapView(docview.ScrollView):
    "A view of a bitmap.  Obtains data from document."

    def __init__(self, doc):
        docview.ScrollView.__init__(self, doc)
        self.width = self.height = 0
        # set up message handlers
        self.HookMessage(self.OnSize, win32con.WM_SIZE)

    def OnInitialUpdate(self):
        doc = self.GetDocument()
        if doc.bitmap:
            bitmapSize = doc.bitmap.GetSize()
            self.SetScrollSizes(win32con.MM_TEXT, bitmapSize)

    def OnSize(self, params):
        lParam = params[3]
        self.width = win32api.LOWORD(lParam)
        self.height = win32api.HIWORD(lParam)

    def OnDraw(self, dc):
        # set sizes used for "non stretch" mode.
        doc = self.GetDocument()
        if doc.bitmap is None:
            return
        bitmapSize = doc.bitmap.GetSize()
        if bStretch:
            # stretch BMP.
            viewRect = (0, 0, self.width, self.height)
            bitmapRect = (0, 0, bitmapSize[0], bitmapSize[1])
            doc.bitmap.Paint(dc, viewRect, bitmapRect)
        else:
            # non stretch.
            doc.bitmap.Paint(dc)


class BitmapFrame(window.MDIChildWnd):
    def OnCreateClient(self, createparams, context):
        borderX = win32api.GetSystemMetrics(win32con.SM_CXFRAME)
        borderY = win32api.GetSystemMetrics(win32con.SM_CYFRAME)
        titleY = win32api.GetSystemMetrics(win32con.SM_CYCAPTION)  # includes border
        # try and maintain default window pos, else adjust if cant fit
        # get the main client window dimensions.
        mdiClient = win32ui.GetMainFrame().GetWindow(win32con.GW_CHILD)
        clientWindowRect = mdiClient.ScreenToClient(mdiClient.GetWindowRect())
        clientWindowSize = (
            clientWindowRect[2] - clientWindowRect[0],
            clientWindowRect[3] - clientWindowRect[1],
        )
        left, top, right, bottom = mdiClient.ScreenToClient(self.GetWindowRect())
        # 		width, height=context.doc.size[0], context.doc.size[1]
        # 		width = width+borderX*2
        # 		height= height+titleY+borderY*2-1
        # 		if (left+width)>clientWindowSize[0]:
        # 			left = clientWindowSize[0] - width
        # 		if left<0:
        # 			left = 0
        # 			width = clientWindowSize[0]
        # 		if (top+height)>clientWindowSize[1]:
        # 			top = clientWindowSize[1] - height
        # 		if top<0:
        # 			top = 0
        # 			height = clientWindowSize[1]
        # 		self.frame.MoveWindow((left, top, left+width, top+height),0)
        window.MDIChildWnd.OnCreateClient(self, createparams, context)
        return 1


class BitmapTemplate(docview.DocTemplate):
    def __init__(self):
        docview.DocTemplate.__init__(
            self, win32ui.IDR_PYTHONTYPE, BitmapDocument, BitmapFrame, BitmapView
        )

    def MatchDocType(self, fileName, fileType):
        doc = self.FindOpenDocument(fileName)
        if doc:
            return doc
        ext = os.path.splitext(fileName)[1].lower()
        if ext == ".bmp":  # removed due to PIL! or ext=='.ppm':
            return win32ui.CDocTemplate_Confidence_yesAttemptNative
        return win32ui.CDocTemplate_Confidence_maybeAttemptForeign


# 		return win32ui.CDocTemplate_Confidence_noAttempt

# For debugging purposes, when this module may be reloaded many times.
try:
    win32ui.GetApp().RemoveDocTemplate(bitmapTemplate)
except NameError:
    pass

bitmapTemplate = BitmapTemplate()
bitmapTemplate.SetDocStrings(
    "\nBitmap\nBitmap\nBitmap (*.bmp)\n.bmp\nPythonBitmapFileType\nPython Bitmap File"
)
win32ui.GetApp().AddDocTemplate(bitmapTemplate)

# This works, but just didnt make it through the code reorg.
# class PPMBitmap(Bitmap):
# 	def LoadBitmapFile(self, file ):
# 		magic=file.readline()
# 		if magic <> "P6\n":
# 			raise TypeError, "The file is not a PPM format file"
# 		rowcollist=string.split(file.readline())
# 		cols=string.atoi(rowcollist[0])
# 		rows=string.atoi(rowcollist[1])
# 		file.readline()	# whats this one?
# 		self.bitmap.LoadPPMFile(file,(cols,rows))


def t():
    bitmapTemplate.OpenDocumentFile("d:\\winnt\\arcade.bmp")
    # OpenBMPFile( 'd:\\winnt\\arcade.bmp')


def demo():
    import glob

    winDir = win32api.GetWindowsDirectory()
    for fileName in glob.glob1(winDir, "*.bmp")[:2]:
        bitmapTemplate.OpenDocumentFile(os.path.join(winDir, fileName))
