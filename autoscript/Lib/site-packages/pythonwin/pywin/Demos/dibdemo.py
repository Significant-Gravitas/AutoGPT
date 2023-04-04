# A demo which creates a view and a frame which displays a PPM format bitmap
#
# This hasnnt been run in a while, as I dont have many of that format around!

import win32api
import win32con
import win32ui


class DIBView:
    def __init__(self, doc, dib):
        self.dib = dib
        self.view = win32ui.CreateView(doc)
        self.width = self.height = 0
        # set up message handlers
        # 		self.view.OnPrepareDC = self.OnPrepareDC
        self.view.HookMessage(self.OnSize, win32con.WM_SIZE)

    def OnSize(self, params):
        lParam = params[3]
        self.width = win32api.LOWORD(lParam)
        self.height = win32api.HIWORD(lParam)

    def OnDraw(self, ob, dc):
        # set sizes used for "non strecth" mode.
        self.view.SetScrollSizes(win32con.MM_TEXT, self.dib.GetSize())
        dibSize = self.dib.GetSize()
        dibRect = (0, 0, dibSize[0], dibSize[1])
        # stretch BMP.
        # self.dib.Paint(dc, (0,0,self.width, self.height),dibRect)
        # non stretch.
        self.dib.Paint(dc)


class DIBDemo:
    def __init__(self, filename, *bPBM):
        # init data members
        f = open(filename, "rb")
        dib = win32ui.CreateDIBitmap()
        if len(bPBM) > 0:
            magic = f.readline()
            if magic != "P6\n":
                print("The file is not a PBM format file")
                raise ValueError("Failed - The file is not a PBM format file")
            # check magic?
            rowcollist = f.readline().split()
            cols = int(rowcollist[0])
            rows = int(rowcollist[1])
            f.readline()  # whats this one?
            dib.LoadPBMData(f, (cols, rows))
        else:
            dib.LoadWindowsFormatFile(f)
        f.close()
        # create doc/view
        self.doc = win32ui.CreateDoc()
        self.dibView = DIBView(self.doc, dib)
        self.frame = win32ui.CreateMDIFrame()
        self.frame.LoadFrame()  # this will force OnCreateClient
        self.doc.SetTitle("DIB Demo")
        self.frame.ShowWindow()

        # display the sucka
        self.frame.ActivateFrame()

    def OnCreateClient(self, createparams, context):
        self.dibView.view.CreateWindow(self.frame)
        return 1


if __name__ == "__main__":
    import demoutils

    demoutils.NotAScript()
