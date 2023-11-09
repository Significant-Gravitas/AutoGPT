# DockingBar.py

# Ported directly (comments and all) from the samples at www.codeguru.com

# WARNING: Use at your own risk, as this interface is highly likely to change.
# Currently we support only one child per DockingBar.  Later we need to add
# support for multiple children.

import struct

import win32api
import win32con
import win32ui
from pywin.mfc import afxres, window

clrBtnHilight = win32api.GetSysColor(win32con.COLOR_BTNHILIGHT)
clrBtnShadow = win32api.GetSysColor(win32con.COLOR_BTNSHADOW)


def CenterPoint(rect):
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    return rect[0] + width // 2, rect[1] + height // 2


def OffsetRect(rect, point):
    (x, y) = point
    return rect[0] + x, rect[1] + y, rect[2] + x, rect[3] + y


def DeflateRect(rect, point):
    (x, y) = point
    return rect[0] + x, rect[1] + y, rect[2] - x, rect[3] - y


def PtInRect(rect, pt):
    return rect[0] <= pt[0] < rect[2] and rect[1] <= pt[1] < rect[3]


class DockingBar(window.Wnd):
    def __init__(self, obj=None):
        if obj is None:
            obj = win32ui.CreateControlBar()
        window.Wnd.__init__(self, obj)
        self.dialog = None
        self.nDockBarID = 0
        self.sizeMin = 32, 32
        self.sizeHorz = 200, 200
        self.sizeVert = 200, 200
        self.sizeFloat = 200, 200
        self.bTracking = 0
        self.bInRecalcNC = 0
        self.cxEdge = 6
        self.cxBorder = 3
        self.cxGripper = 20
        self.brushBkgd = win32ui.CreateBrush()
        self.brushBkgd.CreateSolidBrush(win32api.GetSysColor(win32con.COLOR_BTNFACE))

        # Support for diagonal resizing
        self.cyBorder = 3
        self.cCaptionSize = win32api.GetSystemMetrics(win32con.SM_CYSMCAPTION)
        self.cMinWidth = win32api.GetSystemMetrics(win32con.SM_CXMIN)
        self.cMinHeight = win32api.GetSystemMetrics(win32con.SM_CYMIN)
        self.rectUndock = (0, 0, 0, 0)

    def OnUpdateCmdUI(self, target, bDisableIfNoHndler):
        return self.UpdateDialogControls(target, bDisableIfNoHndler)

    def CreateWindow(
        self,
        parent,
        childCreator,
        title,
        id,
        style=win32con.WS_CHILD | win32con.WS_VISIBLE | afxres.CBRS_LEFT,
        childCreatorArgs=(),
    ):
        assert not (
            (style & afxres.CBRS_SIZE_FIXED) and (style & afxres.CBRS_SIZE_DYNAMIC)
        ), "Invalid style"
        self.rectClose = self.rectBorder = self.rectGripper = self.rectTracker = (
            0,
            0,
            0,
            0,
        )

        # save the style
        self._obj_.dwStyle = style & afxres.CBRS_ALL

        cursor = win32api.LoadCursor(0, win32con.IDC_ARROW)
        wndClass = win32ui.RegisterWndClass(
            win32con.CS_DBLCLKS, cursor, self.brushBkgd.GetSafeHandle(), 0
        )

        self._obj_.CreateWindow(wndClass, title, style, (0, 0, 0, 0), parent, id)

        # Create the child dialog
        self.dialog = childCreator(*(self,) + childCreatorArgs)

        # use the dialog dimensions as default base dimensions
        assert self.dialog.IsWindow(), (
            "The childCreator function %s did not create a window!" % childCreator
        )
        rect = self.dialog.GetWindowRect()
        self.sizeHorz = self.sizeVert = self.sizeFloat = (
            rect[2] - rect[0],
            rect[3] - rect[1],
        )

        self.sizeHorz = self.sizeHorz[0], self.sizeHorz[1] + self.cxEdge + self.cxBorder
        self.sizeVert = self.sizeVert[0] + self.cxEdge + self.cxBorder, self.sizeVert[1]
        self.HookMessages()

    def CalcFixedLayout(self, bStretch, bHorz):
        rectTop = self.dockSite.GetControlBar(
            afxres.AFX_IDW_DOCKBAR_TOP
        ).GetWindowRect()
        rectLeft = self.dockSite.GetControlBar(
            afxres.AFX_IDW_DOCKBAR_LEFT
        ).GetWindowRect()
        if bStretch:
            nHorzDockBarWidth = 32767
            nVertDockBarHeight = 32767
        else:
            nHorzDockBarWidth = rectTop[2] - rectTop[0] + 4
            nVertDockBarHeight = rectLeft[3] - rectLeft[1] + 4

        if self.IsFloating():
            return self.sizeFloat
        if bHorz:
            return nHorzDockBarWidth, self.sizeHorz[1]
        return self.sizeVert[0], nVertDockBarHeight

    def CalcDynamicLayout(self, length, mode):
        # Support for diagonal sizing.
        if self.IsFloating():
            self.GetParent().GetParent().ModifyStyle(win32ui.MFS_4THICKFRAME, 0)
        if mode & (win32ui.LM_HORZDOCK | win32ui.LM_VERTDOCK):
            flags = (
                win32con.SWP_NOSIZE
                | win32con.SWP_NOMOVE
                | win32con.SWP_NOZORDER
                | win32con.SWP_NOACTIVATE
                | win32con.SWP_FRAMECHANGED
            )
            self.SetWindowPos(
                0,
                (
                    0,
                    0,
                    0,
                    0,
                ),
                flags,
            )
            self.dockSite.RecalcLayout()
            return self._obj_.CalcDynamicLayout(length, mode)

        if mode & win32ui.LM_MRUWIDTH:
            return self.sizeFloat
        if mode & win32ui.LM_COMMIT:
            self.sizeFloat = length, self.sizeFloat[1]
            return self.sizeFloat
        # More diagonal sizing.
        if self.IsFloating():
            dc = self.dockContext
            pt = win32api.GetCursorPos()
            windowRect = self.GetParent().GetParent().GetWindowRect()

            hittest = dc.nHitTest
            if hittest == win32con.HTTOPLEFT:
                cx = max(windowRect[2] - pt[0], self.cMinWidth) - self.cxBorder
                cy = max(windowRect[3] - self.cCaptionSize - pt[1], self.cMinHeight) - 1
                self.sizeFloat = cx, cy

                top = (
                    min(pt[1], windowRect[3] - self.cCaptionSize - self.cMinHeight)
                    - self.cyBorder
                )
                left = min(pt[0], windowRect[2] - self.cMinWidth) - 1
                dc.rectFrameDragHorz = (
                    left,
                    top,
                    dc.rectFrameDragHorz[2],
                    dc.rectFrameDragHorz[3],
                )
                return self.sizeFloat
            if hittest == win32con.HTTOPRIGHT:
                cx = max(pt[0] - windowRect[0], self.cMinWidth)
                cy = max(windowRect[3] - self.cCaptionSize - pt[1], self.cMinHeight) - 1
                self.sizeFloat = cx, cy

                top = (
                    min(pt[1], windowRect[3] - self.cCaptionSize - self.cMinHeight)
                    - self.cyBorder
                )
                dc.rectFrameDragHorz = (
                    dc.rectFrameDragHorz[0],
                    top,
                    dc.rectFrameDragHorz[2],
                    dc.rectFrameDragHorz[3],
                )
                return self.sizeFloat

            if hittest == win32con.HTBOTTOMLEFT:
                cx = max(windowRect[2] - pt[0], self.cMinWidth) - self.cxBorder
                cy = max(pt[1] - windowRect[1] - self.cCaptionSize, self.cMinHeight)
                self.sizeFloat = cx, cy

                left = min(pt[0], windowRect[2] - self.cMinWidth) - 1
                dc.rectFrameDragHorz = (
                    left,
                    dc.rectFrameDragHorz[1],
                    dc.rectFrameDragHorz[2],
                    dc.rectFrameDragHorz[3],
                )
                return self.sizeFloat

            if hittest == win32con.HTBOTTOMRIGHT:
                cx = max(pt[0] - windowRect[0], self.cMinWidth)
                cy = max(pt[1] - windowRect[1] - self.cCaptionSize, self.cMinHeight)
                self.sizeFloat = cx, cy
                return self.sizeFloat

        if mode & win32ui.LM_LENGTHY:
            self.sizeFloat = self.sizeFloat[0], max(self.sizeMin[1], length)
            return self.sizeFloat
        else:
            return max(self.sizeMin[0], length), self.sizeFloat[1]

    def OnWindowPosChanged(self, msg):
        if self.GetSafeHwnd() == 0 or self.dialog is None:
            return 0
        lparam = msg[3]
        """ LPARAM used with WM_WINDOWPOSCHANGED:
			typedef struct {
				HWND hwnd;
				HWND hwndInsertAfter;
				int x;
				int y;
				int cx;
				int cy;
				UINT flags;} WINDOWPOS;
		"""
        format = "PPiiiii"
        bytes = win32ui.GetBytes(lparam, struct.calcsize(format))
        hwnd, hwndAfter, x, y, cx, cy, flags = struct.unpack(format, bytes)

        if self.bInRecalcNC:
            rc = self.GetClientRect()
            self.dialog.MoveWindow(rc)
            return 0
        # Find on which side are we docked
        nDockBarID = self.GetParent().GetDlgCtrlID()
        # Return if dropped at same location
        # no docking side change and no size change
        if (
            (nDockBarID == self.nDockBarID)
            and (flags & win32con.SWP_NOSIZE)
            and (
                (self._obj_.dwStyle & afxres.CBRS_BORDER_ANY) != afxres.CBRS_BORDER_ANY
            )
        ):
            return
        self.nDockBarID = nDockBarID

        # Force recalc the non-client area
        self.bInRecalcNC = 1
        try:
            swpflags = (
                win32con.SWP_NOSIZE
                | win32con.SWP_NOMOVE
                | win32con.SWP_NOZORDER
                | win32con.SWP_FRAMECHANGED
            )
            self.SetWindowPos(0, (0, 0, 0, 0), swpflags)
        finally:
            self.bInRecalcNC = 0
        return 0

    # This is a virtual and not a message hook.
    def OnSetCursor(self, window, nHitTest, wMouseMsg):
        if nHitTest != win32con.HTSIZE or self.bTracking:
            return self._obj_.OnSetCursor(window, nHitTest, wMouseMsg)

        if self.IsHorz():
            win32api.SetCursor(win32api.LoadCursor(0, win32con.IDC_SIZENS))
        else:
            win32api.SetCursor(win32api.LoadCursor(0, win32con.IDC_SIZEWE))
        return 1

    # Mouse Handling
    def OnLButtonUp(self, msg):
        if not self.bTracking:
            return 1  # pass it on.
        self.StopTracking(1)
        return 0  # Dont pass on

    def OnLButtonDown(self, msg):
        # UINT nFlags, CPoint point)
        # only start dragging if clicked in "void" space
        if self.dockBar is not None:
            # start the drag
            pt = msg[5]
            pt = self.ClientToScreen(pt)
            self.dockContext.StartDrag(pt)
            return 0
        return 1

    def OnNcLButtonDown(self, msg):
        if self.bTracking:
            return 0
        nHitTest = wparam = msg[2]
        pt = msg[5]

        if nHitTest == win32con.HTSYSMENU and not self.IsFloating():
            self.GetDockingFrame().ShowControlBar(self, 0, 0)
        elif nHitTest == win32con.HTMINBUTTON and not self.IsFloating():
            self.dockContext.ToggleDocking()
        elif (
            nHitTest == win32con.HTCAPTION
            and not self.IsFloating()
            and self.dockBar is not None
        ):
            self.dockContext.StartDrag(pt)
        elif nHitTest == win32con.HTSIZE and not self.IsFloating():
            self.StartTracking()
        else:
            return 1
        return 0

    def OnLButtonDblClk(self, msg):
        # only toggle docking if clicked in "void" space
        if self.dockBar is not None:
            # toggle docking
            self.dockContext.ToggleDocking()
            return 0
        return 1

    def OnNcLButtonDblClk(self, msg):
        nHitTest = wparam = msg[2]
        # UINT nHitTest, CPoint point)
        if self.dockBar is not None and nHitTest == win32con.HTCAPTION:
            # toggle docking
            self.dockContext.ToggleDocking()
            return 0
        return 1

    def OnMouseMove(self, msg):
        flags = wparam = msg[2]
        lparam = msg[3]
        if self.IsFloating() or not self.bTracking:
            return 1

        # Convert unsigned 16 bit to signed 32 bit.
        x = win32api.LOWORD(lparam)
        if x & 32768:
            x = x | -65536
        y = win32api.HIWORD(lparam)
        if y & 32768:
            y = y | -65536
        pt = x, y
        cpt = CenterPoint(self.rectTracker)
        pt = self.ClientToWnd(pt)
        if self.IsHorz():
            if cpt[1] != pt[1]:
                self.OnInvertTracker(self.rectTracker)
                self.rectTracker = OffsetRect(self.rectTracker, (0, pt[1] - cpt[1]))
                self.OnInvertTracker(self.rectTracker)
        else:
            if cpt[0] != pt[0]:
                self.OnInvertTracker(self.rectTracker)
                self.rectTracker = OffsetRect(self.rectTracker, (pt[0] - cpt[0], 0))
                self.OnInvertTracker(self.rectTracker)

        return 0  # Dont pass it on.

    # 	def OnBarStyleChange(self, old, new):

    def OnNcCalcSize(self, bCalcValid, size_info):
        (rc0, rc1, rc2, pos) = size_info
        self.rectBorder = self.GetWindowRect()
        self.rectBorder = OffsetRect(
            self.rectBorder, (-self.rectBorder[0], -self.rectBorder[1])
        )

        dwBorderStyle = self._obj_.dwStyle | afxres.CBRS_BORDER_ANY

        if self.nDockBarID == afxres.AFX_IDW_DOCKBAR_TOP:
            dwBorderStyle = dwBorderStyle & ~afxres.CBRS_BORDER_BOTTOM
            rc0.left = rc0.left + self.cxGripper
            rc0.bottom = rc0.bottom - self.cxEdge
            rc0.top = rc0.top + self.cxBorder
            rc0.right = rc0.right - self.cxBorder
            self.rectBorder = (
                self.rectBorder[0],
                self.rectBorder[3] - self.cxEdge,
                self.rectBorder[2],
                self.rectBorder[3],
            )
        elif self.nDockBarID == afxres.AFX_IDW_DOCKBAR_BOTTOM:
            dwBorderStyle = dwBorderStyle & ~afxres.CBRS_BORDER_TOP
            rc0.left = rc0.left + self.cxGripper
            rc0.top = rc0.top + self.cxEdge
            rc0.bottom = rc0.bottom - self.cxBorder
            rc0.right = rc0.right - self.cxBorder
            self.rectBorder = (
                self.rectBorder[0],
                self.rectBorder[1],
                self.rectBorder[2],
                self.rectBorder[1] + self.cxEdge,
            )
        elif self.nDockBarID == afxres.AFX_IDW_DOCKBAR_LEFT:
            dwBorderStyle = dwBorderStyle & ~afxres.CBRS_BORDER_RIGHT
            rc0.right = rc0.right - self.cxEdge
            rc0.left = rc0.left + self.cxBorder
            rc0.bottom = rc0.bottom - self.cxBorder
            rc0.top = rc0.top + self.cxGripper
            self.rectBorder = (
                self.rectBorder[2] - self.cxEdge,
                self.rectBorder[1],
                self.rectBorder[2],
                self.rectBorder[3],
            )
        elif self.nDockBarID == afxres.AFX_IDW_DOCKBAR_RIGHT:
            dwBorderStyle = dwBorderStyle & ~afxres.CBRS_BORDER_LEFT
            rc0.left = rc0.left + self.cxEdge
            rc0.right = rc0.right - self.cxBorder
            rc0.bottom = rc0.bottom - self.cxBorder
            rc0.top = rc0.top + self.cxGripper
            self.rectBorder = (
                self.rectBorder[0],
                self.rectBorder[1],
                self.rectBorder[0] + self.cxEdge,
                self.rectBorder[3],
            )
        else:
            self.rectBorder = 0, 0, 0, 0

        self.SetBarStyle(dwBorderStyle)
        return 0

    def OnNcPaint(self, msg):
        self.EraseNonClient()
        dc = self.GetWindowDC()
        ctl = win32api.GetSysColor(win32con.COLOR_BTNHIGHLIGHT)
        cbr = win32api.GetSysColor(win32con.COLOR_BTNSHADOW)
        dc.Draw3dRect(self.rectBorder, ctl, cbr)

        self.DrawGripper(dc)

        rect = self.GetClientRect()
        self.InvalidateRect(rect, 1)
        return 0

    def OnNcHitTest(self, pt):  # A virtual, not a hooked message.
        if self.IsFloating():
            return 1

        ptOrig = pt
        rect = self.GetWindowRect()
        pt = pt[0] - rect[0], pt[1] - rect[1]

        if PtInRect(self.rectClose, pt):
            return win32con.HTSYSMENU
        elif PtInRect(self.rectUndock, pt):
            return win32con.HTMINBUTTON
        elif PtInRect(self.rectGripper, pt):
            return win32con.HTCAPTION
        elif PtInRect(self.rectBorder, pt):
            return win32con.HTSIZE
        else:
            return self._obj_.OnNcHitTest(ptOrig)

    def StartTracking(self):
        self.SetCapture()

        # make sure no updates are pending
        self.RedrawWindow(None, None, win32con.RDW_ALLCHILDREN | win32con.RDW_UPDATENOW)
        self.dockSite.LockWindowUpdate()

        self.ptOld = CenterPoint(self.rectBorder)
        self.bTracking = 1

        self.rectTracker = self.rectBorder
        if not self.IsHorz():
            l, t, r, b = self.rectTracker
            b = b - 4
            self.rectTracker = l, t, r, b

        self.OnInvertTracker(self.rectTracker)

    def OnCaptureChanged(self, msg):
        hwnd = lparam = msg[3]
        if self.bTracking and hwnd != self.GetSafeHwnd():
            self.StopTracking(0)  # cancel tracking
        return 1

    def StopTracking(self, bAccept):
        self.OnInvertTracker(self.rectTracker)
        self.dockSite.UnlockWindowUpdate()
        self.bTracking = 0
        self.ReleaseCapture()
        if not bAccept:
            return

        rcc = self.dockSite.GetWindowRect()
        if self.IsHorz():
            newsize = self.sizeHorz[1]
            maxsize = newsize + (rcc[3] - rcc[1])
            minsize = self.sizeMin[1]
        else:
            newsize = self.sizeVert[0]
            maxsize = newsize + (rcc[2] - rcc[0])
            minsize = self.sizeMin[0]

        pt = CenterPoint(self.rectTracker)
        if self.nDockBarID == afxres.AFX_IDW_DOCKBAR_TOP:
            newsize = newsize + (pt[1] - self.ptOld[1])
        elif self.nDockBarID == afxres.AFX_IDW_DOCKBAR_BOTTOM:
            newsize = newsize + (-pt[1] + self.ptOld[1])
        elif self.nDockBarID == afxres.AFX_IDW_DOCKBAR_LEFT:
            newsize = newsize + (pt[0] - self.ptOld[0])
        elif self.nDockBarID == afxres.AFX_IDW_DOCKBAR_RIGHT:
            newsize = newsize + (-pt[0] + self.ptOld[0])
        newsize = max(minsize, min(maxsize, newsize))
        if self.IsHorz():
            self.sizeHorz = self.sizeHorz[0], newsize
        else:
            self.sizeVert = newsize, self.sizeVert[1]
        self.dockSite.RecalcLayout()
        return 0

    def OnInvertTracker(self, rect):
        assert rect[2] - rect[0] > 0 and rect[3] - rect[1] > 0, "rect is empty"
        assert self.bTracking
        rcc = self.GetWindowRect()
        rcf = self.dockSite.GetWindowRect()

        rect = OffsetRect(rect, (rcc[0] - rcf[0], rcc[1] - rcf[1]))
        rect = DeflateRect(rect, (1, 1))

        flags = win32con.DCX_WINDOW | win32con.DCX_CACHE | win32con.DCX_LOCKWINDOWUPDATE
        dc = self.dockSite.GetDCEx(None, flags)
        try:
            brush = win32ui.GetHalftoneBrush()
            oldBrush = dc.SelectObject(brush)

            dc.PatBlt(
                (rect[0], rect[1]),
                (rect[2] - rect[0], rect[3] - rect[1]),
                win32con.PATINVERT,
            )
            dc.SelectObject(oldBrush)
        finally:
            self.dockSite.ReleaseDC(dc)

    def IsHorz(self):
        return (
            self.nDockBarID == afxres.AFX_IDW_DOCKBAR_TOP
            or self.nDockBarID == afxres.AFX_IDW_DOCKBAR_BOTTOM
        )

    def ClientToWnd(self, pt):
        x, y = pt
        if self.nDockBarID == afxres.AFX_IDW_DOCKBAR_BOTTOM:
            y = y + self.cxEdge
        elif self.nDockBarID == afxres.AFX_IDW_DOCKBAR_RIGHT:
            x = x + self.cxEdge
        return x, y

    def DrawGripper(self, dc):
        # no gripper if floating
        if self._obj_.dwStyle & afxres.CBRS_FLOATING:
            return

        # -==HACK==-
        # in order to calculate the client area properly after docking,
        # the client area must be recalculated twice (I have no idea why)
        self.dockSite.RecalcLayout()
        # -==END HACK==-

        gripper = self.GetWindowRect()
        gripper = self.ScreenToClient(gripper)
        gripper = OffsetRect(gripper, (-gripper[0], -gripper[1]))
        gl, gt, gr, gb = gripper

        if self._obj_.dwStyle & afxres.CBRS_ORIENT_HORZ:
            # gripper at left
            self.rectGripper = gl, gt + 40, gl + 20, gb
            # draw close box
            self.rectClose = gl + 7, gt + 10, gl + 19, gt + 22
            dc.DrawFrameControl(
                self.rectClose, win32con.DFC_CAPTION, win32con.DFCS_CAPTIONCLOSE
            )
            # draw docking toggle box
            self.rectUndock = OffsetRect(self.rectClose, (0, 13))
            dc.DrawFrameControl(
                self.rectUndock, win32con.DFC_CAPTION, win32con.DFCS_CAPTIONMAX
            )

            gt = gt + 38
            gb = gb - 10
            gl = gl + 10
            gr = gl + 3
            gripper = gl, gt, gr, gb
            dc.Draw3dRect(gripper, clrBtnHilight, clrBtnShadow)
            dc.Draw3dRect(OffsetRect(gripper, (4, 0)), clrBtnHilight, clrBtnShadow)
        else:
            # gripper at top
            self.rectGripper = gl, gt, gr - 40, gt + 20
            # draw close box
            self.rectClose = gr - 21, gt + 7, gr - 10, gt + 18
            dc.DrawFrameControl(
                self.rectClose, win32con.DFC_CAPTION, win32con.DFCS_CAPTIONCLOSE
            )
            #  draw docking toggle box
            self.rectUndock = OffsetRect(self.rectClose, (-13, 0))
            dc.DrawFrameControl(
                self.rectUndock, win32con.DFC_CAPTION, win32con.DFCS_CAPTIONMAX
            )
            gr = gr - 38
            gl = gl + 10
            gt = gt + 10
            gb = gt + 3

            gripper = gl, gt, gr, gb
            dc.Draw3dRect(gripper, clrBtnHilight, clrBtnShadow)
            dc.Draw3dRect(OffsetRect(gripper, (0, 4)), clrBtnHilight, clrBtnShadow)

    def HookMessages(self):
        self.HookMessage(self.OnLButtonUp, win32con.WM_LBUTTONUP)
        self.HookMessage(self.OnLButtonDown, win32con.WM_LBUTTONDOWN)
        self.HookMessage(self.OnLButtonDblClk, win32con.WM_LBUTTONDBLCLK)
        self.HookMessage(self.OnNcLButtonDown, win32con.WM_NCLBUTTONDOWN)
        self.HookMessage(self.OnNcLButtonDblClk, win32con.WM_NCLBUTTONDBLCLK)
        self.HookMessage(self.OnMouseMove, win32con.WM_MOUSEMOVE)
        self.HookMessage(self.OnNcPaint, win32con.WM_NCPAINT)
        self.HookMessage(self.OnCaptureChanged, win32con.WM_CAPTURECHANGED)
        self.HookMessage(self.OnWindowPosChanged, win32con.WM_WINDOWPOSCHANGED)


# 		self.HookMessage(self.OnSize, win32con.WM_SIZE)


def EditCreator(parent):
    d = win32ui.CreateEdit()
    es = (
        win32con.WS_CHILD
        | win32con.WS_VISIBLE
        | win32con.WS_BORDER
        | win32con.ES_MULTILINE
        | win32con.ES_WANTRETURN
    )
    d.CreateWindow(es, (0, 0, 150, 150), parent, 1000)
    return d


def test():
    import pywin.mfc.dialog

    global bar
    bar = DockingBar()
    creator = EditCreator
    bar.CreateWindow(win32ui.GetMainFrame(), creator, "Coolbar Demo", 0xFFFFF)
    # 	win32ui.GetMainFrame().ShowControlBar(bar, 1, 0)
    bar.SetBarStyle(
        bar.GetBarStyle()
        | afxres.CBRS_TOOLTIPS
        | afxres.CBRS_FLYBY
        | afxres.CBRS_SIZE_DYNAMIC
    )
    bar.EnableDocking(afxres.CBRS_ALIGN_ANY)
    win32ui.GetMainFrame().DockControlBar(bar, afxres.AFX_IDW_DOCKBAR_BOTTOM)


if __name__ == "__main__":
    test()
