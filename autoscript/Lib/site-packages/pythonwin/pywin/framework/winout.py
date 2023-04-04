# winout.py
#
# generic "output window"
#
# This Window will detect itself closing, and recreate next time output is
# written to it.

# This has the option of writing output at idle time (by hooking the
# idle message, and queueing output) or writing as each
# write is executed.
# Updating the window directly gives a jerky appearance as many writes
# take place between commands, and the windows scrolls, and updates etc
# Updating at idle-time may defer all output of a long process, giving the
# appearence nothing is happening.
# There is a compromise "line" mode, which will output whenever
# a complete line is available.

# behaviour depends on self.writeQueueing

# This module is thread safe - output can originate from any thread.  If any thread
# other than the main thread attempts to print, it is always queued until next idle time

import queue
import re

import win32api
import win32con
import win32ui
from pywin.framework import app, window
from pywin.mfc import docview

debug = lambda msg: None

##debug=win32ui.OutputDebugString
##import win32trace;win32trace.InitWrite() # for debugging - delete me!
##debug = win32trace.write


class flags:
    # queueing of output.
    WQ_NONE = 0
    WQ_LINE = 1
    WQ_IDLE = 2


# WindowOutputDocumentParent=docview.RichEditDoc
# WindowOutputDocumentParent=docview.Document
import pywin.scintilla.document
from pywin import default_scintilla_encoding
from pywin.scintilla import scintillacon

WindowOutputDocumentParent = pywin.scintilla.document.CScintillaDocument


class WindowOutputDocument(WindowOutputDocumentParent):
    def SaveModified(self):
        return 1  # say it is OK to destroy my document

    def OnSaveDocument(self, fileName):
        win32ui.SetStatusText("Saving file...", 1)
        try:
            self.SaveFile(fileName)
        except IOError as details:
            win32ui.MessageBox("Error - could not save file\r\n\r\n%s" % details)
            return 0
        win32ui.SetStatusText("Ready")
        return 1


class WindowOutputFrame(window.MDIChildWnd):
    def __init__(self, wnd=None):
        window.MDIChildWnd.__init__(self, wnd)
        self.HookMessage(self.OnSizeMove, win32con.WM_SIZE)
        self.HookMessage(self.OnSizeMove, win32con.WM_MOVE)

    def LoadFrame(self, idResource, style, wndParent, context):
        self.template = context.template
        return self._obj_.LoadFrame(idResource, style, wndParent, context)

    def PreCreateWindow(self, cc):
        cc = self._obj_.PreCreateWindow(cc)
        if (
            self.template.defSize
            and self.template.defSize[0] != self.template.defSize[1]
        ):
            rect = app.RectToCreateStructRect(self.template.defSize)
            cc = cc[0], cc[1], cc[2], cc[3], rect, cc[5], cc[6], cc[7], cc[8]
        return cc

    def OnSizeMove(self, msg):
        # so recreate maintains position.
        # Need to map coordinates from the
        # frame windows first child.
        mdiClient = self.GetParent()
        self.template.defSize = mdiClient.ScreenToClient(self.GetWindowRect())

    def OnDestroy(self, message):
        self.template.OnFrameDestroy(self)
        return 1


class WindowOutputViewImpl:
    def __init__(self):
        self.patErrorMessage = re.compile('\W*File "(.*)", line ([0-9]+)')
        self.template = self.GetDocument().GetDocTemplate()

    def HookHandlers(self):
        # Hook for the right-click menu.
        self.HookMessage(self.OnRClick, win32con.WM_RBUTTONDOWN)

    def OnDestroy(self, msg):
        self.template.OnViewDestroy(self)

    def OnInitialUpdate(self):
        self.RestoreKillBuffer()
        self.SetSel(-2)  # end of buffer

    def GetRightMenuItems(self):
        ret = []
        flags = win32con.MF_STRING | win32con.MF_ENABLED
        ret.append((flags, win32ui.ID_EDIT_COPY, "&Copy"))
        ret.append((flags, win32ui.ID_EDIT_SELECT_ALL, "&Select all"))
        return ret

    #
    # Windows command handlers, virtuals, etc.
    #
    def OnRClick(self, params):
        paramsList = self.GetRightMenuItems()
        menu = win32ui.CreatePopupMenu()
        for appendParams in paramsList:
            if type(appendParams) != type(()):
                appendParams = (appendParams,)
            menu.AppendMenu(*appendParams)
        menu.TrackPopupMenu(params[5])  # track at mouse position.
        return 0

    # as this is often used as an output window, exeptions will often
    # be printed.  Therefore, we support this functionality at this level.
    # Returns TRUE if the current line is an error message line, and will
    # jump to it.  FALSE if no error (and no action taken)
    def HandleSpecialLine(self):
        from . import scriptutils

        line = self.GetLine()
        if line[:11] == "com_error: ":
            # An OLE Exception - pull apart the exception
            # and try and locate a help file.
            try:
                import win32api
                import win32con

                det = eval(line[line.find(":") + 1 :].strip())
                win32ui.SetStatusText("Opening help file on OLE error...")
                from . import help

                help.OpenHelpFile(det[2][3], win32con.HELP_CONTEXT, det[2][4])
                return 1
            except win32api.error as details:
                win32ui.SetStatusText(
                    "The help file could not be opened - %s" % details.strerror
                )
                return 1
            except:
                win32ui.SetStatusText(
                    "Line is a COM error, but no WinHelp details can be parsed"
                )
        # Look for a Python traceback.
        matchResult = self.patErrorMessage.match(line)
        if matchResult is None:
            # No match - try the previous line
            lineNo = self.LineFromChar()
            if lineNo > 0:
                line = self.GetLine(lineNo - 1)
                matchResult = self.patErrorMessage.match(line)
        if matchResult is not None:
            # we have an error line.
            fileName = matchResult.group(1)
            if fileName[0] == "<":
                win32ui.SetStatusText("Can not load this file")
                return 1  # still was an error message.
            else:
                lineNoString = matchResult.group(2)
                # Attempt to locate the file (in case it is a relative spec)
                fileNameSpec = fileName
                fileName = scriptutils.LocatePythonFile(fileName)
                if fileName is None:
                    # Dont force update, so it replaces the idle prompt.
                    win32ui.SetStatusText(
                        "Cant locate the file '%s'" % (fileNameSpec), 0
                    )
                    return 1

                win32ui.SetStatusText(
                    "Jumping to line " + lineNoString + " of file " + fileName, 1
                )
                if not scriptutils.JumpToDocument(fileName, int(lineNoString)):
                    win32ui.SetStatusText("Could not open %s" % fileName)
                    return 1  # still was an error message.
                return 1
        return 0  # not an error line

    def write(self, msg):
        return self.template.write(msg)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        self.template.flush()


class WindowOutputViewRTF(docview.RichEditView, WindowOutputViewImpl):
    def __init__(self, doc):
        docview.RichEditView.__init__(self, doc)
        WindowOutputViewImpl.__init__(self)

    def OnInitialUpdate(self):
        WindowOutputViewImpl.OnInitialUpdate(self)
        return docview.RichEditView.OnInitialUpdate(self)

    def OnDestroy(self, msg):
        WindowOutputViewImpl.OnDestroy(self, msg)
        docview.RichEditView.OnDestroy(self, msg)

    def HookHandlers(self):
        WindowOutputViewImpl.HookHandlers(self)
        # Hook for finding and locating error messages
        self.HookMessage(self.OnLDoubleClick, win32con.WM_LBUTTONDBLCLK)

    # 		docview.RichEditView.HookHandlers(self)

    def OnLDoubleClick(self, params):
        if self.HandleSpecialLine():
            return 0  # dont pass on
        return 1  # pass it on by default.

    def RestoreKillBuffer(self):
        if len(self.template.killBuffer):
            self.StreamIn(win32con.SF_RTF, self._StreamRTFIn)
            self.template.killBuffer = []

    def SaveKillBuffer(self):
        self.StreamOut(win32con.SF_RTFNOOBJS, self._StreamRTFOut)

    def _StreamRTFOut(self, data):
        self.template.killBuffer.append(data)
        return 1  # keep em coming!

    def _StreamRTFIn(self, bytes):
        try:
            item = self.template.killBuffer[0]
            self.template.killBuffer.remove(item)
            if bytes < len(item):
                print("Warning - output buffer not big enough!")
            return item
        except IndexError:
            return None

    def dowrite(self, str):
        self.SetSel(-2)
        self.ReplaceSel(str)


import pywin.scintilla.view


class WindowOutputViewScintilla(
    pywin.scintilla.view.CScintillaView, WindowOutputViewImpl
):
    def __init__(self, doc):
        pywin.scintilla.view.CScintillaView.__init__(self, doc)
        WindowOutputViewImpl.__init__(self)

    def OnInitialUpdate(self):
        pywin.scintilla.view.CScintillaView.OnInitialUpdate(self)
        self.SCISetMarginWidth(3)
        WindowOutputViewImpl.OnInitialUpdate(self)

    def OnDestroy(self, msg):
        WindowOutputViewImpl.OnDestroy(self, msg)
        pywin.scintilla.view.CScintillaView.OnDestroy(self, msg)

    def HookHandlers(self):
        WindowOutputViewImpl.HookHandlers(self)
        pywin.scintilla.view.CScintillaView.HookHandlers(self)
        self.GetParent().HookNotify(
            self.OnScintillaDoubleClick, scintillacon.SCN_DOUBLECLICK
        )

    ##		self.HookMessage(self.OnLDoubleClick,win32con.WM_LBUTTONDBLCLK)

    def OnScintillaDoubleClick(self, std, extra):
        self.HandleSpecialLine()

    ##	def OnLDoubleClick(self,params):
    ##			return 0	# never dont pass on

    def RestoreKillBuffer(self):
        assert len(self.template.killBuffer) in (0, 1), "Unexpected killbuffer contents"
        if self.template.killBuffer:
            self.SCIAddText(self.template.killBuffer[0])
        self.template.killBuffer = []

    def SaveKillBuffer(self):
        self.template.killBuffer = [self.GetTextRange(0, -1)]

    def dowrite(self, str):
        end = self.GetTextLength()
        atEnd = end == self.GetSel()[0]
        self.SCIInsertText(str, end)
        if atEnd:
            self.SetSel(self.GetTextLength())

    def SetWordWrap(self, bWrapOn=1):
        if bWrapOn:
            wrap_mode = scintillacon.SC_WRAP_WORD
        else:
            wrap_mode = scintillacon.SC_WRAP_NONE
        self.SCISetWrapMode(wrap_mode)

    def _MakeColorizer(self):
        return None  # No colorizer for me!


WindowOutputView = WindowOutputViewScintilla


# The WindowOutput class is actually an MFC template.  This is a conventient way of
# making sure that my state can exist beyond the life of the windows themselves.
# This is primarily to support the functionality of a WindowOutput window automatically
# being recreated if necessary when written to.
class WindowOutput(docview.DocTemplate):
    """Looks like a general Output Window - text can be written by the 'write' method.
    Will auto-create itself on first write, and also on next write after being closed"""

    softspace = 1

    def __init__(
        self,
        title=None,
        defSize=None,
        queueing=flags.WQ_LINE,
        bAutoRestore=1,
        style=None,
        makeDoc=None,
        makeFrame=None,
        makeView=None,
    ):
        """init the output window -
        Params
        title=None -- What is the title of the window
        defSize=None -- What is the default size for the window - if this
                        is a string, the size will be loaded from the ini file.
        queueing = flags.WQ_LINE -- When should output be written
        bAutoRestore=1 -- Should a minimized window be restored.
        style -- Style for Window, or None for default.
        makeDoc, makeFrame, makeView -- Classes for frame, view and window respectively.
        """
        if makeDoc is None:
            makeDoc = WindowOutputDocument
        if makeFrame is None:
            makeFrame = WindowOutputFrame
        if makeView is None:
            makeView = WindowOutputViewScintilla
        docview.DocTemplate.__init__(
            self, win32ui.IDR_PYTHONTYPE, makeDoc, makeFrame, makeView
        )
        self.SetDocStrings("\nOutput\n\nText Documents (*.txt)\n.txt\n\n\n")
        win32ui.GetApp().AddDocTemplate(self)
        self.writeQueueing = queueing
        self.errorCantRecreate = 0
        self.killBuffer = []
        self.style = style
        self.bAutoRestore = bAutoRestore
        self.title = title
        self.bCreating = 0
        self.interruptCount = 0
        if type(defSize) == type(""):  # is a string - maintain size pos from ini file.
            self.iniSizeSection = defSize
            self.defSize = app.LoadWindowSize(defSize)
            self.loadedSize = self.defSize
        else:
            self.iniSizeSection = None
            self.defSize = defSize
        self.currentView = None
        self.outputQueue = queue.Queue(-1)
        self.mainThreadId = win32api.GetCurrentThreadId()
        self.idleHandlerSet = 0
        self.SetIdleHandler()

    def __del__(self):
        self.Close()

    def Create(self, title=None, style=None):
        self.bCreating = 1
        if title:
            self.title = title
        if style:
            self.style = style
        doc = self.OpenDocumentFile()
        if doc is None:
            return
        self.currentView = doc.GetFirstView()
        self.bCreating = 0
        if self.title:
            doc.SetTitle(self.title)

    def Close(self):
        self.RemoveIdleHandler()
        try:
            parent = self.currentView.GetParent()
        except (AttributeError, win32ui.error):  # Already closed
            return
        parent.DestroyWindow()

    def SetTitle(self, title):
        self.title = title
        if self.currentView:
            self.currentView.GetDocument().SetTitle(self.title)

    def OnViewDestroy(self, view):
        self.currentView.SaveKillBuffer()
        self.currentView = None

    def OnFrameDestroy(self, frame):
        if self.iniSizeSection:
            # use GetWindowPlacement(), as it works even when min'd or max'd
            newSize = frame.GetWindowPlacement()[4]
            if self.loadedSize != newSize:
                app.SaveWindowSize(self.iniSizeSection, newSize)

    def SetIdleHandler(self):
        if not self.idleHandlerSet:
            debug("Idle handler set\n")
            win32ui.GetApp().AddIdleHandler(self.QueueIdleHandler)
            self.idleHandlerSet = 1

    def RemoveIdleHandler(self):
        if self.idleHandlerSet:
            debug("Idle handler reset\n")
            if win32ui.GetApp().DeleteIdleHandler(self.QueueIdleHandler) == 0:
                debug("Error deleting idle handler\n")
            self.idleHandlerSet = 0

    def RecreateWindow(self):
        if self.errorCantRecreate:
            debug("Error = not trying again")
            return 0
        try:
            # This will fail if app shutting down
            win32ui.GetMainFrame().GetSafeHwnd()
            self.Create()
            return 1
        except (win32ui.error, AttributeError):
            self.errorCantRecreate = 1
            debug("Winout can not recreate the Window!\n")
            return 0

    # this handles the idle message, and does the printing.
    def QueueIdleHandler(self, handler, count):
        try:
            bEmpty = self.QueueFlush(20)
            # If the queue is empty, then we are back to idle and restart interrupt logic.
            if bEmpty:
                self.interruptCount = 0
        except KeyboardInterrupt:
            # First interrupt since idle we just pass on.
            # later ones we dump the queue and give up.
            self.interruptCount = self.interruptCount + 1
            if self.interruptCount > 1:
                # Drop the queue quickly as the user is already annoyed :-)
                self.outputQueue = queue.Queue(-1)
                print("Interrupted.")
                bEmpty = 1
            else:
                raise  # re-raise the error so the users exception filters up.
        return not bEmpty  # More to do if not empty.

    # Returns true if the Window needs to be recreated.
    def NeedRecreateWindow(self):
        try:
            if self.currentView is not None and self.currentView.IsWindow():
                return 0
        except (
            win32ui.error,
            AttributeError,
        ):  # Attribute error if the win32ui object has died.
            pass
        return 1

    # Returns true if the Window is OK (either cos it was, or because it was recreated
    def CheckRecreateWindow(self):
        if self.bCreating:
            return 1
        if not self.NeedRecreateWindow():
            return 1
        if self.bAutoRestore:
            if self.RecreateWindow():
                return 1
        return 0

    def QueueFlush(self, max=None):
        # Returns true if the queue is empty after the flush
        # 		debug("Queueflush - %d, %d\n" % (max, self.outputQueue.qsize()))
        if self.bCreating:
            return 1
        items = []
        rc = 0
        while max is None or max > 0:
            try:
                item = self.outputQueue.get_nowait()
                items.append(item)
            except queue.Empty:
                rc = 1
                break
            if max is not None:
                max = max - 1
        if len(items) != 0:
            if not self.CheckRecreateWindow():
                debug(":Recreate failed!\n")
                return 1  # In trouble - so say we have nothing to do.
            win32ui.PumpWaitingMessages()  # Pump paint messages
            self.currentView.dowrite("".join(items))
        return rc

    def HandleOutput(self, message):
        # 		debug("QueueOutput on thread %d, flags %d with '%s'...\n" % (win32api.GetCurrentThreadId(), self.writeQueueing, message ))
        self.outputQueue.put(message)
        if win32api.GetCurrentThreadId() != self.mainThreadId:
            pass
        # 			debug("not my thread - ignoring queue options!\n")
        elif self.writeQueueing == flags.WQ_LINE:
            pos = message.rfind("\n")
            if pos >= 0:
                # 				debug("Line queueing - forcing flush\n")
                self.QueueFlush()
                return
        elif self.writeQueueing == flags.WQ_NONE:
            # 			debug("WQ_NONE - flushing!\n")
            self.QueueFlush()
            return
        # Let our idle handler get it - wake it up
        try:
            win32ui.GetMainFrame().PostMessage(
                win32con.WM_USER
            )  # Kick main thread off.
        except win32ui.error:
            # This can happen as the app is shutting down, so we send it to the C++ debugger
            win32api.OutputDebugString(message)

    # delegate certain fns to my view.
    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def write(self, message):
        self.HandleOutput(message)

    def flush(self):
        self.QueueFlush()

    def HandleSpecialLine(self):
        self.currentView.HandleSpecialLine()


def RTFWindowOutput(*args, **kw):
    kw["makeView"] = WindowOutputViewRTF
    return WindowOutput(*args, **kw)


def thread_test(o):
    for i in range(5):
        o.write("Hi from thread %d\n" % (win32api.GetCurrentThreadId()))
        win32api.Sleep(100)


def test():
    w = WindowOutput(queueing=flags.WQ_IDLE)
    w.write("First bit of text\n")
    import _thread

    for i in range(5):
        w.write("Hello from the main thread\n")
        _thread.start_new(thread_test, (w,))
    for i in range(2):
        w.write("Hello from the main thread\n")
        win32api.Sleep(50)
    return w


if __name__ == "__main__":
    test()
