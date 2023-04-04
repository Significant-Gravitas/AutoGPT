# Color Editor originally by Neil Hodgson, but restructured by mh to integrate
# even tighter into Pythonwin.

import pywin.scintilla.keycodes
import win32api
import win32con
import win32ui
from pywin.framework.editor import (
    GetEditorFontOption,
    GetEditorOption,
    SetEditorFontOption,
    SetEditorOption,
    defaultCharacterFormat,
)
from pywin.scintilla import bindings

# from pywin.framework.editor import EditorPropertyPage

MSG_CHECK_EXTERNAL_FILE = (
    win32con.WM_USER + 1999
)  ## WARNING: Duplicated in document.py and editor.py

# Define a few common markers
MARKER_BOOKMARK = 0
MARKER_BREAKPOINT = 1
MARKER_CURRENT = 2

import pywin.scintilla.view
from pywin.debugger import dbgcon
from pywin.framework.editor.document import EditorDocumentBase
from pywin.scintilla import scintillacon  # For the marker definitions
from pywin.scintilla.document import CScintillaDocument


class SyntEditDocument(EditorDocumentBase):
    "A SyntEdit document."

    def OnDebuggerStateChange(self, state):
        self._ApplyOptionalToViews("OnDebuggerStateChange", state)

    def HookViewNotifications(self, view):
        EditorDocumentBase.HookViewNotifications(self, view)
        view.SCISetUndoCollection(1)

    def FinalizeViewCreation(self, view):
        EditorDocumentBase.FinalizeViewCreation(self, view)
        if view == self.GetFirstView():
            self.GetDocTemplate().CheckIDLEMenus(view.idle)


SyntEditViewParent = pywin.scintilla.view.CScintillaView


class SyntEditView(SyntEditViewParent):
    "A view of a SyntEdit.  Obtains data from document."

    def __init__(self, doc):
        SyntEditViewParent.__init__(self, doc)
        self.bCheckingFile = 0

    def OnInitialUpdate(self):
        SyntEditViewParent.OnInitialUpdate(self)

        self.HookMessage(self.OnRClick, win32con.WM_RBUTTONDOWN)

        for id in (
            win32ui.ID_VIEW_FOLD_COLLAPSE,
            win32ui.ID_VIEW_FOLD_COLLAPSE_ALL,
            win32ui.ID_VIEW_FOLD_EXPAND,
            win32ui.ID_VIEW_FOLD_EXPAND_ALL,
        ):
            self.HookCommand(self.OnCmdViewFold, id)
            self.HookCommandUpdate(self.OnUpdateViewFold, id)
        self.HookCommand(self.OnCmdViewFoldTopLevel, win32ui.ID_VIEW_FOLD_TOPLEVEL)

        # Define the markers
        # 		self.SCIMarkerDeleteAll()
        self.SCIMarkerDefineAll(
            MARKER_BOOKMARK,
            scintillacon.SC_MARK_ROUNDRECT,
            win32api.RGB(0x0, 0x0, 0x0),
            win32api.RGB(0, 0xFF, 0xFF),
        )

        self.SCIMarkerDefine(MARKER_CURRENT, scintillacon.SC_MARK_ARROW)
        self.SCIMarkerSetBack(MARKER_CURRENT, win32api.RGB(0xFF, 0xFF, 0x00))

        # Define the folding markers
        if 1:  # traditional markers
            self.SCIMarkerDefineAll(
                scintillacon.SC_MARKNUM_FOLDEROPEN,
                scintillacon.SC_MARK_MINUS,
                win32api.RGB(0xFF, 0xFF, 0xFF),
                win32api.RGB(0, 0, 0),
            )
            self.SCIMarkerDefineAll(
                scintillacon.SC_MARKNUM_FOLDER,
                scintillacon.SC_MARK_PLUS,
                win32api.RGB(0xFF, 0xFF, 0xFF),
                win32api.RGB(0, 0, 0),
            )
            self.SCIMarkerDefineAll(
                scintillacon.SC_MARKNUM_FOLDERSUB,
                scintillacon.SC_MARK_EMPTY,
                win32api.RGB(0xFF, 0xFF, 0xFF),
                win32api.RGB(0, 0, 0),
            )
            self.SCIMarkerDefineAll(
                scintillacon.SC_MARKNUM_FOLDERTAIL,
                scintillacon.SC_MARK_EMPTY,
                win32api.RGB(0xFF, 0xFF, 0xFF),
                win32api.RGB(0, 0, 0),
            )
            self.SCIMarkerDefineAll(
                scintillacon.SC_MARKNUM_FOLDEREND,
                scintillacon.SC_MARK_EMPTY,
                win32api.RGB(0xFF, 0xFF, 0xFF),
                win32api.RGB(0, 0, 0),
            )
            self.SCIMarkerDefineAll(
                scintillacon.SC_MARKNUM_FOLDEROPENMID,
                scintillacon.SC_MARK_EMPTY,
                win32api.RGB(0xFF, 0xFF, 0xFF),
                win32api.RGB(0, 0, 0),
            )
            self.SCIMarkerDefineAll(
                scintillacon.SC_MARKNUM_FOLDERMIDTAIL,
                scintillacon.SC_MARK_EMPTY,
                win32api.RGB(0xFF, 0xFF, 0xFF),
                win32api.RGB(0, 0, 0),
            )
        else:  # curved markers
            self.SCIMarkerDefineAll(
                scintillacon.SC_MARKNUM_FOLDEROPEN,
                scintillacon.SC_MARK_CIRCLEMINUS,
                win32api.RGB(0xFF, 0xFF, 0xFF),
                win32api.RGB(0, 0, 0),
            )
            self.SCIMarkerDefineAll(
                scintillacon.SC_MARKNUM_FOLDER,
                scintillacon.SC_MARK_CIRCLEPLUS,
                win32api.RGB(0xFF, 0xFF, 0xFF),
                win32api.RGB(0, 0, 0),
            )
            self.SCIMarkerDefineAll(
                scintillacon.SC_MARKNUM_FOLDERSUB,
                scintillacon.SC_MARK_VLINE,
                win32api.RGB(0xFF, 0xFF, 0xFF),
                win32api.RGB(0, 0, 0),
            )
            self.SCIMarkerDefineAll(
                scintillacon.SC_MARKNUM_FOLDERTAIL,
                scintillacon.SC_MARK_LCORNERCURVE,
                win32api.RGB(0xFF, 0xFF, 0xFF),
                win32api.RGB(0, 0, 0),
            )
            self.SCIMarkerDefineAll(
                scintillacon.SC_MARKNUM_FOLDEREND,
                scintillacon.SC_MARK_CIRCLEPLUSCONNECTED,
                win32api.RGB(0xFF, 0xFF, 0xFF),
                win32api.RGB(0, 0, 0),
            )
            self.SCIMarkerDefineAll(
                scintillacon.SC_MARKNUM_FOLDEROPENMID,
                scintillacon.SC_MARK_CIRCLEMINUSCONNECTED,
                win32api.RGB(0xFF, 0xFF, 0xFF),
                win32api.RGB(0, 0, 0),
            )
            self.SCIMarkerDefineAll(
                scintillacon.SC_MARKNUM_FOLDERMIDTAIL,
                scintillacon.SC_MARK_TCORNERCURVE,
                win32api.RGB(0xFF, 0xFF, 0xFF),
                win32api.RGB(0, 0, 0),
            )

        self.SCIMarkerDefine(MARKER_BREAKPOINT, scintillacon.SC_MARK_CIRCLE)
        # Marker background depends on debugger state
        self.SCIMarkerSetFore(MARKER_BREAKPOINT, win32api.RGB(0x0, 0, 0))
        # Get the current debugger state.
        try:
            import pywin.debugger

            if pywin.debugger.currentDebugger is None:
                state = dbgcon.DBGSTATE_NOT_DEBUGGING
            else:
                state = pywin.debugger.currentDebugger.debuggerState
        except ImportError:
            state = dbgcon.DBGSTATE_NOT_DEBUGGING
        self.OnDebuggerStateChange(state)

    def _GetSubConfigNames(self):
        return ["editor"]  # Allow [Keys:Editor] sections to be specific to us

    def DoConfigChange(self):
        SyntEditViewParent.DoConfigChange(self)
        tabSize = GetEditorOption("Tab Size", 4, 2)
        indentSize = GetEditorOption("Indent Size", 4, 2)
        bUseTabs = GetEditorOption("Use Tabs", 0)
        bSmartTabs = GetEditorOption("Smart Tabs", 1)
        ext = self.idle.IDLEExtension("AutoIndent")  # Required extension.

        self.SCISetViewWS(GetEditorOption("View Whitespace", 0))
        self.SCISetViewEOL(GetEditorOption("View EOL", 0))
        self.SCISetIndentationGuides(GetEditorOption("View Indentation Guides", 0))

        if GetEditorOption("Right Edge Enabled", 0):
            mode = scintillacon.EDGE_BACKGROUND
        else:
            mode = scintillacon.EDGE_NONE
        self.SCISetEdgeMode(mode)
        self.SCISetEdgeColumn(GetEditorOption("Right Edge Column", 75))
        self.SCISetEdgeColor(
            GetEditorOption("Right Edge Color", win32api.RGB(0xEF, 0xEF, 0xEF))
        )

        width = GetEditorOption("Marker Margin Width", 16)
        self.SCISetMarginWidthN(1, width)
        width = GetEditorOption("Fold Margin Width", 12)
        self.SCISetMarginWidthN(2, width)
        width = GetEditorOption("Line Number Margin Width", 0)
        self.SCISetMarginWidthN(0, width)
        self.bFolding = GetEditorOption("Enable Folding", 1)
        fold_flags = 0
        self.SendScintilla(
            scintillacon.SCI_SETMODEVENTMASK, scintillacon.SC_MOD_CHANGEFOLD
        )
        if self.bFolding:
            if GetEditorOption("Fold Lines", 1):
                fold_flags = 16

        self.SCISetProperty("fold", self.bFolding)
        self.SCISetFoldFlags(fold_flags)

        tt_color = GetEditorOption("Tab Timmy Color", win32api.RGB(0xFF, 0, 0))
        self.SendScintilla(scintillacon.SCI_INDICSETFORE, 1, tt_color)

        tt_use = GetEditorOption("Use Tab Timmy", 1)
        if tt_use:
            self.SCISetProperty("tab.timmy.whinge.level", "1")

        # Auto-indent has very complicated behaviour.  In a nutshell, the only
        # way to get sensible behaviour from it is to ensure tabwidth != indentsize.
        # Further, usetabs will only ever go from 1->0, never 0->1.
        # This is _not_ the behaviour Pythonwin wants:
        # * Tab width is arbitary, so should have no impact on smarts.
        # * bUseTabs setting should reflect how new files are created, and
        #   if Smart Tabs disabled, existing files are edited
        # * If "Smart Tabs" is enabled, bUseTabs should have no bearing
        #   for existing files (unless of course no context can be determined)
        #
        # So for smart tabs we configure the widget with completely dummy
        # values (ensuring tabwidth != indentwidth), ask it to guess, then
        # look at the values it has guessed, and re-configure
        if bSmartTabs:
            ext.config(usetabs=1, tabwidth=5, indentwidth=4)
            ext.set_indentation_params(1)
            if ext.indentwidth == 5:
                # Either 5 literal spaces, or a single tab character. Assume a tab
                usetabs = 1
                indentwidth = tabSize
            else:
                # Either Indented with spaces, and indent size has been guessed or
                # an empty file (or no context found - tough!)
                if self.GetTextLength() == 0:  # emtpy
                    usetabs = bUseTabs
                    indentwidth = indentSize
                else:  # guessed.
                    indentwidth = ext.indentwidth
                    usetabs = 0
            # Tab size can never be guessed - set at user preference.
            ext.config(usetabs=usetabs, indentwidth=indentwidth, tabwidth=tabSize)
        else:
            # Dont want smart-tabs - just set the options!
            ext.config(usetabs=bUseTabs, tabwidth=tabSize, indentwidth=indentSize)
        self.SCISetIndent(indentSize)
        self.SCISetTabWidth(tabSize)

    def OnDebuggerStateChange(self, state):
        if state == dbgcon.DBGSTATE_NOT_DEBUGGING:
            # Indicate breakpoints arent really usable.
            # Not quite white - useful when no marker margin, so set as background color.
            self.SCIMarkerSetBack(MARKER_BREAKPOINT, win32api.RGB(0xEF, 0xEF, 0xEF))
        else:
            # A light-red, so still readable when no marker margin.
            self.SCIMarkerSetBack(MARKER_BREAKPOINT, win32api.RGB(0xFF, 0x80, 0x80))

    def HookDocumentHandlers(self):
        SyntEditViewParent.HookDocumentHandlers(self)
        self.HookMessage(self.OnCheckExternalDocumentUpdated, MSG_CHECK_EXTERNAL_FILE)

    def HookHandlers(self):
        SyntEditViewParent.HookHandlers(self)
        self.HookMessage(self.OnSetFocus, win32con.WM_SETFOCUS)

    def _PrepareUserStateChange(self):
        return self.GetSel(), self.GetFirstVisibleLine()

    def _EndUserStateChange(self, info):
        scrollOff = info[1] - self.GetFirstVisibleLine()
        if scrollOff:
            self.LineScroll(scrollOff)
        # Make sure we dont reset the cursor beyond the buffer.
        max = self.GetTextLength()
        newPos = min(info[0][0], max), min(info[0][1], max)
        self.SetSel(newPos)

    #######################################
    # The Windows Message or Notify handlers.
    #######################################
    def OnMarginClick(self, std, extra):
        notify = self.SCIUnpackNotifyMessage(extra)
        if notify.margin == 2:  # Our fold margin
            line_click = self.LineFromChar(notify.position)
            # 			max_line = self.GetLineCount()
            if self.SCIGetFoldLevel(line_click) & scintillacon.SC_FOLDLEVELHEADERFLAG:
                # If a fold point.
                self.SCIToggleFold(line_click)
        return 1

    def OnSetFocus(self, msg):
        # Even though we use file change notifications, we should be very sure about it here.
        self.OnCheckExternalDocumentUpdated(msg)
        return 1

    def OnCheckExternalDocumentUpdated(self, msg):
        if self.bCheckingFile:
            return
        self.bCheckingFile = 1
        self.GetDocument().CheckExternalDocumentUpdated()
        self.bCheckingFile = 0

    def OnRClick(self, params):
        menu = win32ui.CreatePopupMenu()
        self.AppendMenu(menu, "&Locate module", "LocateModule")
        self.AppendMenu(menu, flags=win32con.MF_SEPARATOR)
        self.AppendMenu(menu, "&Undo", "EditUndo")
        self.AppendMenu(menu, "&Redo", "EditRedo")
        self.AppendMenu(menu, flags=win32con.MF_SEPARATOR)
        self.AppendMenu(menu, "Cu&t", "EditCut")
        self.AppendMenu(menu, "&Copy", "EditCopy")
        self.AppendMenu(menu, "&Paste", "EditPaste")
        self.AppendMenu(menu, flags=win32con.MF_SEPARATOR)
        self.AppendMenu(menu, "&Select all", "EditSelectAll")
        self.AppendMenu(
            menu, "View &Whitespace", "ViewWhitespace", checked=self.SCIGetViewWS()
        )
        self.AppendMenu(
            menu, "&Fixed Font", "ViewFixedFont", checked=self._GetColorizer().bUseFixed
        )
        self.AppendMenu(menu, flags=win32con.MF_SEPARATOR)
        self.AppendMenu(menu, "&Goto line...", "GotoLine")

        submenu = win32ui.CreatePopupMenu()
        newitems = self.idle.GetMenuItems("edit")
        for text, event in newitems:
            self.AppendMenu(submenu, text, event)

        flags = win32con.MF_STRING | win32con.MF_ENABLED | win32con.MF_POPUP
        menu.AppendMenu(flags, submenu.GetHandle(), "&Source code")

        flags = (
            win32con.TPM_LEFTALIGN | win32con.TPM_LEFTBUTTON | win32con.TPM_RIGHTBUTTON
        )
        menu.TrackPopupMenu(params[5], flags, self)
        return 0

    def OnCmdViewFold(self, cid, code):  # Handle the menu command
        if cid == win32ui.ID_VIEW_FOLD_EXPAND_ALL:
            self.FoldExpandAllEvent(None)
        elif cid == win32ui.ID_VIEW_FOLD_EXPAND:
            self.FoldExpandEvent(None)
        elif cid == win32ui.ID_VIEW_FOLD_COLLAPSE_ALL:
            self.FoldCollapseAllEvent(None)
        elif cid == win32ui.ID_VIEW_FOLD_COLLAPSE:
            self.FoldCollapseEvent(None)
        else:
            print("Unknown collapse/expand ID")

    def OnUpdateViewFold(self, cmdui):  # Update the tick on the UI.
        if not self.bFolding:
            cmdui.Enable(0)
            return
        id = cmdui.m_nID
        if id in (win32ui.ID_VIEW_FOLD_EXPAND_ALL, win32ui.ID_VIEW_FOLD_COLLAPSE_ALL):
            cmdui.Enable()
        else:
            enable = 0
            lineno = self.LineFromChar(self.GetSel()[0])
            foldable = (
                self.SCIGetFoldLevel(lineno) & scintillacon.SC_FOLDLEVELHEADERFLAG
            )
            is_expanded = self.SCIGetFoldExpanded(lineno)
            if id == win32ui.ID_VIEW_FOLD_EXPAND:
                if foldable and not is_expanded:
                    enable = 1
            elif id == win32ui.ID_VIEW_FOLD_COLLAPSE:
                if foldable and is_expanded:
                    enable = 1
            cmdui.Enable(enable)

    def OnCmdViewFoldTopLevel(self, cid, code):  # Handle the menu command
        self.FoldTopLevelEvent(None)

    #######################################
    # The Events
    #######################################
    def ToggleBookmarkEvent(self, event, pos=-1):
        """Toggle a bookmark at the specified or current position"""
        if pos == -1:
            pos, end = self.GetSel()
        startLine = self.LineFromChar(pos)
        self.GetDocument().MarkerToggle(startLine + 1, MARKER_BOOKMARK)
        return 0

    def GotoNextBookmarkEvent(self, event, fromPos=-1):
        """Move to the next bookmark"""
        if fromPos == -1:
            fromPos, end = self.GetSel()
        startLine = self.LineFromChar(fromPos) + 1  # Zero based line to start
        nextLine = self.GetDocument().MarkerGetNext(startLine + 1, MARKER_BOOKMARK) - 1
        if nextLine < 0:
            nextLine = self.GetDocument().MarkerGetNext(0, MARKER_BOOKMARK) - 1
        if nextLine < 0 or nextLine == startLine - 1:
            win32api.MessageBeep()
        else:
            self.SCIEnsureVisible(nextLine)
            self.SCIGotoLine(nextLine)
        return 0

    def TabKeyEvent(self, event):
        """Insert an indent.  If no selection, a single indent, otherwise a block indent"""
        # Handle auto-complete first.
        if self.SCIAutoCActive():
            self.SCIAutoCComplete()
            return 0
        # Call the IDLE event.
        return self.bindings.fire("<<smart-indent>>", event)

    def EnterKeyEvent(self, event):
        """Handle the enter key with special handling for auto-complete"""
        # Handle auto-complete first.
        if self.SCIAutoCActive():
            self.SCIAutoCComplete()
            self.SCIAutoCCancel()
        # Call the IDLE event.
        return self.bindings.fire("<<newline-and-indent>>", event)

    def ShowInteractiveWindowEvent(self, event):
        import pywin.framework.interact

        pywin.framework.interact.ShowInteractiveWindow()

    def FoldTopLevelEvent(self, event=None):
        if not self.bFolding:
            return 1

        win32ui.DoWaitCursor(1)
        try:
            self.Colorize()
            maxLine = self.GetLineCount()
            # Find the first line, and check out its state.
            for lineSeek in range(maxLine):
                if self.SCIGetFoldLevel(lineSeek) & scintillacon.SC_FOLDLEVELHEADERFLAG:
                    expanding = not self.SCIGetFoldExpanded(lineSeek)
                    break
            else:
                # no folds here!
                return
            for lineSeek in range(lineSeek, maxLine):
                level = self.SCIGetFoldLevel(lineSeek)
                level_no = (
                    level
                    & scintillacon.SC_FOLDLEVELNUMBERMASK
                    - scintillacon.SC_FOLDLEVELBASE
                )
                is_header = level & scintillacon.SC_FOLDLEVELHEADERFLAG
                # 			print lineSeek, level_no, is_header
                if level_no == 0 and is_header:
                    if (expanding and not self.SCIGetFoldExpanded(lineSeek)) or (
                        not expanding and self.SCIGetFoldExpanded(lineSeek)
                    ):
                        self.SCIToggleFold(lineSeek)
        finally:
            win32ui.DoWaitCursor(-1)

    def FoldExpandSecondLevelEvent(self, event):
        if not self.bFolding:
            return 1
        win32ui.DoWaitCursor(1)
        ## I think this is needed since Scintilla may not have
        ## already formatted parts of file outside visible window.
        self.Colorize()
        levels = [scintillacon.SC_FOLDLEVELBASE]
        ## Scintilla's level number is based on amount of whitespace indentation
        for lineno in range(self.GetLineCount()):
            level = self.SCIGetFoldLevel(lineno)
            if not level & scintillacon.SC_FOLDLEVELHEADERFLAG:
                continue
            curr_level = level & scintillacon.SC_FOLDLEVELNUMBERMASK
            if curr_level > levels[-1]:
                levels.append(curr_level)
            try:
                level_ind = levels.index(curr_level)
            except ValueError:
                ## probably syntax error in source file, bail
                break
            levels = levels[: level_ind + 1]
            if level_ind == 1 and not self.SCIGetFoldExpanded(lineno):
                self.SCIToggleFold(lineno)
        win32ui.DoWaitCursor(-1)

    def FoldCollapseSecondLevelEvent(self, event):
        if not self.bFolding:
            return 1
        win32ui.DoWaitCursor(1)
        ## I think this is needed since Scintilla may not have
        ## already formatted parts of file outside visible window.
        self.Colorize()
        levels = [scintillacon.SC_FOLDLEVELBASE]
        ## Scintilla's level number is based on amount of whitespace indentation
        for lineno in range(self.GetLineCount()):
            level = self.SCIGetFoldLevel(lineno)
            if not level & scintillacon.SC_FOLDLEVELHEADERFLAG:
                continue
            curr_level = level & scintillacon.SC_FOLDLEVELNUMBERMASK
            if curr_level > levels[-1]:
                levels.append(curr_level)
            try:
                level_ind = levels.index(curr_level)
            except ValueError:
                ## probably syntax error in source file, bail
                break
            levels = levels[: level_ind + 1]
            if level_ind == 1 and self.SCIGetFoldExpanded(lineno):
                self.SCIToggleFold(lineno)
        win32ui.DoWaitCursor(-1)

    def FoldExpandEvent(self, event):
        if not self.bFolding:
            return 1
        win32ui.DoWaitCursor(1)
        lineno = self.LineFromChar(self.GetSel()[0])
        if self.SCIGetFoldLevel(
            lineno
        ) & scintillacon.SC_FOLDLEVELHEADERFLAG and not self.SCIGetFoldExpanded(lineno):
            self.SCIToggleFold(lineno)
        win32ui.DoWaitCursor(-1)

    def FoldExpandAllEvent(self, event):
        if not self.bFolding:
            return 1
        win32ui.DoWaitCursor(1)
        for lineno in range(0, self.GetLineCount()):
            if self.SCIGetFoldLevel(
                lineno
            ) & scintillacon.SC_FOLDLEVELHEADERFLAG and not self.SCIGetFoldExpanded(
                lineno
            ):
                self.SCIToggleFold(lineno)
        win32ui.DoWaitCursor(-1)

    def FoldCollapseEvent(self, event):
        if not self.bFolding:
            return 1
        win32ui.DoWaitCursor(1)
        lineno = self.LineFromChar(self.GetSel()[0])
        if self.SCIGetFoldLevel(
            lineno
        ) & scintillacon.SC_FOLDLEVELHEADERFLAG and self.SCIGetFoldExpanded(lineno):
            self.SCIToggleFold(lineno)
        win32ui.DoWaitCursor(-1)

    def FoldCollapseAllEvent(self, event):
        if not self.bFolding:
            return 1
        win32ui.DoWaitCursor(1)
        self.Colorize()
        for lineno in range(0, self.GetLineCount()):
            if self.SCIGetFoldLevel(
                lineno
            ) & scintillacon.SC_FOLDLEVELHEADERFLAG and self.SCIGetFoldExpanded(lineno):
                self.SCIToggleFold(lineno)
        win32ui.DoWaitCursor(-1)


from pywin.framework.editor.frame import EditorFrame


class SplitterFrame(EditorFrame):
    def OnCreate(self, cs):
        self.HookCommand(self.OnWindowSplit, win32ui.ID_WINDOW_SPLIT)
        return 1

    def OnWindowSplit(self, id, code):
        self.GetDlgItem(win32ui.AFX_IDW_PANE_FIRST).DoKeyboardSplit()
        return 1


from pywin.framework.editor.template import EditorTemplateBase


class SyntEditTemplate(EditorTemplateBase):
    def __init__(
        self, res=win32ui.IDR_TEXTTYPE, makeDoc=None, makeFrame=None, makeView=None
    ):
        if makeDoc is None:
            makeDoc = SyntEditDocument
        if makeView is None:
            makeView = SyntEditView
        if makeFrame is None:
            makeFrame = SplitterFrame
        self.bSetMenus = 0
        EditorTemplateBase.__init__(self, res, makeDoc, makeFrame, makeView)

    def CheckIDLEMenus(self, idle):
        if self.bSetMenus:
            return
        self.bSetMenus = 1

        submenu = win32ui.CreatePopupMenu()
        newitems = idle.GetMenuItems("edit")
        flags = win32con.MF_STRING | win32con.MF_ENABLED
        for text, event in newitems:
            id = bindings.event_to_commands.get(event)
            if id is not None:
                keyname = pywin.scintilla.view.configManager.get_key_binding(
                    event, ["editor"]
                )
                if keyname is not None:
                    text = text + "\t" + keyname
                submenu.AppendMenu(flags, id, text)

        mainMenu = self.GetSharedMenu()
        editMenu = mainMenu.GetSubMenu(1)
        editMenu.AppendMenu(win32con.MF_SEPARATOR, 0, "")
        editMenu.AppendMenu(
            win32con.MF_STRING | win32con.MF_POPUP | win32con.MF_ENABLED,
            submenu.GetHandle(),
            "&Source Code",
        )

    def _CreateDocTemplate(self, resourceId):
        return win32ui.CreateDocTemplate(resourceId)

    def CreateWin32uiDocument(self):
        return self.DoCreateDoc()

    def GetPythonPropertyPages(self):
        """Returns a list of property pages"""
        from pywin.scintilla import configui

        return EditorTemplateBase.GetPythonPropertyPages(self) + [
            configui.ScintillaFormatPropertyPage()
        ]


# For debugging purposes, when this module may be reloaded many times.
try:
    win32ui.GetApp().RemoveDocTemplate(editorTemplate)
except NameError:
    pass

editorTemplate = SyntEditTemplate()
win32ui.GetApp().AddDocTemplate(editorTemplate)
