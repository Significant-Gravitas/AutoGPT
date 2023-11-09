##################################################################
##
## Interactive Shell Window
##

import array
import code
import os
import string
import sys
import traceback

import __main__
import afxres
import pywin.framework.app
import pywin.scintilla.control
import pywin.scintilla.formatter
import pywin.scintilla.IDLEenvironment
import win32api
import win32clipboard
import win32con
import win32ui

## sequential after ID_GOTO_LINE defined in editor.py
ID_EDIT_COPY_CODE = 0xE2002
ID_EDIT_EXEC_CLIPBOARD = 0x2003

trace = pywin.scintilla.formatter.trace

import re

from . import winout

# from IDLE.
_is_block_opener = re.compile(r":\s*(#.*)?$").search
_is_block_closer = re.compile(
    r"""
    \s*
    ( return
    | break
    | continue
    | raise
    | pass
    )
    \b
""",
    re.VERBOSE,
).match

tracebackHeader = "Traceback (".encode("ascii")

sectionProfile = "Interactive Window"
valueFormatTitle = "FormatTitle"
valueFormatInput = "FormatInput"
valueFormatOutput = "FormatOutput"
valueFormatOutputError = "FormatOutputError"

# These are defaults only.  Values are read from the registry.
formatTitle = (-536870897, 0, 220, 0, 16711680, 184, 34, "Arial")
formatInput = (-402653169, 0, 200, 0, 0, 0, 49, "Courier New")
formatOutput = (-402653169, 0, 200, 0, 8421376, 0, 49, "Courier New")
formatOutputError = (-402653169, 0, 200, 0, 255, 0, 49, "Courier New")

try:
    sys.ps1
except AttributeError:
    sys.ps1 = ">>> "
    sys.ps2 = "... "


def LoadPreference(preference, default=""):
    return win32ui.GetProfileVal(sectionProfile, preference, default)


def SavePreference(prefName, prefValue):
    win32ui.WriteProfileVal(sectionProfile, prefName, prefValue)


def GetPromptPrefix(line):
    ps1 = sys.ps1
    if line[: len(ps1)] == ps1:
        return ps1
    ps2 = sys.ps2
    if line[: len(ps2)] == ps2:
        return ps2


#############################################################
#
# Colorizer related code.
#
#############################################################
STYLE_INTERACTIVE_EOL = "Interactive EOL"
STYLE_INTERACTIVE_OUTPUT = "Interactive Output"
STYLE_INTERACTIVE_PROMPT = "Interactive Prompt"
STYLE_INTERACTIVE_BANNER = "Interactive Banner"
STYLE_INTERACTIVE_ERROR = "Interactive Error"
STYLE_INTERACTIVE_ERROR_FINALLINE = "Interactive Error (final line)"

INTERACTIVE_STYLES = [
    STYLE_INTERACTIVE_EOL,
    STYLE_INTERACTIVE_OUTPUT,
    STYLE_INTERACTIVE_PROMPT,
    STYLE_INTERACTIVE_BANNER,
    STYLE_INTERACTIVE_ERROR,
    STYLE_INTERACTIVE_ERROR_FINALLINE,
]

FormatterParent = pywin.scintilla.formatter.PythonSourceFormatter


class InteractiveFormatter(FormatterParent):
    def __init__(self, scintilla):
        FormatterParent.__init__(self, scintilla)
        self.bannerDisplayed = False

    def SetStyles(self):
        FormatterParent.SetStyles(self)
        Style = pywin.scintilla.formatter.Style
        self.RegisterStyle(Style(STYLE_INTERACTIVE_EOL, STYLE_INTERACTIVE_PROMPT))
        self.RegisterStyle(Style(STYLE_INTERACTIVE_PROMPT, formatInput))
        self.RegisterStyle(Style(STYLE_INTERACTIVE_OUTPUT, formatOutput))
        self.RegisterStyle(Style(STYLE_INTERACTIVE_BANNER, formatTitle))
        self.RegisterStyle(Style(STYLE_INTERACTIVE_ERROR, formatOutputError))
        self.RegisterStyle(
            Style(STYLE_INTERACTIVE_ERROR_FINALLINE, STYLE_INTERACTIVE_ERROR)
        )

    def LoadPreference(self, name, default):
        rc = win32ui.GetProfileVal("Format", name, default)
        if rc == default:
            rc = win32ui.GetProfileVal(sectionProfile, name, default)
        return rc

    def ColorizeInteractiveCode(self, cdoc, styleStart, stylePyStart):
        lengthDoc = len(cdoc)
        if lengthDoc == 0:
            return
        state = styleStart
        # As per comments in Colorize(), we work with the raw utf8
        # bytes. To avoid too muych py3k pain, we treat each utf8 byte
        # as a latin-1 unicode character - we only use it to compare
        # against ascii chars anyway...
        chNext = cdoc[0:1].decode("latin-1")
        startSeg = 0
        i = 0
        lastState = state  # debug only
        while i < lengthDoc:
            ch = chNext
            chNext = cdoc[i + 1 : i + 2].decode("latin-1")

            # 			trace("ch=%r, i=%d, next=%r, state=%s" % (ch, i, chNext, state))
            if state == STYLE_INTERACTIVE_EOL:
                if ch not in "\r\n":
                    self.ColorSeg(startSeg, i - 1, state)
                    startSeg = i
                    if ch in (sys.ps1[0], sys.ps2[0]):
                        state = STYLE_INTERACTIVE_PROMPT
                    elif cdoc[i : i + len(tracebackHeader)] == tracebackHeader:
                        state = STYLE_INTERACTIVE_ERROR
                    else:
                        state = STYLE_INTERACTIVE_OUTPUT
            elif state == STYLE_INTERACTIVE_PROMPT:
                if ch not in sys.ps1 + sys.ps2 + " ":
                    self.ColorSeg(startSeg, i - 1, state)
                    startSeg = i
                    if ch in "\r\n":
                        state = STYLE_INTERACTIVE_EOL
                    else:
                        state = stylePyStart  # Start coloring Python code.
            elif state in (STYLE_INTERACTIVE_OUTPUT,):
                if ch in "\r\n":
                    self.ColorSeg(startSeg, i - 1, state)
                    startSeg = i
                    state = STYLE_INTERACTIVE_EOL
            elif state == STYLE_INTERACTIVE_ERROR:
                if ch in "\r\n" and chNext and chNext not in string.whitespace:
                    # Everything including me
                    self.ColorSeg(startSeg, i, state)
                    startSeg = i + 1
                    state = STYLE_INTERACTIVE_ERROR_FINALLINE
                elif i == 0 and ch not in string.whitespace:
                    # If we are coloring from the start of a line,
                    # we need this better check for the last line
                    # Color up to not including me
                    self.ColorSeg(startSeg, i - 1, state)
                    startSeg = i
                    state = STYLE_INTERACTIVE_ERROR_FINALLINE
            elif state == STYLE_INTERACTIVE_ERROR_FINALLINE:
                if ch in "\r\n":
                    self.ColorSeg(startSeg, i - 1, state)
                    startSeg = i
                    state = STYLE_INTERACTIVE_EOL
            elif state == STYLE_INTERACTIVE_BANNER:
                if ch in "\r\n" and (chNext == "" or chNext in ">["):
                    # Everything including me
                    self.ColorSeg(startSeg, i - 1, state)
                    startSeg = i
                    state = STYLE_INTERACTIVE_EOL
            else:
                # It is a PythonColorizer state - seek past the end of the line
                # and ask the Python colorizer to color that.
                end = startSeg
                while end < lengthDoc and cdoc[end] not in "\r\n".encode("ascii"):
                    end = end + 1
                self.ColorizePythonCode(cdoc[:end], startSeg, state)
                stylePyStart = self.GetStringStyle(end - 1)
                if stylePyStart is None:
                    stylePyStart = pywin.scintilla.formatter.STYLE_DEFAULT
                else:
                    stylePyStart = stylePyStart.name
                startSeg = end
                i = end - 1  # ready for increment.
                chNext = cdoc[end : end + 1].decode("latin-1")
                state = STYLE_INTERACTIVE_EOL
            if lastState != state:
                lastState = state
            i = i + 1
        # and the rest
        if startSeg < i:
            self.ColorSeg(startSeg, i - 1, state)

    def Colorize(self, start=0, end=-1):
        # scintilla's formatting is all done in terms of utf, so
        # we work with utf8 bytes instead of unicode.  This magically
        # works as any extended chars found in the utf8 don't change
        # the semantics.
        stringVal = self.scintilla.GetTextRange(start, end, decode=False)
        styleStart = None
        stylePyStart = None
        if start > 1:
            # Likely we are being asked to color from the start of the line.
            # We find the last formatted character on the previous line.
            # If TQString, we continue it.  Otherwise, we reset.
            look = start - 1
            while look and self.scintilla.SCIGetCharAt(look) in "\n\r":
                look = look - 1
            if look and look < start - 1:  # Did we find a char before the \n\r sets?
                strstyle = self.GetStringStyle(look)
                quote_char = None
                if strstyle is not None:
                    if strstyle.name == pywin.scintilla.formatter.STYLE_TQSSTRING:
                        quote_char = "'"
                    elif strstyle.name == pywin.scintilla.formatter.STYLE_TQDSTRING:
                        quote_char = '"'
                    if quote_char is not None:
                        # It is a TQS.  If the TQS is not terminated, we
                        # carry the style through.
                        if look > 2:
                            look_str = (
                                self.scintilla.SCIGetCharAt(look - 2)
                                + self.scintilla.SCIGetCharAt(look - 1)
                                + self.scintilla.SCIGetCharAt(look)
                            )
                            if look_str != quote_char * 3:
                                stylePyStart = strstyle.name
        if stylePyStart is None:
            stylePyStart = pywin.scintilla.formatter.STYLE_DEFAULT

        if start > 0:
            stylenum = self.scintilla.SCIGetStyleAt(start - 1)
            styleStart = self.GetStyleByNum(stylenum).name
        elif self.bannerDisplayed:
            styleStart = STYLE_INTERACTIVE_EOL
        else:
            styleStart = STYLE_INTERACTIVE_BANNER
            self.bannerDisplayed = True
        self.scintilla.SCIStartStyling(start, 31)
        self.style_buffer = array.array("b", (0,) * len(stringVal))
        self.ColorizeInteractiveCode(stringVal, styleStart, stylePyStart)
        self.scintilla.SCISetStylingEx(self.style_buffer)
        self.style_buffer = None


###############################################################
#
# This class handles the Python interactive interpreter.
#
# It uses a basic EditWindow, and does all the magic.
# This is triggered by the enter key hander attached by the
# start-up code.  It determines if a command is to be executed
# or continued (ie, emit "... ") by snooping around the current
# line, looking for the prompts
#
class PythonwinInteractiveInterpreter(code.InteractiveInterpreter):
    def __init__(self, locals=None, globals=None):
        if locals is None:
            locals = __main__.__dict__
        if globals is None:
            globals = locals
        self.globals = globals
        code.InteractiveInterpreter.__init__(self, locals)

    def showsyntaxerror(self, filename=None):
        sys.stderr.write(
            tracebackHeader.decode("ascii")
        )  # So the color syntaxer recognises it.
        code.InteractiveInterpreter.showsyntaxerror(self, filename)

    def runcode(self, code):
        try:
            exec(code, self.globals, self.locals)
        except SystemExit:
            raise
        except:
            self.showtraceback()


class InteractiveCore:
    def __init__(self, banner=None):
        self.banner = banner

    # 		LoadFontPreferences()
    def Init(self):
        self.oldStdOut = self.oldStdErr = None

        # 		self.SetWordWrap(win32ui.CRichEditView_WrapNone)
        self.interp = PythonwinInteractiveInterpreter()

        self.OutputGrab()  # Release at cleanup.

        if self.GetTextLength() == 0:
            if self.banner is None:
                suffix = ""
                if win32ui.debug:
                    suffix = ", debug build"
                sys.stderr.write(
                    "PythonWin %s on %s%s.\n" % (sys.version, sys.platform, suffix)
                )
                sys.stderr.write(
                    "Portions %s - see 'Help/About PythonWin' for further copyright information.\n"
                    % (win32ui.copyright,)
                )
            else:
                sys.stderr.write(banner)
        rcfile = os.environ.get("PYTHONSTARTUP")
        if rcfile:
            import __main__

            try:
                exec(
                    compile(
                        open(rcfile, "rb").read(), rcfile, "exec", dont_inherit=True
                    ),
                    __main__.__dict__,
                    __main__.__dict__,
                )
            except:
                sys.stderr.write(
                    ">>> \nError executing PYTHONSTARTUP script %r\n" % (rcfile)
                )
                traceback.print_exc(file=sys.stderr)
        self.AppendToPrompt([])

    def SetContext(self, globals, locals, name="Dbg"):
        oldPrompt = sys.ps1
        if globals is None:
            # Reset
            sys.ps1 = ">>> "
            sys.ps2 = "... "
            locals = globals = __main__.__dict__
        else:
            sys.ps1 = "[%s]>>> " % name
            sys.ps2 = "[%s]... " % name
        self.interp.locals = locals
        self.interp.globals = globals
        self.AppendToPrompt([], oldPrompt)

    def GetContext(self):
        return self.interp.globals, self.interp.locals

    def DoGetLine(self, line=-1):
        if line == -1:
            line = self.LineFromChar()
        line = self.GetLine(line)
        while line and line[-1] in ("\r", "\n"):
            line = line[:-1]
        return line

    def AppendToPrompt(self, bufLines, oldPrompt=None):
        "Take a command and stick it at the end of the buffer (with python prompts inserted if required)."
        self.flush()
        lastLineNo = self.GetLineCount() - 1
        line = self.DoGetLine(lastLineNo)
        if oldPrompt and line == oldPrompt:
            self.SetSel(self.GetTextLength() - len(oldPrompt), self.GetTextLength())
            self.ReplaceSel(sys.ps1)
        elif line != str(sys.ps1):
            if len(line) != 0:
                self.write("\n")
            self.write(sys.ps1)
        self.flush()
        self.idle.text.mark_set("iomark", "end-1c")
        if not bufLines:
            return
        terms = (["\n" + sys.ps2] * (len(bufLines) - 1)) + [""]
        for bufLine, term in zip(bufLines, terms):
            if bufLine.strip():
                self.write(bufLine + term)
        self.flush()

    def EnsureNoPrompt(self):
        # Get ready to write some text NOT at a Python prompt.
        self.flush()
        lastLineNo = self.GetLineCount() - 1
        line = self.DoGetLine(lastLineNo)
        if not line or line in (sys.ps1, sys.ps2):
            self.SetSel(self.GetTextLength() - len(line), self.GetTextLength())
            self.ReplaceSel("")
        else:
            # Just add a new line.
            self.write("\n")

    def _GetSubConfigNames(self):
        return ["interactive"]  # Allow [Keys:Interactive] sections to be specific

    def HookHandlers(self):
        # Hook menu command (executed when a menu item with that ID is selected from a menu/toolbar
        self.HookCommand(self.OnSelectBlock, win32ui.ID_EDIT_SELECT_BLOCK)
        self.HookCommand(self.OnEditCopyCode, ID_EDIT_COPY_CODE)
        self.HookCommand(self.OnEditExecClipboard, ID_EDIT_EXEC_CLIPBOARD)
        mod = pywin.scintilla.IDLEenvironment.GetIDLEModule("IdleHistory")
        if mod is not None:
            self.history = mod.History(self.idle.text, "\n" + sys.ps2)
        else:
            self.history = None
        # hack for now for event handling.

    # GetBlockBoundary takes a line number, and will return the
    # start and and line numbers of the block, and a flag indicating if the
    # block is a Python code block.
    # If the line specified has a Python prompt, then the lines are parsed
    # backwards and forwards, and the flag is true.
    # If the line does not start with a prompt, the block is searched forward
    # and backward until a prompt _is_ found, and all lines in between without
    # prompts are returned, and the flag is false.
    def GetBlockBoundary(self, lineNo):
        line = self.DoGetLine(lineNo)
        maxLineNo = self.GetLineCount() - 1
        prefix = GetPromptPrefix(line)
        if prefix is None:  # Non code block
            flag = 0
            startLineNo = lineNo
            while startLineNo > 0:
                if GetPromptPrefix(self.DoGetLine(startLineNo - 1)) is not None:
                    break  # there _is_ a prompt
                startLineNo = startLineNo - 1
            endLineNo = lineNo
            while endLineNo < maxLineNo:
                if GetPromptPrefix(self.DoGetLine(endLineNo + 1)) is not None:
                    break  # there _is_ a prompt
                endLineNo = endLineNo + 1
        else:  # Code block
            flag = 1
            startLineNo = lineNo
            while startLineNo > 0 and prefix != str(sys.ps1):
                prefix = GetPromptPrefix(self.DoGetLine(startLineNo - 1))
                if prefix is None:
                    break
                    # there is no prompt.
                startLineNo = startLineNo - 1
            endLineNo = lineNo
            while endLineNo < maxLineNo:
                prefix = GetPromptPrefix(self.DoGetLine(endLineNo + 1))
                if prefix is None:
                    break  # there is no prompt
                if prefix == str(sys.ps1):
                    break  # this is another command
                endLineNo = endLineNo + 1
                # continue until end of buffer, or no prompt
        return (startLineNo, endLineNo, flag)

    def ExtractCommand(self, lines):
        start, end = lines
        retList = []
        while end >= start:
            thisLine = self.DoGetLine(end)
            promptLen = len(GetPromptPrefix(thisLine))
            retList = [thisLine[promptLen:]] + retList
            end = end - 1
        return retList

    def OutputGrab(self):
        # 		import win32traceutil; return
        self.oldStdOut = sys.stdout
        self.oldStdErr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        self.flush()

    def OutputRelease(self):
        # a command may have overwritten these - only restore if not.
        if self.oldStdOut is not None:
            if sys.stdout == self:
                sys.stdout = self.oldStdOut
        if self.oldStdErr is not None:
            if sys.stderr == self:
                sys.stderr = self.oldStdErr
        self.oldStdOut = None
        self.oldStdErr = None
        self.flush()

    ###################################
    #
    # Message/Command/Key Hooks.
    #
    # Enter key handler
    #
    def ProcessEnterEvent(self, event):
        # If autocompletion has been triggered, complete and do not process event
        if self.SCIAutoCActive():
            self.SCIAutoCComplete()
            self.SCICancel()
            return

        self.SCICancel()
        # First, check for an error message
        haveGrabbedOutput = 0
        if self.HandleSpecialLine():
            return 0

        lineNo = self.LineFromChar()
        start, end, isCode = self.GetBlockBoundary(lineNo)
        # If we are not in a code block just go to the prompt (or create a new one)
        if not isCode:
            self.AppendToPrompt([])
            win32ui.SetStatusText(win32ui.LoadString(afxres.AFX_IDS_IDLEMESSAGE))
            return

        lines = self.ExtractCommand((start, end))

        # If we are in a code-block, but it isnt at the end of the buffer
        # then copy it to the end ready for editing and subsequent execution
        if end != self.GetLineCount() - 1:
            win32ui.SetStatusText("Press ENTER to execute command")
            self.AppendToPrompt(lines)
            self.SetSel(-2)
            return

        # If SHIFT held down, we want new code here and now!
        bNeedIndent = (
            win32api.GetKeyState(win32con.VK_SHIFT) < 0
            or win32api.GetKeyState(win32con.VK_CONTROL) < 0
        )
        if bNeedIndent:
            self.ReplaceSel("\n")
        else:
            self.SetSel(-2)
            self.ReplaceSel("\n")
            source = "\n".join(lines)
            while source and source[-1] in "\t ":
                source = source[:-1]
            self.OutputGrab()  # grab the output for the command exec.
            try:
                if self.interp.runsource(
                    source, "<interactive input>"
                ):  # Need more input!
                    bNeedIndent = 1
                else:
                    # If the last line isnt empty, append a newline
                    if self.history is not None:
                        self.history.history_store(source)
                    self.AppendToPrompt([])
                    win32ui.SetStatusText(
                        win32ui.LoadString(afxres.AFX_IDS_IDLEMESSAGE)
                    )
            # 					win32ui.SetStatusText('Successfully executed statement')
            finally:
                self.OutputRelease()
        if bNeedIndent:
            win32ui.SetStatusText("Ready to continue the command")
            # Now attempt correct indentation (should use IDLE?)
            curLine = self.DoGetLine(lineNo)[len(sys.ps2) :]
            pos = 0
            indent = ""
            while len(curLine) > pos and curLine[pos] in string.whitespace:
                indent = indent + curLine[pos]
                pos = pos + 1
            if _is_block_opener(curLine):
                indent = indent + "\t"
            elif _is_block_closer(curLine):
                indent = indent[:-1]
            # use ReplaceSel to ensure it goes at the cursor rather than end of buffer.
            self.ReplaceSel(sys.ps2 + indent)
        return 0

    # ESC key handler
    def ProcessEscEvent(self, event):
        # Implement a cancel.
        if self.SCIAutoCActive() or self.SCICallTipActive():
            self.SCICancel()
        else:
            win32ui.SetStatusText("Cancelled.")
            self.AppendToPrompt(("",))
        return 0

    def OnSelectBlock(self, command, code):
        lineNo = self.LineFromChar()
        start, end, isCode = self.GetBlockBoundary(lineNo)
        startIndex = self.LineIndex(start)
        endIndex = self.LineIndex(end + 1) - 2  # skip \r + \n
        if endIndex < 0:  # must be beyond end of buffer
            endIndex = -2  # self.Length()
        self.SetSel(startIndex, endIndex)

    def OnEditCopyCode(self, command, code):
        """Sanitizes code from interactive window, removing prompts and output,
        and inserts it in the clipboard."""
        code = self.GetSelText()
        lines = code.splitlines()
        out_lines = []
        for line in lines:
            if line.startswith(sys.ps1):
                line = line[len(sys.ps1) :]
                out_lines.append(line)
            elif line.startswith(sys.ps2):
                line = line[len(sys.ps2) :]
                out_lines.append(line)
        out_code = os.linesep.join(out_lines)
        win32clipboard.OpenClipboard()
        try:
            win32clipboard.SetClipboardData(
                win32clipboard.CF_UNICODETEXT, str(out_code)
            )
        finally:
            win32clipboard.CloseClipboard()

    def OnEditExecClipboard(self, command, code):
        """Executes python code directly from the clipboard."""
        win32clipboard.OpenClipboard()
        try:
            code = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
        finally:
            win32clipboard.CloseClipboard()

        code = code.replace("\r\n", "\n") + "\n"
        try:
            o = compile(code, "<clipboard>", "exec")
            exec(o, __main__.__dict__)
        except:
            traceback.print_exc()

    def GetRightMenuItems(self):
        # Just override parents
        ret = []
        flags = 0
        ret.append((flags, win32ui.ID_EDIT_UNDO, "&Undo"))
        ret.append(win32con.MF_SEPARATOR)
        ret.append((flags, win32ui.ID_EDIT_CUT, "Cu&t"))
        ret.append((flags, win32ui.ID_EDIT_COPY, "&Copy"))

        start, end = self.GetSel()
        if start != end:
            ret.append((flags, ID_EDIT_COPY_CODE, "Copy code without prompts"))
        if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
            ret.append(
                (flags, ID_EDIT_EXEC_CLIPBOARD, "Execute python code from clipboard")
            )

        ret.append((flags, win32ui.ID_EDIT_PASTE, "&Paste"))
        ret.append(win32con.MF_SEPARATOR)
        ret.append((flags, win32ui.ID_EDIT_SELECT_ALL, "&Select all"))
        ret.append((flags, win32ui.ID_EDIT_SELECT_BLOCK, "Select &block"))
        ret.append((flags, win32ui.ID_VIEW_WHITESPACE, "View &Whitespace"))
        return ret

    def MDINextEvent(self, event):
        win32ui.GetMainFrame().MDINext(0)

    def MDIPrevEvent(self, event):
        win32ui.GetMainFrame().MDINext(0)

    def WindowBackEvent(self, event):
        parent = self.GetParentFrame()
        if parent == win32ui.GetMainFrame():
            # It is docked.
            try:
                wnd, isactive = parent.MDIGetActive()
                wnd.SetFocus()
            except win32ui.error:
                # No MDI window active!
                pass
        else:
            # Normal Window
            try:
                lastActive = self.GetParentFrame().lastActive
                # If the window is invalid, reset it.
                if lastActive is not None and (
                    lastActive._obj_ is None or lastActive.GetSafeHwnd() == 0
                ):
                    lastActive = self.GetParentFrame().lastActive = None
                    win32ui.SetStatusText("The last active Window has been closed.")
            except AttributeError:
                print("Can't find the last active window!")
                lastActive = None
            if lastActive is not None:
                lastActive.MDIActivate()


class InteractiveView(InteractiveCore, winout.WindowOutputView):
    def __init__(self, doc):
        InteractiveCore.__init__(self)
        winout.WindowOutputView.__init__(self, doc)
        self.encoding = pywin.default_scintilla_encoding

    def _MakeColorizer(self):
        return InteractiveFormatter(self)

    def OnInitialUpdate(self):
        winout.WindowOutputView.OnInitialUpdate(self)
        self.SetWordWrap()
        self.Init()

    def HookHandlers(self):
        winout.WindowOutputView.HookHandlers(self)
        InteractiveCore.HookHandlers(self)


class CInteractivePython(winout.WindowOutput):
    def __init__(self, makeDoc=None, makeFrame=None):
        self.IsFinalDestroy = 0
        winout.WindowOutput.__init__(
            self,
            sectionProfile,
            sectionProfile,
            winout.flags.WQ_LINE,
            1,
            None,
            makeDoc,
            makeFrame,
            InteractiveView,
        )
        self.Create()

    def OnViewDestroy(self, view):
        if self.IsFinalDestroy:
            view.OutputRelease()
        winout.WindowOutput.OnViewDestroy(self, view)

    def Close(self):
        self.IsFinalDestroy = 1
        winout.WindowOutput.Close(self)


class InteractiveFrame(winout.WindowOutputFrame):
    def __init__(self):
        self.lastActive = None
        winout.WindowOutputFrame.__init__(self)

    def OnMDIActivate(self, bActive, wndActive, wndDeactive):
        if bActive:
            self.lastActive = wndDeactive


######################################################################
##
## Dockable Window Support
##
######################################################################
ID_DOCKED_INTERACTIVE_CONTROLBAR = 0xE802

DockedInteractiveViewParent = InteractiveView


class DockedInteractiveView(DockedInteractiveViewParent):
    def HookHandlers(self):
        DockedInteractiveViewParent.HookHandlers(self)
        self.HookMessage(self.OnSetFocus, win32con.WM_SETFOCUS)
        self.HookMessage(self.OnKillFocus, win32con.WM_KILLFOCUS)

    def OnSetFocus(self, msg):
        self.GetParentFrame().SetActiveView(self)
        return 1

    def OnKillFocus(self, msg):
        # If we are losing focus to another in this app, reset the main frame's active view.
        hwnd = wparam = msg[2]
        try:
            wnd = win32ui.CreateWindowFromHandle(hwnd)
            reset = wnd.GetTopLevelFrame() == self.GetTopLevelFrame()
        except win32ui.error:
            reset = 0  # Not my window
        if reset:
            self.GetParentFrame().SetActiveView(None)
        return 1

    def OnDestroy(self, msg):
        newSize = self.GetWindowPlacement()[4]
        pywin.framework.app.SaveWindowSize("Interactive Window", newSize, "docked")
        try:
            if self.GetParentFrame().GetActiveView == self:
                self.GetParentFrame().SetActiveView(None)
        except win32ui.error:
            pass
        try:
            if win32ui.GetMainFrame().GetActiveView() == self:
                win32ui.GetMainFrame().SetActiveView(None)
        except win32ui.error:
            pass
        return DockedInteractiveViewParent.OnDestroy(self, msg)


class CDockedInteractivePython(CInteractivePython):
    def __init__(self, dockbar):
        self.bFirstCreated = 0
        self.dockbar = dockbar
        CInteractivePython.__init__(self)

    def NeedRecreateWindow(self):
        if self.bCreating:
            return 0
        try:
            frame = win32ui.GetMainFrame()
            if frame.closing:
                return 0  # Dieing!
        except (win32ui.error, AttributeError):
            return 0  # The app is dieing!
        try:
            cb = frame.GetControlBar(ID_DOCKED_INTERACTIVE_CONTROLBAR)
            return not cb.IsWindowVisible()
        except win32ui.error:
            return 1  # Control bar does not exist!

    def RecreateWindow(self):
        try:
            dockbar = win32ui.GetMainFrame().GetControlBar(
                ID_DOCKED_INTERACTIVE_CONTROLBAR
            )
            win32ui.GetMainFrame().ShowControlBar(dockbar, 1, 1)
        except win32ui.error:
            CreateDockedInteractiveWindow()

    def Create(self):
        self.bCreating = 1
        doc = InteractiveDocument(None, self.DoCreateDoc())
        view = DockedInteractiveView(doc)
        defRect = pywin.framework.app.LoadWindowSize("Interactive Window", "docked")
        if defRect[2] - defRect[0] == 0:
            defRect = 0, 0, 500, 200
        style = win32con.WS_CHILD | win32con.WS_VISIBLE | win32con.WS_BORDER
        id = 1050  # win32ui.AFX_IDW_PANE_FIRST
        view.CreateWindow(self.dockbar, id, style, defRect)
        view.OnInitialUpdate()
        self.bFirstCreated = 1

        self.currentView = doc.GetFirstView()
        self.bCreating = 0
        if self.title:
            doc.SetTitle(self.title)


# The factory we pass to the dockable window support.
def InteractiveViewCreator(parent):
    global edit
    edit = CDockedInteractivePython(parent)
    return edit.currentView


def CreateDockedInteractiveWindow():
    # Later, the DockingBar should be capable of hosting multiple
    # children.
    from pywin.docking.DockingBar import DockingBar

    bar = DockingBar()
    creator = InteractiveViewCreator
    bar.CreateWindow(
        win32ui.GetMainFrame(),
        creator,
        "Interactive Window",
        ID_DOCKED_INTERACTIVE_CONTROLBAR,
    )
    bar.SetBarStyle(
        bar.GetBarStyle()
        | afxres.CBRS_TOOLTIPS
        | afxres.CBRS_FLYBY
        | afxres.CBRS_SIZE_DYNAMIC
    )
    bar.EnableDocking(afxres.CBRS_ALIGN_ANY)
    win32ui.GetMainFrame().DockControlBar(bar, afxres.AFX_IDW_DOCKBAR_BOTTOM)


######################################################################
#
# The public interface to this module.
#
######################################################################
# No extra functionality now, but maybe later, so
# publicize these names.
InteractiveDocument = winout.WindowOutputDocument

# We remember our one and only interactive window in the "edit" variable.
edit = None


def CreateInteractiveWindowUserPreference(makeDoc=None, makeFrame=None):
    """Create some sort of interactive window if the user's preference say we should."""
    bCreate = LoadPreference("Show at startup", 1)
    if bCreate:
        CreateInteractiveWindow(makeDoc, makeFrame)


def CreateInteractiveWindow(makeDoc=None, makeFrame=None):
    """Create a standard or docked interactive window unconditionally"""
    assert edit is None, "Creating second interactive window!"
    bDocking = LoadPreference("Docking", 0)
    if bDocking:
        CreateDockedInteractiveWindow()
    else:
        CreateMDIInteractiveWindow(makeDoc, makeFrame)
    assert edit is not None, "Created interactive window, but did not set the global!"
    edit.currentView.SetFocus()


def CreateMDIInteractiveWindow(makeDoc=None, makeFrame=None):
    """Create a standard (non-docked) interactive window unconditionally"""
    global edit
    if makeDoc is None:
        makeDoc = InteractiveDocument
    if makeFrame is None:
        makeFrame = InteractiveFrame
    edit = CInteractivePython(makeDoc=makeDoc, makeFrame=makeFrame)


def DestroyInteractiveWindow():
    """Destroy the interactive window.
    This is different to Closing the window,
    which may automatically re-appear.  Once destroyed, it can never be recreated,
    and a complete new instance must be created (which the various other helper
    functions will then do after making this call
    """
    global edit
    if edit is not None and edit.currentView is not None:
        if edit.currentView.GetParentFrame() == win32ui.GetMainFrame():
            # It is docked - do nothing now (this is only called at shutdown!)
            pass
        else:
            # It is a standard window - call Close on the container.
            edit.Close()
            edit = None


def CloseInteractiveWindow():
    """Close the interactive window, allowing it to be re-created on demand."""
    global edit
    if edit is not None and edit.currentView is not None:
        if edit.currentView.GetParentFrame() == win32ui.GetMainFrame():
            # It is docked, just hide the dock bar.
            frame = win32ui.GetMainFrame()
            cb = frame.GetControlBar(ID_DOCKED_INTERACTIVE_CONTROLBAR)
            frame.ShowControlBar(cb, 0, 1)
        else:
            # It is a standard window - destroy the frame/view, allowing the object itself to remain.
            edit.currentView.GetParentFrame().DestroyWindow()


def ToggleInteractiveWindow():
    """If the interactive window is visible, hide it, otherwise show it."""
    if edit is None:
        CreateInteractiveWindow()
    else:
        if edit.NeedRecreateWindow():
            edit.RecreateWindow()
        else:
            # Close it, allowing a reopen.
            CloseInteractiveWindow()


def ShowInteractiveWindow():
    """Shows (or creates if necessary) an interactive window"""
    if edit is None:
        CreateInteractiveWindow()
    else:
        if edit.NeedRecreateWindow():
            edit.RecreateWindow()
        else:
            parent = edit.currentView.GetParentFrame()
            if parent == win32ui.GetMainFrame():  # It is docked.
                edit.currentView.SetFocus()
            else:  # It is a "normal" window
                edit.currentView.GetParentFrame().AutoRestore()
                win32ui.GetMainFrame().MDIActivate(edit.currentView.GetParentFrame())


def IsInteractiveWindowVisible():
    return edit is not None and not edit.NeedRecreateWindow()
