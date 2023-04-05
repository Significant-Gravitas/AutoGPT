# An Python interface to the Scintilla control.
#
# Exposes Python classes that allow you to use Scintilla as
# a "standard" MFC edit control (eg, control.GetTextLength(), control.GetSel()
# plus many Scintilla specific features (eg control.SCIAddStyledText())

import array
import os
import struct

import win32api
import win32con
import win32ui
from pywin import default_scintilla_encoding
from pywin.mfc import window

from . import scintillacon

# Load Scintilla.dll to get access to the control.
# We expect to find this in the same directory as win32ui.pyd
dllid = None
if win32ui.debug:  # If running _d version of Pythonwin...
    try:
        dllid = win32api.LoadLibrary(
            os.path.join(os.path.split(win32ui.__file__)[0], "Scintilla_d.DLL")
        )
    except (
        win32api.error
    ):  # Not there - we dont _need_ a debug ver, so ignore this error.
        pass
if dllid is None:
    try:
        dllid = win32api.LoadLibrary(
            os.path.join(os.path.split(win32ui.__file__)[0], "Scintilla.DLL")
        )
    except win32api.error:
        pass
if dllid is None:
    # Still not there - lets see if Windows can find it by searching?
    dllid = win32api.LoadLibrary("Scintilla.DLL")

# null_byte is str in py2k, bytes on py3k
null_byte = "\0".encode("ascii")

## These are from Richedit.h - need to add to win32con or commctrl
EM_GETTEXTRANGE = 1099
EM_EXLINEFROMCHAR = 1078
EM_FINDTEXTEX = 1103
EM_GETSELTEXT = 1086
EM_EXSETSEL = win32con.WM_USER + 55


class ScintillaNotification:
    def __init__(self, **args):
        self.__dict__.update(args)


class ScintillaControlInterface:
    def SCIUnpackNotifyMessage(self, msg):
        format = "iiiiPiiiPPiiii"
        bytes = win32ui.GetBytes(msg, struct.calcsize(format))
        (
            position,
            ch,
            modifiers,
            modificationType,
            text_ptr,
            length,
            linesAdded,
            msg,
            wParam,
            lParam,
            line,
            foldLevelNow,
            foldLevelPrev,
            margin,
        ) = struct.unpack(format, bytes)
        return ScintillaNotification(
            position=position,
            ch=ch,
            modifiers=modifiers,
            modificationType=modificationType,
            text_ptr=text_ptr,
            length=length,
            linesAdded=linesAdded,
            msg=msg,
            wParam=wParam,
            lParam=lParam,
            line=line,
            foldLevelNow=foldLevelNow,
            foldLevelPrev=foldLevelPrev,
            margin=margin,
        )

    def SCIAddText(self, text):
        self.SendMessage(
            scintillacon.SCI_ADDTEXT, text.encode(default_scintilla_encoding)
        )

    def SCIAddStyledText(self, text, style=None):
        # If style is None, text is assumed to be a "native" Scintilla buffer.
        # If style is specified, text is a normal string, and the style is
        # assumed to apply to the entire string.
        if style is not None:
            text = list(map(lambda char, style=style: char + chr(style), text))
            text = "".join(text)
        self.SendMessage(
            scintillacon.SCI_ADDSTYLEDTEXT, text.encode(default_scintilla_encoding)
        )

    def SCIInsertText(self, text, pos=-1):
        # SCIInsertText allows unicode or bytes - but if they are bytes,
        # the caller must ensure it is encoded correctly.
        if isinstance(text, str):
            text = text.encode(default_scintilla_encoding)
        self.SendScintilla(scintillacon.SCI_INSERTTEXT, pos, text + null_byte)

    def SCISetSavePoint(self):
        self.SendScintilla(scintillacon.SCI_SETSAVEPOINT)

    def SCISetUndoCollection(self, collectFlag):
        self.SendScintilla(scintillacon.SCI_SETUNDOCOLLECTION, collectFlag)

    def SCIBeginUndoAction(self):
        self.SendScintilla(scintillacon.SCI_BEGINUNDOACTION)

    def SCIEndUndoAction(self):
        self.SendScintilla(scintillacon.SCI_ENDUNDOACTION)

    def SCIGetCurrentPos(self):
        return self.SendScintilla(scintillacon.SCI_GETCURRENTPOS)

    def SCIGetCharAt(self, pos):
        # Must ensure char is unsigned!
        return chr(self.SendScintilla(scintillacon.SCI_GETCHARAT, pos) & 0xFF)

    def SCIGotoLine(self, line):
        self.SendScintilla(scintillacon.SCI_GOTOLINE, line)

    def SCIBraceMatch(self, pos, maxReStyle):
        return self.SendScintilla(scintillacon.SCI_BRACEMATCH, pos, maxReStyle)

    def SCIBraceHighlight(self, pos, posOpposite):
        return self.SendScintilla(scintillacon.SCI_BRACEHIGHLIGHT, pos, posOpposite)

    def SCIBraceBadHighlight(self, pos):
        return self.SendScintilla(scintillacon.SCI_BRACEBADLIGHT, pos)

    ####################################
    # Styling
    # 	def SCIColourise(self, start=0, end=-1):
    #   NOTE - dependent on of we use builtin lexer, so handled below.
    def SCIGetEndStyled(self):
        return self.SendScintilla(scintillacon.SCI_GETENDSTYLED)

    def SCIStyleSetFore(self, num, v):
        return self.SendScintilla(scintillacon.SCI_STYLESETFORE, num, v)

    def SCIStyleSetBack(self, num, v):
        return self.SendScintilla(scintillacon.SCI_STYLESETBACK, num, v)

    def SCIStyleSetEOLFilled(self, num, v):
        return self.SendScintilla(scintillacon.SCI_STYLESETEOLFILLED, num, v)

    def SCIStyleSetFont(self, num, name, characterset=0):
        buff = (name + "\0").encode(default_scintilla_encoding)
        self.SendScintilla(scintillacon.SCI_STYLESETFONT, num, buff)
        self.SendScintilla(scintillacon.SCI_STYLESETCHARACTERSET, num, characterset)

    def SCIStyleSetBold(self, num, bBold):
        self.SendScintilla(scintillacon.SCI_STYLESETBOLD, num, bBold)

    def SCIStyleSetItalic(self, num, bItalic):
        self.SendScintilla(scintillacon.SCI_STYLESETITALIC, num, bItalic)

    def SCIStyleSetSize(self, num, size):
        self.SendScintilla(scintillacon.SCI_STYLESETSIZE, num, size)

    def SCIGetViewWS(self):
        return self.SendScintilla(scintillacon.SCI_GETVIEWWS)

    def SCISetViewWS(self, val):
        self.SendScintilla(scintillacon.SCI_SETVIEWWS, not (val == 0))
        self.InvalidateRect()

    def SCISetIndentationGuides(self, val):
        self.SendScintilla(scintillacon.SCI_SETINDENTATIONGUIDES, val)

    def SCIGetIndentationGuides(self):
        return self.SendScintilla(scintillacon.SCI_GETINDENTATIONGUIDES)

    def SCISetIndent(self, val):
        self.SendScintilla(scintillacon.SCI_SETINDENT, val)

    def SCIGetIndent(self, val):
        return self.SendScintilla(scintillacon.SCI_GETINDENT)

    def SCIGetViewEOL(self):
        return self.SendScintilla(scintillacon.SCI_GETVIEWEOL)

    def SCISetViewEOL(self, val):
        self.SendScintilla(scintillacon.SCI_SETVIEWEOL, not (val == 0))
        self.InvalidateRect()

    def SCISetTabWidth(self, width):
        self.SendScintilla(scintillacon.SCI_SETTABWIDTH, width, 0)

    def SCIStartStyling(self, pos, mask):
        self.SendScintilla(scintillacon.SCI_STARTSTYLING, pos, mask)

    def SCISetStyling(self, pos, attr):
        self.SendScintilla(scintillacon.SCI_SETSTYLING, pos, attr)

    def SCISetStylingEx(self, ray):  # ray is an array.
        address, length = ray.buffer_info()
        self.SendScintilla(scintillacon.SCI_SETSTYLINGEX, length, address)

    def SCIGetStyleAt(self, pos):
        return self.SendScintilla(scintillacon.SCI_GETSTYLEAT, pos)

    def SCISetMarginWidth(self, width):
        self.SendScintilla(scintillacon.SCI_SETMARGINWIDTHN, 1, width)

    def SCISetMarginWidthN(self, n, width):
        self.SendScintilla(scintillacon.SCI_SETMARGINWIDTHN, n, width)

    def SCISetFoldFlags(self, flags):
        self.SendScintilla(scintillacon.SCI_SETFOLDFLAGS, flags)

    # Markers
    def SCIMarkerDefineAll(self, markerNum, markerType, fore, back):
        self.SCIMarkerDefine(markerNum, markerType)
        self.SCIMarkerSetFore(markerNum, fore)
        self.SCIMarkerSetBack(markerNum, back)

    def SCIMarkerDefine(self, markerNum, markerType):
        self.SendScintilla(scintillacon.SCI_MARKERDEFINE, markerNum, markerType)

    def SCIMarkerSetFore(self, markerNum, fore):
        self.SendScintilla(scintillacon.SCI_MARKERSETFORE, markerNum, fore)

    def SCIMarkerSetBack(self, markerNum, back):
        self.SendScintilla(scintillacon.SCI_MARKERSETBACK, markerNum, back)

    def SCIMarkerAdd(self, lineNo, markerNum):
        self.SendScintilla(scintillacon.SCI_MARKERADD, lineNo, markerNum)

    def SCIMarkerDelete(self, lineNo, markerNum):
        self.SendScintilla(scintillacon.SCI_MARKERDELETE, lineNo, markerNum)

    def SCIMarkerDeleteAll(self, markerNum=-1):
        self.SendScintilla(scintillacon.SCI_MARKERDELETEALL, markerNum)

    def SCIMarkerGet(self, lineNo):
        return self.SendScintilla(scintillacon.SCI_MARKERGET, lineNo)

    def SCIMarkerNext(self, lineNo, markerNum):
        return self.SendScintilla(scintillacon.SCI_MARKERNEXT, lineNo, markerNum)

    def SCICancel(self):
        self.SendScintilla(scintillacon.SCI_CANCEL)

    # AutoComplete
    def SCIAutoCShow(self, text):
        if type(text) in [type([]), type(())]:
            text = " ".join(text)
        buff = (text + "\0").encode(default_scintilla_encoding)
        return self.SendScintilla(scintillacon.SCI_AUTOCSHOW, 0, buff)

    def SCIAutoCCancel(self):
        self.SendScintilla(scintillacon.SCI_AUTOCCANCEL)

    def SCIAutoCActive(self):
        return self.SendScintilla(scintillacon.SCI_AUTOCACTIVE)

    def SCIAutoCComplete(self):
        return self.SendScintilla(scintillacon.SCI_AUTOCCOMPLETE)

    def SCIAutoCStops(self, stops):
        buff = (stops + "\0").encode(default_scintilla_encoding)
        self.SendScintilla(scintillacon.SCI_AUTOCSTOPS, 0, buff)

    def SCIAutoCSetAutoHide(self, hide):
        self.SendScintilla(scintillacon.SCI_AUTOCSETAUTOHIDE, hide)

    def SCIAutoCSetFillups(self, fillups):
        self.SendScintilla(scintillacon.SCI_AUTOCSETFILLUPS, fillups)

    # Call tips
    def SCICallTipShow(self, text, pos=-1):
        if pos == -1:
            pos = self.GetSel()[0]
        buff = (text + "\0").encode(default_scintilla_encoding)
        self.SendScintilla(scintillacon.SCI_CALLTIPSHOW, pos, buff)

    def SCICallTipCancel(self):
        self.SendScintilla(scintillacon.SCI_CALLTIPCANCEL)

    def SCICallTipActive(self):
        return self.SendScintilla(scintillacon.SCI_CALLTIPACTIVE)

    def SCICallTipPosStart(self):
        return self.SendScintilla(scintillacon.SCI_CALLTIPPOSSTART)

    def SCINewline(self):
        self.SendScintilla(scintillacon.SCI_NEWLINE)

    # Lexer etc
    def SCISetKeywords(self, keywords, kw_list_no=0):
        buff = (keywords + "\0").encode(default_scintilla_encoding)
        self.SendScintilla(scintillacon.SCI_SETKEYWORDS, kw_list_no, buff)

    def SCISetProperty(self, name, value):
        name_buff = array.array("b", (name + "\0").encode(default_scintilla_encoding))
        val_buff = array.array(
            "b", (str(value) + "\0").encode(default_scintilla_encoding)
        )
        address_name_buffer = name_buff.buffer_info()[0]
        address_val_buffer = val_buff.buffer_info()[0]
        self.SendScintilla(
            scintillacon.SCI_SETPROPERTY, address_name_buffer, address_val_buffer
        )

    def SCISetStyleBits(self, nbits):
        self.SendScintilla(scintillacon.SCI_SETSTYLEBITS, nbits)

    # Folding
    def SCIGetFoldLevel(self, lineno):
        return self.SendScintilla(scintillacon.SCI_GETFOLDLEVEL, lineno)

    def SCIToggleFold(self, lineno):
        return self.SendScintilla(scintillacon.SCI_TOGGLEFOLD, lineno)

    def SCIEnsureVisible(self, lineno):
        self.SendScintilla(scintillacon.SCI_ENSUREVISIBLE, lineno)

    def SCIGetFoldExpanded(self, lineno):
        return self.SendScintilla(scintillacon.SCI_GETFOLDEXPANDED, lineno)

    # right edge
    def SCISetEdgeColumn(self, edge):
        self.SendScintilla(scintillacon.SCI_SETEDGECOLUMN, edge)

    def SCIGetEdgeColumn(self):
        return self.SendScintilla(scintillacon.SCI_GETEDGECOLUMN)

    def SCISetEdgeMode(self, mode):
        self.SendScintilla(scintillacon.SCI_SETEDGEMODE, mode)

    def SCIGetEdgeMode(self):
        return self.SendScintilla(scintillacon.SCI_GETEDGEMODE)

    def SCISetEdgeColor(self, color):
        self.SendScintilla(scintillacon.SCI_SETEDGECOLOUR, color)

    def SCIGetEdgeColor(self):
        return self.SendScintilla(scintillacon.SCI_GETEDGECOLOR)

    # Multi-doc
    def SCIGetDocPointer(self):
        return self.SendScintilla(scintillacon.SCI_GETDOCPOINTER)

    def SCISetDocPointer(self, p):
        return self.SendScintilla(scintillacon.SCI_SETDOCPOINTER, 0, p)

    def SCISetWrapMode(self, mode):
        return self.SendScintilla(scintillacon.SCI_SETWRAPMODE, mode)

    def SCIGetWrapMode(self):
        return self.SendScintilla(scintillacon.SCI_GETWRAPMODE)


class CScintillaEditInterface(ScintillaControlInterface):
    def close(self):
        self.colorizer = None

    def Clear(self):
        self.SendScintilla(win32con.WM_CLEAR)

    def FindText(self, flags, range, findText):
        """LPARAM for EM_FINDTEXTEX:
                typedef struct _findtextex {
                CHARRANGE chrg;
                LPCTSTR lpstrText;
                CHARRANGE chrgText;} FINDTEXTEX;
        typedef struct _charrange {
                LONG cpMin;
                LONG cpMax;} CHARRANGE;
        """
        findtextex_fmt = "llPll"
        ## Scintilla does not handle unicode in EM_FINDTEXT msg (FINDTEXTEX struct)
        txt_buff = (findText + "\0").encode(default_scintilla_encoding)
        txt_array = array.array("b", txt_buff)
        ft_buff = struct.pack(
            findtextex_fmt, range[0], range[1], txt_array.buffer_info()[0], 0, 0
        )
        ft_array = array.array("b", ft_buff)
        rc = self.SendScintilla(EM_FINDTEXTEX, flags, ft_array.buffer_info()[0])
        ftUnpacked = struct.unpack(findtextex_fmt, ft_array)
        return rc, (ftUnpacked[3], ftUnpacked[4])

    def GetSel(self):
        currentPos = self.SendScintilla(scintillacon.SCI_GETCURRENTPOS)
        anchorPos = self.SendScintilla(scintillacon.SCI_GETANCHOR)
        if currentPos < anchorPos:
            return (currentPos, anchorPos)
        else:
            return (anchorPos, currentPos)
        return currentPos

    def GetSelText(self):
        start, end = self.GetSel()
        txtBuf = array.array("b", null_byte * (end - start + 1))
        addressTxtBuf = txtBuf.buffer_info()[0]
        # EM_GETSELTEXT is documented as returning the number of chars
        # not including the NULL, but scintilla includes the NULL.  A
        # quick glance at the scintilla impl doesn't make this
        # obvious - the NULL is included in the 'selection' object
        # and reflected in the length of that 'selection' object.
        # I expect that is a bug in scintilla and may be fixed by now,
        # but we just blindly assume that the last char is \0 and
        # strip it.
        self.SendScintilla(EM_GETSELTEXT, 0, addressTxtBuf)
        return txtBuf.tobytes()[:-1].decode(default_scintilla_encoding)

    def SetSel(self, start=0, end=None):
        if type(start) == type(()):
            assert (
                end is None
            ), "If you pass a point in the first param, the second must be None"
            start, end = start
        elif end is None:
            end = start
        if start < 0:
            start = self.GetTextLength()
        if end < 0:
            end = self.GetTextLength()
        assert start <= self.GetTextLength(), "The start postion is invalid (%d/%d)" % (
            start,
            self.GetTextLength(),
        )
        assert end <= self.GetTextLength(), "The end postion is invalid (%d/%d)" % (
            end,
            self.GetTextLength(),
        )
        cr = struct.pack("ll", start, end)
        crBuff = array.array("b", cr)
        addressCrBuff = crBuff.buffer_info()[0]
        rc = self.SendScintilla(EM_EXSETSEL, 0, addressCrBuff)

    def GetLineCount(self):
        return self.SendScintilla(win32con.EM_GETLINECOUNT)

    def LineFromChar(self, charPos=-1):
        if charPos == -1:
            charPos = self.GetSel()[0]
        assert (
            charPos >= 0 and charPos <= self.GetTextLength()
        ), "The charPos postion (%s) is invalid (max=%s)" % (
            charPos,
            self.GetTextLength(),
        )
        # return self.SendScintilla(EM_EXLINEFROMCHAR, charPos)
        # EM_EXLINEFROMCHAR puts charPos in lParam, not wParam
        return self.SendScintilla(EM_EXLINEFROMCHAR, 0, charPos)

    def LineIndex(self, line):
        return self.SendScintilla(win32con.EM_LINEINDEX, line)

    def ScrollCaret(self):
        return self.SendScintilla(win32con.EM_SCROLLCARET)

    def GetCurLineNumber(self):
        return self.LineFromChar(self.SCIGetCurrentPos())

    def GetTextLength(self):
        return self.SendScintilla(scintillacon.SCI_GETTEXTLENGTH)

    def GetTextRange(self, start=0, end=-1, decode=True):
        if end == -1:
            end = self.SendScintilla(scintillacon.SCI_GETTEXTLENGTH)
        assert end >= start, "Negative index requested (%d/%d)" % (start, end)
        assert (
            start >= 0 and start <= self.GetTextLength()
        ), "The start postion is invalid"
        assert end >= 0 and end <= self.GetTextLength(), "The end postion is invalid"
        initer = null_byte * (end - start + 1)
        buff = array.array("b", initer)
        addressBuffer = buff.buffer_info()[0]
        tr = struct.pack("llP", start, end, addressBuffer)
        trBuff = array.array("b", tr)
        addressTrBuff = trBuff.buffer_info()[0]
        num_bytes = self.SendScintilla(EM_GETTEXTRANGE, 0, addressTrBuff)
        ret = buff.tobytes()[:num_bytes]
        if decode:
            ret = ret.decode(default_scintilla_encoding)
        return ret

    def ReplaceSel(self, str):
        buff = (str + "\0").encode(default_scintilla_encoding)
        self.SendScintilla(scintillacon.SCI_REPLACESEL, 0, buff)

    def GetLine(self, line=-1):
        if line == -1:
            line = self.GetCurLineNumber()
        start = self.LineIndex(line)
        end = self.LineIndex(line + 1)
        return self.GetTextRange(start, end)

    def SetReadOnly(self, flag=1):
        return self.SendScintilla(win32con.EM_SETREADONLY, flag)

    def LineScroll(self, lines, cols=0):
        return self.SendScintilla(win32con.EM_LINESCROLL, cols, lines)

    def GetFirstVisibleLine(self):
        return self.SendScintilla(win32con.EM_GETFIRSTVISIBLELINE)

    def SetWordWrap(self, mode):
        if mode != win32ui.CRichEditView_WrapNone:
            raise ValueError("We dont support word-wrap (I dont think :-)")


class CScintillaColorEditInterface(CScintillaEditInterface):
    ################################
    # Plug-in colorizer support
    def _GetColorizer(self):
        if not hasattr(self, "colorizer"):
            self.colorizer = self._MakeColorizer()
        return self.colorizer

    def _MakeColorizer(self):
        # Give parent a chance to hook.
        parent_func = getattr(self.GetParentFrame(), "_MakeColorizer", None)
        if parent_func is not None:
            return parent_func()
        from . import formatter

        ##		return formatter.PythonSourceFormatter(self)
        return formatter.BuiltinPythonSourceFormatter(self)

    def Colorize(self, start=0, end=-1):
        c = self._GetColorizer()
        if c is not None:
            c.Colorize(start, end)

    def ApplyFormattingStyles(self, bReload=1):
        c = self._GetColorizer()
        if c is not None:
            c.ApplyFormattingStyles(bReload)

    # The Parent window will normally hook
    def HookFormatter(self, parent=None):
        c = self._GetColorizer()
        if c is not None:  # No need if we have no color!
            c.HookFormatter(parent)


class CScintillaEdit(window.Wnd, CScintillaColorEditInterface):
    def __init__(self, wnd=None):
        if wnd is None:
            wnd = win32ui.CreateWnd()
        window.Wnd.__init__(self, wnd)

    def SendScintilla(self, msg, w=0, l=0):
        return self.SendMessage(msg, w, l)

    def CreateWindow(self, style, rect, parent, id):
        self._obj_.CreateWindow("Scintilla", "Scintilla", style, rect, parent, id, None)
