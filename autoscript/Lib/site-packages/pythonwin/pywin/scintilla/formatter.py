# Does Python source formatting for Scintilla controls.
import array
import string

import win32api
import win32con
import win32ui

from . import scintillacon

WM_KICKIDLE = 0x036A

# Used to indicate that style should use default color
from win32con import CLR_INVALID

debugging = 0
if debugging:
    # Output must go to another process else the result of
    # the printing itself will trigger again trigger a trace.

    import win32trace
    import win32traceutil

    def trace(*args):
        win32trace.write(" ".join(map(str, args)) + "\n")

else:
    trace = lambda *args: None


class Style:
    """Represents a single format"""

    def __init__(self, name, format, background=CLR_INVALID):
        self.name = name  # Name the format representes eg, "String", "Class"
        # Default background for each style is only used when there are no
        # saved settings (generally on first startup)
        self.background = self.default_background = background
        if type(format) == type(""):
            self.aliased = format
            self.format = None
        else:
            self.format = format
            self.aliased = None
        self.stylenum = None  # Not yet registered.

    def IsBasedOnDefault(self):
        return len(self.format) == 5

    # If the currently extended font defintion matches the
    # default format, restore the format to the "simple" format.
    def NormalizeAgainstDefault(self, defaultFormat):
        if self.IsBasedOnDefault():
            return 0  # No more to do, and not changed.
        bIsDefault = (
            self.format[7] == defaultFormat[7] and self.format[2] == defaultFormat[2]
        )
        if bIsDefault:
            self.ForceAgainstDefault()
        return bIsDefault

    def ForceAgainstDefault(self):
        self.format = self.format[:5]

    def GetCompleteFormat(self, defaultFormat):
        # Get the complete style after applying any relevant defaults.
        if len(self.format) == 5:  # It is a default one
            fmt = self.format + defaultFormat[5:]
        else:
            fmt = self.format
        flags = (
            win32con.CFM_BOLD
            | win32con.CFM_CHARSET
            | win32con.CFM_COLOR
            | win32con.CFM_FACE
            | win32con.CFM_ITALIC
            | win32con.CFM_SIZE
        )
        return (flags,) + fmt[1:]


# The Formatter interface
# used primarily when the actual formatting is done by Scintilla!
class FormatterBase:
    def __init__(self, scintilla):
        self.scintilla = scintilla
        self.baseFormatFixed = (-402653169, 0, 200, 0, 0, 0, 49, "Courier New")
        self.baseFormatProp = (-402653169, 0, 200, 0, 0, 0, 49, "Arial")
        self.bUseFixed = 1
        self.styles = {}  # Indexed by name
        self.styles_by_id = {}  # Indexed by allocated ID.
        self.SetStyles()

    def HookFormatter(self, parent=None):
        raise NotImplementedError()

    # Used by the IDLE extensions to quickly determine if a character is a string.
    def GetStringStyle(self, pos):
        try:
            style = self.styles_by_id[self.scintilla.SCIGetStyleAt(pos)]
        except KeyError:
            # A style we dont know about - probably not even a .py file - can't be a string
            return None
        if style.name in self.string_style_names:
            return style
        return None

    def RegisterStyle(self, style, stylenum):
        assert stylenum is not None, "We must have a style number"
        assert style.stylenum is None, "Style has already been registered"
        assert stylenum not in self.styles, "We are reusing a style number!"
        style.stylenum = stylenum
        self.styles[style.name] = style
        self.styles_by_id[stylenum] = style

    def SetStyles(self):
        raise NotImplementedError()

    def GetSampleText(self):
        return "Sample Text for the Format Dialog"

    def GetDefaultFormat(self):
        if self.bUseFixed:
            return self.baseFormatFixed
        return self.baseFormatProp

    # Update the control with the new style format.
    def _ReformatStyle(self, style):
        ## Selection (background only for now)
        ## Passing False for WPARAM to SCI_SETSELBACK is documented as resetting to scintilla default,
        ## but does not work - selection background is not visible at all.
        ## Default value in SPECIAL_STYLES taken from scintilla source.
        if style.name == STYLE_SELECTION:
            clr = style.background
            self.scintilla.SendScintilla(scintillacon.SCI_SETSELBACK, True, clr)

            ## Can't change font for selection, but could set color
            ## However, the font color dropbox has no option for default, and thus would
            ## always override syntax coloring
            ## clr = style.format[4]
            ## self.scintilla.SendScintilla(scintillacon.SCI_SETSELFORE, clr != CLR_INVALID, clr)
            return

        assert style.stylenum is not None, "Unregistered style."
        # print "Reformat style", style.name, style.stylenum
        scintilla = self.scintilla
        stylenum = style.stylenum
        # Now we have the style number, indirect for the actual style.
        if style.aliased is not None:
            style = self.styles[style.aliased]
        f = style.format
        if style.IsBasedOnDefault():
            baseFormat = self.GetDefaultFormat()
        else:
            baseFormat = f
        scintilla.SCIStyleSetFore(stylenum, f[4])
        scintilla.SCIStyleSetFont(stylenum, baseFormat[7], baseFormat[5])
        if f[1] & 1:
            scintilla.SCIStyleSetBold(stylenum, 1)
        else:
            scintilla.SCIStyleSetBold(stylenum, 0)
        if f[1] & 2:
            scintilla.SCIStyleSetItalic(stylenum, 1)
        else:
            scintilla.SCIStyleSetItalic(stylenum, 0)
        scintilla.SCIStyleSetSize(stylenum, int(baseFormat[2] / 20))
        scintilla.SCIStyleSetEOLFilled(stylenum, 1)  # Only needed for unclosed strings.

        ## Default style background to whitespace background if set,
        ##	otherwise use system window color
        bg = style.background
        if bg == CLR_INVALID:
            bg = self.styles[STYLE_DEFAULT].background
        if bg == CLR_INVALID:
            bg = win32api.GetSysColor(win32con.COLOR_WINDOW)
        scintilla.SCIStyleSetBack(stylenum, bg)

    def GetStyleByNum(self, stylenum):
        return self.styles_by_id[stylenum]

    def ApplyFormattingStyles(self, bReload=1):
        if bReload:
            self.LoadPreferences()
        baseFormat = self.GetDefaultFormat()
        defaultStyle = Style("default", baseFormat)
        defaultStyle.stylenum = scintillacon.STYLE_DEFAULT
        self._ReformatStyle(defaultStyle)
        for style in list(self.styles.values()):
            if style.aliased is None:
                style.NormalizeAgainstDefault(baseFormat)
            self._ReformatStyle(style)
        self.scintilla.InvalidateRect()

    # Some functions for loading and saving preferences.  By default
    # an INI file (well, MFC maps this to the registry) is used.
    def LoadPreferences(self):
        self.baseFormatFixed = eval(
            self.LoadPreference("Base Format Fixed", str(self.baseFormatFixed))
        )
        self.baseFormatProp = eval(
            self.LoadPreference("Base Format Proportional", str(self.baseFormatProp))
        )
        self.bUseFixed = int(self.LoadPreference("Use Fixed", 1))

        for style in list(self.styles.values()):
            new = self.LoadPreference(style.name, str(style.format))
            try:
                style.format = eval(new)
            except:
                print("Error loading style data for", style.name)
            # Use "vanilla" background hardcoded in PYTHON_STYLES if no settings in registry
            style.background = int(
                self.LoadPreference(
                    style.name + " background", style.default_background
                )
            )

    def LoadPreference(self, name, default):
        return win32ui.GetProfileVal("Format", name, default)

    def SavePreferences(self):
        self.SavePreference("Base Format Fixed", str(self.baseFormatFixed))
        self.SavePreference("Base Format Proportional", str(self.baseFormatProp))
        self.SavePreference("Use Fixed", self.bUseFixed)
        for style in list(self.styles.values()):
            if style.aliased is None:
                self.SavePreference(style.name, str(style.format))
                bg_name = style.name + " background"
                self.SavePreference(bg_name, style.background)

    def SavePreference(self, name, value):
        win32ui.WriteProfileVal("Format", name, value)


# An abstract formatter
# For all formatters we actually implement here.
# (as opposed to those formatters built in to Scintilla)
class Formatter(FormatterBase):
    def __init__(self, scintilla):
        self.bCompleteWhileIdle = 0
        self.bHaveIdleHandler = 0  # Dont currently have an idle handle
        self.nextstylenum = 0
        FormatterBase.__init__(self, scintilla)

    def HookFormatter(self, parent=None):
        if parent is None:
            parent = self.scintilla.GetParent()  # was GetParentFrame()!?
        parent.HookNotify(self.OnStyleNeeded, scintillacon.SCN_STYLENEEDED)

    def OnStyleNeeded(self, std, extra):
        notify = self.scintilla.SCIUnpackNotifyMessage(extra)
        endStyledChar = self.scintilla.SendScintilla(scintillacon.SCI_GETENDSTYLED)
        lineEndStyled = self.scintilla.LineFromChar(endStyledChar)
        endStyled = self.scintilla.LineIndex(lineEndStyled)
        # print "enPosPaint %d endStyledChar %d lineEndStyled %d endStyled %d" % (endPosPaint, endStyledChar, lineEndStyled, endStyled)
        self.Colorize(endStyled, notify.position)

    def ColorSeg(self, start, end, styleName):
        end = end + 1
        # 		assert end-start>=0, "Can't have negative styling"
        stylenum = self.styles[styleName].stylenum
        while start < end:
            self.style_buffer[start] = stylenum
            start = start + 1
        # self.scintilla.SCISetStyling(end - start + 1, stylenum)

    def RegisterStyle(self, style, stylenum=None):
        if stylenum is None:
            stylenum = self.nextstylenum
            self.nextstylenum = self.nextstylenum + 1
        FormatterBase.RegisterStyle(self, style, stylenum)

    def ColorizeString(self, str, charStart, styleStart):
        raise RuntimeError("You must override this method")

    def Colorize(self, start=0, end=-1):
        scintilla = self.scintilla
        # scintilla's formatting is all done in terms of utf, so
        # we work with utf8 bytes instead of unicode.  This magically
        # works as any extended chars found in the utf8 don't change
        # the semantics.
        stringVal = scintilla.GetTextRange(start, end, decode=False)
        if start > 0:
            stylenum = scintilla.SCIGetStyleAt(start - 1)
            styleStart = self.GetStyleByNum(stylenum).name
        else:
            styleStart = None
        # 		trace("Coloring", start, end, end-start, len(stringVal), styleStart, self.scintilla.SCIGetCharAt(start))
        scintilla.SCIStartStyling(start, 31)
        self.style_buffer = array.array("b", (0,) * len(stringVal))
        self.ColorizeString(stringVal, styleStart)
        scintilla.SCISetStylingEx(self.style_buffer)
        self.style_buffer = None
        # 		trace("After styling, end styled is", self.scintilla.SCIGetEndStyled())
        if (
            self.bCompleteWhileIdle
            and not self.bHaveIdleHandler
            and end != -1
            and end < scintilla.GetTextLength()
        ):
            self.bHaveIdleHandler = 1
            win32ui.GetApp().AddIdleHandler(self.DoMoreColoring)
            # Kicking idle makes the app seem slower when initially repainting!

    # 			win32ui.GetMainFrame().PostMessage(WM_KICKIDLE, 0, 0)

    def DoMoreColoring(self, handler, count):
        try:
            scintilla = self.scintilla
            endStyled = scintilla.SCIGetEndStyled()
            lineStartStyled = scintilla.LineFromChar(endStyled)
            start = scintilla.LineIndex(lineStartStyled)
            end = scintilla.LineIndex(lineStartStyled + 1)
            textlen = scintilla.GetTextLength()
            if end < 0:
                end = textlen

            finished = end >= textlen
            self.Colorize(start, end)
        except (win32ui.error, AttributeError):
            # Window may have closed before we finished - no big deal!
            finished = 1

        if finished:
            self.bHaveIdleHandler = 0
            win32ui.GetApp().DeleteIdleHandler(handler)
        return not finished


# A Formatter that knows how to format Python source
from keyword import iskeyword, kwlist

wordstarts = "_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
wordchars = "._0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
operators = "%^&*()-+=|{}[]:;<>,/?!.~"

STYLE_DEFAULT = "Whitespace"
STYLE_COMMENT = "Comment"
STYLE_COMMENT_BLOCK = "Comment Blocks"
STYLE_NUMBER = "Number"
STYLE_STRING = "String"
STYLE_SQSTRING = "SQ String"
STYLE_TQSSTRING = "TQS String"
STYLE_TQDSTRING = "TQD String"
STYLE_KEYWORD = "Keyword"
STYLE_CLASS = "Class"
STYLE_METHOD = "Method"
STYLE_OPERATOR = "Operator"
STYLE_IDENTIFIER = "Identifier"
STYLE_BRACE = "Brace/Paren - matching"
STYLE_BRACEBAD = "Brace/Paren - unmatched"
STYLE_STRINGEOL = "String with no terminator"
STYLE_LINENUMBER = "Line numbers"
STYLE_INDENTGUIDE = "Indent guide"
STYLE_SELECTION = "Selection"

STRING_STYLES = [
    STYLE_STRING,
    STYLE_SQSTRING,
    STYLE_TQSSTRING,
    STYLE_TQDSTRING,
    STYLE_STRINGEOL,
]

# These styles can have any ID - they are not special to scintilla itself.
# However, if we use the built-in lexer, then we must use its style numbers
# so in that case, they _are_ special.
# (name, format, background, scintilla id)
PYTHON_STYLES = [
    (STYLE_DEFAULT, (0, 0, 200, 0, 0x808080), CLR_INVALID, scintillacon.SCE_P_DEFAULT),
    (
        STYLE_COMMENT,
        (0, 2, 200, 0, 0x008000),
        CLR_INVALID,
        scintillacon.SCE_P_COMMENTLINE,
    ),
    (
        STYLE_COMMENT_BLOCK,
        (0, 2, 200, 0, 0x808080),
        CLR_INVALID,
        scintillacon.SCE_P_COMMENTBLOCK,
    ),
    (STYLE_NUMBER, (0, 0, 200, 0, 0x808000), CLR_INVALID, scintillacon.SCE_P_NUMBER),
    (STYLE_STRING, (0, 0, 200, 0, 0x008080), CLR_INVALID, scintillacon.SCE_P_STRING),
    (STYLE_SQSTRING, STYLE_STRING, CLR_INVALID, scintillacon.SCE_P_CHARACTER),
    (STYLE_TQSSTRING, STYLE_STRING, CLR_INVALID, scintillacon.SCE_P_TRIPLE),
    (STYLE_TQDSTRING, STYLE_STRING, CLR_INVALID, scintillacon.SCE_P_TRIPLEDOUBLE),
    (STYLE_STRINGEOL, (0, 0, 200, 0, 0x000000), 0x008080, scintillacon.SCE_P_STRINGEOL),
    (STYLE_KEYWORD, (0, 1, 200, 0, 0x800000), CLR_INVALID, scintillacon.SCE_P_WORD),
    (STYLE_CLASS, (0, 1, 200, 0, 0xFF0000), CLR_INVALID, scintillacon.SCE_P_CLASSNAME),
    (STYLE_METHOD, (0, 1, 200, 0, 0x808000), CLR_INVALID, scintillacon.SCE_P_DEFNAME),
    (
        STYLE_OPERATOR,
        (0, 0, 200, 0, 0x000000),
        CLR_INVALID,
        scintillacon.SCE_P_OPERATOR,
    ),
    (
        STYLE_IDENTIFIER,
        (0, 0, 200, 0, 0x000000),
        CLR_INVALID,
        scintillacon.SCE_P_IDENTIFIER,
    ),
]

# These styles _always_ have this specific style number, regardless of
# internal or external formatter.
SPECIAL_STYLES = [
    (STYLE_BRACE, (0, 0, 200, 0, 0x000000), 0xFFFF80, scintillacon.STYLE_BRACELIGHT),
    (STYLE_BRACEBAD, (0, 0, 200, 0, 0x000000), 0x8EA5F2, scintillacon.STYLE_BRACEBAD),
    (
        STYLE_LINENUMBER,
        (0, 0, 200, 0, 0x000000),
        win32api.GetSysColor(win32con.COLOR_3DFACE),
        scintillacon.STYLE_LINENUMBER,
    ),
    (
        STYLE_INDENTGUIDE,
        (0, 0, 200, 0, 0x000000),
        CLR_INVALID,
        scintillacon.STYLE_INDENTGUIDE,
    ),
    ## Not actually a style; requires special handling to send appropriate messages to scintilla
    (
        STYLE_SELECTION,
        (0, 0, 200, 0, CLR_INVALID),
        win32api.RGB(0xC0, 0xC0, 0xC0),
        999999,
    ),
]

PythonSampleCode = """\
# Some Python
class Sample(Super):
  def Fn(self):
\tself.v = 1024
dest = 'dest.html'
x = func(a + 1)|)
s = "I forget...
## A large
## comment block"""


class PythonSourceFormatter(Formatter):
    string_style_names = STRING_STYLES

    def GetSampleText(self):
        return PythonSampleCode

    def LoadStyles(self):
        pass

    def SetStyles(self):
        for name, format, bg, ignore in PYTHON_STYLES:
            self.RegisterStyle(Style(name, format, bg))
        for name, format, bg, sc_id in SPECIAL_STYLES:
            self.RegisterStyle(Style(name, format, bg), sc_id)

    def ClassifyWord(self, cdoc, start, end, prevWord):
        word = cdoc[start : end + 1].decode("latin-1")
        attr = STYLE_IDENTIFIER
        if prevWord == "class":
            attr = STYLE_CLASS
        elif prevWord == "def":
            attr = STYLE_METHOD
        elif word[0] in string.digits:
            attr = STYLE_NUMBER
        elif iskeyword(word):
            attr = STYLE_KEYWORD
        self.ColorSeg(start, end, attr)
        return word

    def ColorizeString(self, str, styleStart):
        if styleStart is None:
            styleStart = STYLE_DEFAULT
        return self.ColorizePythonCode(str, 0, styleStart)

    def ColorizePythonCode(self, cdoc, charStart, styleStart):
        # Straight translation of C++, should do better
        lengthDoc = len(cdoc)
        if lengthDoc <= charStart:
            return
        prevWord = ""
        state = styleStart
        chPrev = chPrev2 = chPrev3 = " "
        chNext2 = chNext = cdoc[charStart : charStart + 1].decode("latin-1")
        startSeg = i = charStart
        while i < lengthDoc:
            ch = chNext
            chNext = " "
            if i + 1 < lengthDoc:
                chNext = cdoc[i + 1 : i + 2].decode("latin-1")
            chNext2 = " "
            if i + 2 < lengthDoc:
                chNext2 = cdoc[i + 2 : i + 3].decode("latin-1")
            if state == STYLE_DEFAULT:
                if ch in wordstarts:
                    self.ColorSeg(startSeg, i - 1, STYLE_DEFAULT)
                    state = STYLE_KEYWORD
                    startSeg = i
                elif ch == "#":
                    self.ColorSeg(startSeg, i - 1, STYLE_DEFAULT)
                    if chNext == "#":
                        state = STYLE_COMMENT_BLOCK
                    else:
                        state = STYLE_COMMENT
                    startSeg = i
                elif ch == '"':
                    self.ColorSeg(startSeg, i - 1, STYLE_DEFAULT)
                    startSeg = i
                    state = STYLE_COMMENT
                    if chNext == '"' and chNext2 == '"':
                        i = i + 2
                        state = STYLE_TQDSTRING
                        ch = " "
                        chPrev = " "
                        chNext = " "
                        if i + 1 < lengthDoc:
                            chNext = cdoc[i + 1]
                    else:
                        state = STYLE_STRING
                elif ch == "'":
                    self.ColorSeg(startSeg, i - 1, STYLE_DEFAULT)
                    startSeg = i
                    state = STYLE_COMMENT
                    if chNext == "'" and chNext2 == "'":
                        i = i + 2
                        state = STYLE_TQSSTRING
                        ch = " "
                        chPrev = " "
                        chNext = " "
                        if i + 1 < lengthDoc:
                            chNext = cdoc[i + 1]
                    else:
                        state = STYLE_SQSTRING
                elif ch in operators:
                    self.ColorSeg(startSeg, i - 1, STYLE_DEFAULT)
                    self.ColorSeg(i, i, STYLE_OPERATOR)
                    startSeg = i + 1
            elif state == STYLE_KEYWORD:
                if ch not in wordchars:
                    prevWord = self.ClassifyWord(cdoc, startSeg, i - 1, prevWord)
                    state = STYLE_DEFAULT
                    startSeg = i
                    if ch == "#":
                        if chNext == "#":
                            state = STYLE_COMMENT_BLOCK
                        else:
                            state = STYLE_COMMENT
                    elif ch == '"':
                        if chNext == '"' and chNext2 == '"':
                            i = i + 2
                            state = STYLE_TQDSTRING
                            ch = " "
                            chPrev = " "
                            chNext = " "
                            if i + 1 < lengthDoc:
                                chNext = cdoc[i + 1]
                        else:
                            state = STYLE_STRING
                    elif ch == "'":
                        if chNext == "'" and chNext2 == "'":
                            i = i + 2
                            state = STYLE_TQSSTRING
                            ch = " "
                            chPrev = " "
                            chNext = " "
                            if i + 1 < lengthDoc:
                                chNext = cdoc[i + 1]
                        else:
                            state = STYLE_SQSTRING
                    elif ch in operators:
                        self.ColorSeg(startSeg, i, STYLE_OPERATOR)
                        startSeg = i + 1
            elif state == STYLE_COMMENT or state == STYLE_COMMENT_BLOCK:
                if ch == "\r" or ch == "\n":
                    self.ColorSeg(startSeg, i - 1, state)
                    state = STYLE_DEFAULT
                    startSeg = i
            elif state == STYLE_STRING:
                if ch == "\\":
                    if chNext == '"' or chNext == "'" or chNext == "\\":
                        i = i + 1
                        ch = chNext
                        chNext = " "
                        if i + 1 < lengthDoc:
                            chNext = cdoc[i + 1]
                elif ch == '"':
                    self.ColorSeg(startSeg, i, STYLE_STRING)
                    state = STYLE_DEFAULT
                    startSeg = i + 1
            elif state == STYLE_SQSTRING:
                if ch == "\\":
                    if chNext == '"' or chNext == "'" or chNext == "\\":
                        i = i + 1
                        ch = chNext
                        chNext = " "
                        if i + 1 < lengthDoc:
                            chNext = cdoc[i + 1]
                elif ch == "'":
                    self.ColorSeg(startSeg, i, STYLE_SQSTRING)
                    state = STYLE_DEFAULT
                    startSeg = i + 1
            elif state == STYLE_TQSSTRING:
                if ch == "'" and chPrev == "'" and chPrev2 == "'" and chPrev3 != "\\":
                    self.ColorSeg(startSeg, i, STYLE_TQSSTRING)
                    state = STYLE_DEFAULT
                    startSeg = i + 1
            elif (
                state == STYLE_TQDSTRING
                and ch == '"'
                and chPrev == '"'
                and chPrev2 == '"'
                and chPrev3 != "\\"
            ):
                self.ColorSeg(startSeg, i, STYLE_TQDSTRING)
                state = STYLE_DEFAULT
                startSeg = i + 1
            chPrev3 = chPrev2
            chPrev2 = chPrev
            chPrev = ch
            i = i + 1
        if startSeg < lengthDoc:
            if state == STYLE_KEYWORD:
                self.ClassifyWord(cdoc, startSeg, lengthDoc - 1, prevWord)
            else:
                self.ColorSeg(startSeg, lengthDoc - 1, state)


# These taken from the SciTE properties file.
source_formatter_extensions = [
    (".py .pys .pyw".split(), scintillacon.SCLEX_PYTHON),
    (".html .htm .asp .shtml".split(), scintillacon.SCLEX_HTML),
    (
        "c .cc .cpp .cxx .h .hh .hpp .hxx .idl .odl .php3 .phtml .inc .js".split(),
        scintillacon.SCLEX_CPP,
    ),
    (".vbs .frm .ctl .cls".split(), scintillacon.SCLEX_VB),
    (".pl .pm .cgi .pod".split(), scintillacon.SCLEX_PERL),
    (".sql .spec .body .sps .spb .sf .sp".split(), scintillacon.SCLEX_SQL),
    (".tex .sty".split(), scintillacon.SCLEX_LATEX),
    (".xml .xul".split(), scintillacon.SCLEX_XML),
    (".err".split(), scintillacon.SCLEX_ERRORLIST),
    (".mak".split(), scintillacon.SCLEX_MAKEFILE),
    (".bat .cmd".split(), scintillacon.SCLEX_BATCH),
]


class BuiltinSourceFormatter(FormatterBase):
    # A class that represents a formatter built-in to Scintilla
    def __init__(self, scintilla, ext):
        self.ext = ext
        FormatterBase.__init__(self, scintilla)

    def Colorize(self, start=0, end=-1):
        self.scintilla.SendScintilla(scintillacon.SCI_COLOURISE, start, end)

    def RegisterStyle(self, style, stylenum=None):
        assert style.stylenum is None, "Style has already been registered"
        if stylenum is None:
            stylenum = self.nextstylenum
            self.nextstylenum = self.nextstylenum + 1
        assert self.styles.get(stylenum) is None, "We are reusing a style number!"
        style.stylenum = stylenum
        self.styles[style.name] = style
        self.styles_by_id[stylenum] = style

    def HookFormatter(self, parent=None):
        sc = self.scintilla
        for exts, formatter in source_formatter_extensions:
            if self.ext in exts:
                formatter_use = formatter
                break
        else:
            formatter_use = scintillacon.SCLEX_PYTHON
        sc.SendScintilla(scintillacon.SCI_SETLEXER, formatter_use)
        keywords = " ".join(kwlist)
        sc.SCISetKeywords(keywords)


class BuiltinPythonSourceFormatter(BuiltinSourceFormatter):
    sci_lexer_name = scintillacon.SCLEX_PYTHON
    string_style_names = STRING_STYLES

    def __init__(self, sc, ext=".py"):
        BuiltinSourceFormatter.__init__(self, sc, ext)

    def SetStyles(self):
        for name, format, bg, sc_id in PYTHON_STYLES:
            self.RegisterStyle(Style(name, format, bg), sc_id)
        for name, format, bg, sc_id in SPECIAL_STYLES:
            self.RegisterStyle(Style(name, format, bg), sc_id)

    def GetSampleText(self):
        return PythonSampleCode
