# Code that allows Pythonwin to pretend it is IDLE
# (at least as far as most IDLE extensions are concerned)

import string
import sys

import win32api
import win32con
import win32ui
from pywin import default_scintilla_encoding
from pywin.mfc.dialog import GetSimpleInput

wordchars = string.ascii_uppercase + string.ascii_lowercase + string.digits


class TextError(Exception):  # When a TclError would normally be raised.
    pass


class EmptyRange(Exception):  # Internally raised.
    pass


def GetIDLEModule(module):
    try:
        # First get it from Pythonwin it is exists.
        modname = "pywin.idle." + module
        __import__(modname)
    except ImportError as details:
        msg = (
            "The IDLE extension '%s' can not be located.\r\n\r\n"
            "Please correct the installation and restart the"
            " application.\r\n\r\n%s" % (module, details)
        )
        win32ui.MessageBox(msg)
        return None
    mod = sys.modules[modname]
    mod.TclError = TextError  # A hack that can go soon!
    return mod


# A class that is injected into the IDLE auto-indent extension.
# It allows for decent performance when opening a new file,
# as auto-indent uses the tokenizer module to determine indents.
# The default AutoIndent readline method works OK, but it goes through
# this layer of Tk index indirection for every single line.  For large files
# without indents (and even small files with indents :-) it was pretty slow!
def fast_readline(self):
    if self.finished:
        val = ""
    else:
        if "_scint_lines" not in self.__dict__:
            # XXX - note - assumes this is only called once the file is loaded!
            self._scint_lines = self.text.edit.GetTextRange().split("\n")
        sl = self._scint_lines
        i = self.i = self.i + 1
        if i >= len(sl):
            val = ""
        else:
            val = sl[i] + "\n"
    return val.encode(default_scintilla_encoding)


try:
    GetIDLEModule("AutoIndent").IndentSearcher.readline = fast_readline
except AttributeError:  # GetIDLEModule may return None
    pass


# A class that attempts to emulate an IDLE editor window.
# Construct with a Pythonwin view.
class IDLEEditorWindow:
    def __init__(self, edit):
        self.edit = edit
        self.text = TkText(edit)
        self.extensions = {}
        self.extension_menus = {}

    def close(self):
        self.edit = self.text = None
        self.extension_menus = None
        try:
            for ext in self.extensions.values():
                closer = getattr(ext, "close", None)
                if closer is not None:
                    closer()
        finally:
            self.extensions = {}

    def IDLEExtension(self, extension):
        ext = self.extensions.get(extension)
        if ext is not None:
            return ext
        mod = GetIDLEModule(extension)
        if mod is None:
            return None
        klass = getattr(mod, extension)
        ext = self.extensions[extension] = klass(self)
        # Find and bind all the events defined in the extension.
        events = [item for item in dir(klass) if item[-6:] == "_event"]
        for event in events:
            name = "<<%s>>" % (event[:-6].replace("_", "-"),)
            self.edit.bindings.bind(name, getattr(ext, event))
        return ext

    def GetMenuItems(self, menu_name):
        # Get all menu items for the menu name (eg, "edit")
        bindings = self.edit.bindings
        ret = []
        for ext in self.extensions.values():
            menudefs = getattr(ext, "menudefs", [])
            for name, items in menudefs:
                if name == menu_name:
                    for text, event in [item for item in items if item is not None]:
                        text = text.replace("&", "&&")
                        text = text.replace("_", "&")
                        ret.append((text, event))
        return ret

    ######################################################################
    # The IDLE "Virtual UI" methods that are exposed to the IDLE extensions.
    #
    def askinteger(
        self, caption, prompt, parent=None, initialvalue=0, minvalue=None, maxvalue=None
    ):
        while 1:
            rc = GetSimpleInput(prompt, str(initialvalue), caption)
            if rc is None:
                return 0  # Correct "cancel" semantics?
            err = None
            try:
                rc = int(rc)
            except ValueError:
                err = "Please enter an integer"
            if not err and minvalue is not None and rc < minvalue:
                err = "Please enter an integer greater then or equal to %s" % (
                    minvalue,
                )
            if not err and maxvalue is not None and rc > maxvalue:
                err = "Please enter an integer less then or equal to %s" % (maxvalue,)
            if err:
                win32ui.MessageBox(err, caption, win32con.MB_OK)
                continue
            return rc

    def askyesno(self, caption, prompt, parent=None):
        return win32ui.MessageBox(prompt, caption, win32con.MB_YESNO) == win32con.IDYES

    ######################################################################
    # The IDLE "Virtual Text Widget" methods that are exposed to the IDLE extensions.
    #

    # Is character at text_index in a Python string?  Return 0 for
    # "guaranteed no", true for anything else.
    def is_char_in_string(self, text_index):
        # A helper for the code analyser - we need internal knowledge of
        # the colorizer to get this information
        # This assumes the colorizer has got to this point!
        text_index = self.text._getoffset(text_index)
        c = self.text.edit._GetColorizer()
        if c and c.GetStringStyle(text_index) is None:
            return 0
        return 1

    # If a selection is defined in the text widget, return
    # (start, end) as Tkinter text indices, otherwise return
    # (None, None)
    def get_selection_indices(self):
        try:
            first = self.text.index("sel.first")
            last = self.text.index("sel.last")
            return first, last
        except TextError:
            return None, None

    def set_tabwidth(self, width):
        self.edit.SCISetTabWidth(width)

    def get_tabwidth(self):
        return self.edit.GetTabWidth()


# A class providing the generic "Call Tips" interface
class CallTips:
    def __init__(self, edit):
        self.edit = edit

    def showtip(self, tip_text):
        self.edit.SCICallTipShow(tip_text)

    def hidetip(self):
        self.edit.SCICallTipCancel()


########################################
#
# Helpers for the TkText emulation.
def TkOffsetToIndex(offset, edit):
    lineoff = 0
    # May be 1 > actual end if we pretended there was a trailing '\n'
    offset = min(offset, edit.GetTextLength())
    line = edit.LineFromChar(offset)
    lineIndex = edit.LineIndex(line)
    return "%d.%d" % (line + 1, offset - lineIndex)


def _NextTok(str, pos):
    # Returns (token, endPos)
    end = len(str)
    if pos >= end:
        return None, 0
    while pos < end and str[pos] in string.whitespace:
        pos = pos + 1
    # Special case for +-
    if str[pos] in "+-":
        return str[pos], pos + 1
    # Digits also a special case.
    endPos = pos
    while endPos < end and str[endPos] in string.digits + ".":
        endPos = endPos + 1
    if pos != endPos:
        return str[pos:endPos], endPos
    endPos = pos
    while endPos < end and str[endPos] not in string.whitespace + string.digits + "+-":
        endPos = endPos + 1
    if pos != endPos:
        return str[pos:endPos], endPos
    return None, 0


def TkIndexToOffset(bm, edit, marks):
    base, nextTokPos = _NextTok(bm, 0)
    if base is None:
        raise ValueError("Empty bookmark ID!")
    if base.find(".") > 0:
        try:
            line, col = base.split(".", 2)
            if col == "first" or col == "last":
                # Tag name
                if line != "sel":
                    raise ValueError("Tags arent here!")
                sel = edit.GetSel()
                if sel[0] == sel[1]:
                    raise EmptyRange
                if col == "first":
                    pos = sel[0]
                else:
                    pos = sel[1]
            else:
                # Lines are 1 based for tkinter
                line = int(line) - 1
                if line > edit.GetLineCount():
                    pos = edit.GetTextLength() + 1
                else:
                    pos = edit.LineIndex(line)
                    if pos == -1:
                        pos = edit.GetTextLength()
                    pos = pos + int(col)
        except (ValueError, IndexError):
            raise ValueError("Unexpected literal in '%s'" % base)
    elif base == "insert":
        pos = edit.GetSel()[0]
    elif base == "end":
        pos = edit.GetTextLength()
        # Pretend there is a trailing '\n' if necessary
        if pos and edit.SCIGetCharAt(pos - 1) != "\n":
            pos = pos + 1
    else:
        try:
            pos = marks[base]
        except KeyError:
            raise ValueError("Unsupported base offset or undefined mark '%s'" % base)

    while 1:
        word, nextTokPos = _NextTok(bm, nextTokPos)
        if word is None:
            break
        if word in ("+", "-"):
            num, nextTokPos = _NextTok(bm, nextTokPos)
            if num is None:
                raise ValueError("+/- operator needs 2 args")
            what, nextTokPos = _NextTok(bm, nextTokPos)
            if what is None:
                raise ValueError("+/- operator needs 2 args")
            if what[0] != "c":
                raise ValueError("+/- only supports chars")
            if word == "+":
                pos = pos + int(num)
            else:
                pos = pos - int(num)
        elif word == "wordstart":
            while pos > 0 and edit.SCIGetCharAt(pos - 1) in wordchars:
                pos = pos - 1
        elif word == "wordend":
            end = edit.GetTextLength()
            while pos < end and edit.SCIGetCharAt(pos) in wordchars:
                pos = pos + 1
        elif word == "linestart":
            while pos > 0 and edit.SCIGetCharAt(pos - 1) not in "\n\r":
                pos = pos - 1
        elif word == "lineend":
            end = edit.GetTextLength()
            while pos < end and edit.SCIGetCharAt(pos) not in "\n\r":
                pos = pos + 1
        else:
            raise ValueError("Unsupported relative offset '%s'" % word)
    return max(pos, 0)  # Tkinter is tollerant of -ve indexes - we aren't


# A class that resembles an IDLE (ie, a Tk) text widget.
# Construct with an edit object (eg, an editor view)
class TkText:
    def __init__(self, edit):
        self.calltips = None
        self.edit = edit
        self.marks = {}

    ##	def __getattr__(self, attr):
    ##		if attr=="tk": return self # So text.tk.call works.
    ##		if attr=="master": return None # ditto!
    ##		raise AttributeError, attr
    ##	def __getitem__(self, item):
    ##		if item=="tabs":
    ##			size = self.edit.GetTabWidth()
    ##			if size==8: return "" # Tk default
    ##			return size # correct semantics?
    ##		elif item=="font": # Used for measurements we dont need to do!
    ##			return "Dont know the font"
    ##		raise IndexError, "Invalid index '%s'" % item
    def make_calltip_window(self):
        if self.calltips is None:
            self.calltips = CallTips(self.edit)
        return self.calltips

    def _getoffset(self, index):
        return TkIndexToOffset(index, self.edit, self.marks)

    def _getindex(self, off):
        return TkOffsetToIndex(off, self.edit)

    def _fix_indexes(self, start, end):
        # first some magic to handle skipping over utf8 extended chars.
        while start > 0 and ord(self.edit.SCIGetCharAt(start)) & 0xC0 == 0x80:
            start -= 1
        while (
            end < self.edit.GetTextLength()
            and ord(self.edit.SCIGetCharAt(end)) & 0xC0 == 0x80
        ):
            end += 1
        # now handling fixing \r\n->\n disparities...
        if (
            start > 0
            and self.edit.SCIGetCharAt(start) == "\n"
            and self.edit.SCIGetCharAt(start - 1) == "\r"
        ):
            start = start - 1
        if (
            end < self.edit.GetTextLength()
            and self.edit.SCIGetCharAt(end - 1) == "\r"
            and self.edit.SCIGetCharAt(end) == "\n"
        ):
            end = end + 1
        return start, end

    ##	def get_tab_width(self):
    ##		return self.edit.GetTabWidth()
    ##	def call(self, *rest):
    ##		# Crap to support Tk measurement hacks for tab widths
    ##		if rest[0] != "font" or rest[1] != "measure":
    ##			raise ValueError, "Unsupport call type"
    ##		return len(rest[5])
    ##	def configure(self, **kw):
    ##		for name, val in kw.items():
    ##			if name=="tabs":
    ##				self.edit.SCISetTabWidth(int(val))
    ##			else:
    ##				raise ValueError, "Unsupported configuration item %s" % kw
    def bind(self, binding, handler):
        self.edit.bindings.bind(binding, handler)

    def get(self, start, end=None):
        try:
            start = self._getoffset(start)
            if end is None:
                end = start + 1
            else:
                end = self._getoffset(end)
        except EmptyRange:
            return ""
        # Simple semantic checks to conform to the Tk text interface
        if end <= start:
            return ""
        max = self.edit.GetTextLength()
        checkEnd = 0
        if end > max:
            end = max
            checkEnd = 1
        start, end = self._fix_indexes(start, end)
        ret = self.edit.GetTextRange(start, end)
        # pretend a trailing '\n' exists if necessary.
        if checkEnd and (not ret or ret[-1] != "\n"):
            ret = ret + "\n"
        return ret.replace("\r", "")

    def index(self, spec):
        try:
            return self._getindex(self._getoffset(spec))
        except EmptyRange:
            return ""

    def insert(self, pos, text):
        try:
            pos = self._getoffset(pos)
        except EmptyRange:
            raise TextError("Empty range")
        self.edit.SetSel((pos, pos))
        # IDLE only deals with "\n" - we will be nicer

        bits = text.split("\n")
        self.edit.SCIAddText(bits[0])
        for bit in bits[1:]:
            self.edit.SCINewline()
            self.edit.SCIAddText(bit)

    def delete(self, start, end=None):
        try:
            start = self._getoffset(start)
            if end is not None:
                end = self._getoffset(end)
        except EmptyRange:
            raise TextError("Empty range")
        # If end is specified and == start, then we must delete nothing.
        if start == end:
            return
        # If end is not specified, delete one char
        if end is None:
            end = start + 1
        else:
            # Tk says not to delete in this case, but our control would.
            if end < start:
                return
        if start == self.edit.GetTextLength():
            return  # Nothing to delete.
        old = self.edit.GetSel()[0]  # Lose a selection
        # Hack for partial '\r\n' and UTF-8 char removal
        start, end = self._fix_indexes(start, end)
        self.edit.SetSel((start, end))
        self.edit.Clear()
        if old >= start and old < end:
            old = start
        elif old >= end:
            old = old - (end - start)
        self.edit.SetSel(old)

    def bell(self):
        win32api.MessageBeep()

    def see(self, pos):
        # Most commands we use in Scintilla actually force the selection
        # to be seen, making this unnecessary.
        pass

    def mark_set(self, name, pos):
        try:
            pos = self._getoffset(pos)
        except EmptyRange:
            raise TextError("Empty range '%s'" % pos)
        if name == "insert":
            self.edit.SetSel(pos)
        else:
            self.marks[name] = pos

    def tag_add(self, name, start, end):
        if name != "sel":
            raise ValueError("Only sel tag is supported")
        try:
            start = self._getoffset(start)
            end = self._getoffset(end)
        except EmptyRange:
            raise TextError("Empty range")
        self.edit.SetSel(start, end)

    def tag_remove(self, name, start, end):
        if name != "sel" or start != "1.0" or end != "end":
            raise ValueError("Cant remove this tag")
        # Turn the sel into a cursor
        self.edit.SetSel(self.edit.GetSel()[0])

    def compare(self, i1, op, i2):
        try:
            i1 = self._getoffset(i1)
        except EmptyRange:
            i1 = ""
        try:
            i2 = self._getoffset(i2)
        except EmptyRange:
            i2 = ""
        return eval("%d%s%d" % (i1, op, i2))

    def undo_block_start(self):
        self.edit.SCIBeginUndoAction()

    def undo_block_stop(self):
        self.edit.SCIEndUndoAction()


######################################################################
#
# Test related code.
#
######################################################################
def TestCheck(index, edit, expected=None):
    rc = TkIndexToOffset(index, edit, {})
    if rc != expected:
        print("ERROR: Index", index, ", expected", expected, "but got", rc)


def TestGet(fr, to, t, expected):
    got = t.get(fr, to)
    if got != expected:
        print(
            "ERROR: get(%s, %s) expected %s, but got %s"
            % (repr(fr), repr(to), repr(expected), repr(got))
        )


def test():
    import pywin.framework.editor

    d = pywin.framework.editor.editorTemplate.OpenDocumentFile(None)
    e = d.GetFirstView()
    t = TkText(e)
    e.SCIAddText("hi there how\nare you today\r\nI hope you are well")
    e.SetSel((4, 4))

    skip = """
	TestCheck("insert", e, 4)
	TestCheck("insert wordstart", e, 3)
	TestCheck("insert wordend", e, 8)
	TestCheck("insert linestart", e, 0)
	TestCheck("insert lineend", e, 12)
	TestCheck("insert + 4 chars", e, 8)
	TestCheck("insert +4c", e, 8)
	TestCheck("insert - 2 chars", e, 2)
	TestCheck("insert -2c", e, 2)
	TestCheck("insert-2c", e, 2)
	TestCheck("insert-2 c", e, 2)
	TestCheck("insert- 2c", e, 2)
	TestCheck("1.1", e, 1)
	TestCheck("1.0", e, 0)
	TestCheck("2.0", e, 13)
	try:
		TestCheck("sel.first", e, 0)
		print "*** sel.first worked with an empty selection"
	except TextError:
		pass
	e.SetSel((4,5))
	TestCheck("sel.first- 2c", e, 2)
	TestCheck("sel.last- 2c", e, 3)
	"""
    # Check EOL semantics
    e.SetSel((4, 4))
    TestGet("insert lineend", "insert lineend +1c", t, "\n")
    e.SetSel((20, 20))
    TestGet("insert lineend", "insert lineend +1c", t, "\n")
    e.SetSel((35, 35))
    TestGet("insert lineend", "insert lineend +1c", t, "\n")


class IDLEWrapper:
    def __init__(self, control):
        self.text = control


def IDLETest(extension):
    import os
    import sys

    modname = "pywin.idle." + extension
    __import__(modname)
    mod = sys.modules[modname]
    mod.TclError = TextError
    klass = getattr(mod, extension)

    # Create a new Scintilla Window.
    import pywin.framework.editor

    d = pywin.framework.editor.editorTemplate.OpenDocumentFile(None)
    v = d.GetFirstView()
    fname = os.path.splitext(__file__)[0] + ".py"
    v.SCIAddText(open(fname).read())
    d.SetModifiedFlag(0)
    r = klass(IDLEWrapper(TkText(v)))
    return r


if __name__ == "__main__":
    test()
