import codecs
import re
import string

import win32con
import win32ui
from pywin import default_scintilla_encoding
from pywin.mfc import docview

from . import scintillacon

crlf_bytes = "\r\n".encode("ascii")
lf_bytes = "\n".encode("ascii")

# re from pep263 - but we use it both on bytes and strings.
re_encoding_bytes = re.compile("coding[:=]\s*([-\w.]+)".encode("ascii"))
re_encoding_text = re.compile("coding[:=]\s*([-\w.]+)")

ParentScintillaDocument = docview.Document


class CScintillaDocument(ParentScintillaDocument):
    "A SyntEdit document."

    def __init__(self, *args):
        self.bom = None  # the BOM, if any, read from the file.
        # the encoding we detected from the source.  Might have
        # detected via the BOM or an encoding decl.  Note that in
        # the latter case (ie, while self.bom is None), it can't be
        # trusted - the user may have edited the encoding decl between
        # open and save.
        self.source_encoding = None
        ParentScintillaDocument.__init__(self, *args)

    def DeleteContents(self):
        pass

    def OnOpenDocument(self, filename):
        # init data members
        # print "Opening", filename
        self.SetPathName(filename)  # Must set this early!
        try:
            # load the text as binary we can get smart
            # about detecting any existing EOL conventions.
            f = open(filename, "rb")
            try:
                self._LoadTextFromFile(f)
            finally:
                f.close()
        except IOError:
            rc = win32ui.MessageBox(
                "Could not load the file from %s\n\nDo you want to create a new file?"
                % filename,
                "Pythonwin",
                win32con.MB_YESNO | win32con.MB_ICONWARNING,
            )
            if rc == win32con.IDNO:
                return 0
            assert rc == win32con.IDYES, rc
            try:
                f = open(filename, "wb+")
                try:
                    self._LoadTextFromFile(f)
                finally:
                    f.close()
            except IOError as e:
                rc = win32ui.MessageBox("Cannot create the file %s" % filename)
        return 1

    def SaveFile(self, fileName, encoding=None):
        view = self.GetFirstView()
        ok = view.SaveTextFile(fileName, encoding=encoding)
        if ok:
            view.SCISetSavePoint()
        return ok

    def ApplyFormattingStyles(self):
        self._ApplyOptionalToViews("ApplyFormattingStyles")

    # #####################
    # File related functions
    # Helper to transfer text from the MFC document to the control.
    def _LoadTextFromFile(self, f):
        # detect EOL mode - we don't support \r only - so find the
        # first '\n' and guess based on the char before.
        l = f.readline()
        l2 = f.readline()
        # If line ends with \r\n or has no line ending, use CRLF.
        if l.endswith(crlf_bytes) or not l.endswith(lf_bytes):
            eol_mode = scintillacon.SC_EOL_CRLF
        else:
            eol_mode = scintillacon.SC_EOL_LF

        # Detect the encoding - first look for a BOM, and if not found,
        # look for a pep263 encoding declaration.
        for bom, encoding in (
            (codecs.BOM_UTF8, "utf8"),
            (codecs.BOM_UTF16_LE, "utf_16_le"),
            (codecs.BOM_UTF16_BE, "utf_16_be"),
        ):
            if l.startswith(bom):
                self.bom = bom
                self.source_encoding = encoding
                l = l[len(bom) :]  # remove it.
                break
        else:
            # no bom detected - look for pep263 encoding decl.
            for look in (l, l2):
                # Note we are looking at raw bytes here: so
                # both the re itself uses bytes and the result
                # is bytes - but we need the result as a string.
                match = re_encoding_bytes.search(look)
                if match is not None:
                    self.source_encoding = match.group(1).decode("ascii")
                    break

        # reading by lines would be too slow?  Maybe we can use the
        # incremental encoders? For now just stick with loading the
        # entire file in memory.
        text = l + l2 + f.read()

        # Translate from source encoding to UTF-8 bytes for Scintilla
        source_encoding = self.source_encoding
        # If we don't know an encoding, try utf-8 - if that fails we will
        # fallback to latin-1 to treat it as bytes...
        if source_encoding is None:
            source_encoding = "utf-8"
        # we could optimize this by avoiding utf8 to-ing and from-ing,
        # but then we would lose the ability to handle invalid utf8
        # (and even then, the use of encoding aliases makes this tricky)
        # To create an invalid utf8 file:
        # >>> open(filename, "wb").write(codecs.BOM_UTF8+"bad \xa9har\r\n")
        try:
            dec = text.decode(source_encoding)
        except UnicodeError:
            print(
                "WARNING: Failed to decode bytes from '%s' encoding - treating as latin1"
                % source_encoding
            )
            dec = text.decode("latin1")
        except LookupError:
            print(
                "WARNING: Invalid encoding '%s' specified - treating as latin1"
                % source_encoding
            )
            dec = text.decode("latin1")
        # and put it back as utf8 - this shouldn't fail.
        text = dec.encode(default_scintilla_encoding)

        view = self.GetFirstView()
        if view.IsWindow():
            # Turn off undo collection while loading
            view.SendScintilla(scintillacon.SCI_SETUNDOCOLLECTION, 0, 0)
            # Make sure the control isnt read-only
            view.SetReadOnly(0)
            view.SendScintilla(scintillacon.SCI_CLEARALL)
            view.SendMessage(scintillacon.SCI_ADDTEXT, text)
            view.SendScintilla(scintillacon.SCI_SETUNDOCOLLECTION, 1, 0)
            view.SendScintilla(win32con.EM_EMPTYUNDOBUFFER, 0, 0)
            # set EOL mode
            view.SendScintilla(scintillacon.SCI_SETEOLMODE, eol_mode)

    def _SaveTextToFile(self, view, filename, encoding=None):
        s = view.GetTextRange()  # already decoded from scintilla's encoding
        source_encoding = encoding
        if source_encoding is None:
            if self.bom:
                source_encoding = self.source_encoding
            else:
                # no BOM - look for an encoding.
                bits = re.split("[\r\n]+", s, 3)
                for look in bits[:-1]:
                    match = re_encoding_text.search(look)
                    if match is not None:
                        source_encoding = match.group(1)
                        self.source_encoding = source_encoding
                        break

            if source_encoding is None:
                source_encoding = "utf-8"

        ## encode data before opening file so script is not lost if encoding fails
        file_contents = s.encode(source_encoding)
        # Open in binary mode as scintilla itself ensures the
        # line endings are already appropriate
        f = open(filename, "wb")
        try:
            if self.bom:
                f.write(self.bom)
            f.write(file_contents)
        finally:
            f.close()
        self.SetModifiedFlag(0)

    def FinalizeViewCreation(self, view):
        pass

    def HookViewNotifications(self, view):
        parent = view.GetParentFrame()
        parent.HookNotify(
            ViewNotifyDelegate(self, "OnBraceMatch"), scintillacon.SCN_CHECKBRACE
        )
        parent.HookNotify(
            ViewNotifyDelegate(self, "OnMarginClick"), scintillacon.SCN_MARGINCLICK
        )
        parent.HookNotify(
            ViewNotifyDelegate(self, "OnNeedShown"), scintillacon.SCN_NEEDSHOWN
        )

        parent.HookNotify(
            DocumentNotifyDelegate(self, "OnSavePointReached"),
            scintillacon.SCN_SAVEPOINTREACHED,
        )
        parent.HookNotify(
            DocumentNotifyDelegate(self, "OnSavePointLeft"),
            scintillacon.SCN_SAVEPOINTLEFT,
        )
        parent.HookNotify(
            DocumentNotifyDelegate(self, "OnModifyAttemptRO"),
            scintillacon.SCN_MODIFYATTEMPTRO,
        )
        # Tell scintilla what characters should abort auto-complete.
        view.SCIAutoCStops(string.whitespace + "()[]:;+-/*=\\?'!#@$%^&,<>\"'|")

        if view != self.GetFirstView():
            view.SCISetDocPointer(self.GetFirstView().SCIGetDocPointer())

    def OnSavePointReached(self, std, extra):
        self.SetModifiedFlag(0)

    def OnSavePointLeft(self, std, extra):
        self.SetModifiedFlag(1)

    def OnModifyAttemptRO(self, std, extra):
        self.MakeDocumentWritable()

    # All Marker functions are 1 based.
    def MarkerAdd(self, lineNo, marker):
        self.GetEditorView().SCIMarkerAdd(lineNo - 1, marker)

    def MarkerCheck(self, lineNo, marker):
        v = self.GetEditorView()
        lineNo = lineNo - 1  # Make 0 based
        markerState = v.SCIMarkerGet(lineNo)
        return markerState & (1 << marker) != 0

    def MarkerToggle(self, lineNo, marker):
        v = self.GetEditorView()
        if self.MarkerCheck(lineNo, marker):
            v.SCIMarkerDelete(lineNo - 1, marker)
        else:
            v.SCIMarkerAdd(lineNo - 1, marker)

    def MarkerDelete(self, lineNo, marker):
        self.GetEditorView().SCIMarkerDelete(lineNo - 1, marker)

    def MarkerDeleteAll(self, marker):
        self.GetEditorView().SCIMarkerDeleteAll(marker)

    def MarkerGetNext(self, lineNo, marker):
        return self.GetEditorView().SCIMarkerNext(lineNo - 1, 1 << marker) + 1

    def MarkerAtLine(self, lineNo, marker):
        markerState = self.GetEditorView().SCIMarkerGet(lineNo - 1)
        return markerState & (1 << marker)

    # Helper for reflecting functions to views.
    def _ApplyToViews(self, funcName, *args):
        for view in self.GetAllViews():
            func = getattr(view, funcName)
            func(*args)

    def _ApplyOptionalToViews(self, funcName, *args):
        for view in self.GetAllViews():
            func = getattr(view, funcName, None)
            if func is not None:
                func(*args)

    def GetEditorView(self):
        # Find the first frame with a view,
        # then ask it to give the editor view
        # as it knows which one is "active"
        try:
            frame_gev = self.GetFirstView().GetParentFrame().GetEditorView
        except AttributeError:
            return self.GetFirstView()
        return frame_gev()


# Delegate to the correct view, based on the control that sent it.
class ViewNotifyDelegate:
    def __init__(self, doc, name):
        self.doc = doc
        self.name = name

    def __call__(self, std, extra):
        (hwndFrom, idFrom, code) = std
        for v in self.doc.GetAllViews():
            if v.GetSafeHwnd() == hwndFrom:
                return getattr(v, self.name)(*(std, extra))


# Delegate to the document, but only from a single view (as each view sends it seperately)
class DocumentNotifyDelegate:
    def __init__(self, doc, name):
        self.doc = doc
        self.delegate = getattr(doc, name)

    def __call__(self, std, extra):
        (hwndFrom, idFrom, code) = std
        if hwndFrom == self.doc.GetEditorView().GetSafeHwnd():
            self.delegate(*(std, extra))
