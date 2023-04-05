"""A utility class for a code container.

A code container is a class which holds source code for a debugger.  It knows how
to color the text, and also how to translate lines into offsets, and back.
"""

import sys
import tokenize

import win32api
import winerror
from win32com.axdebug import axdebug
from win32com.server.exception import Exception

from . import contexts
from .util import RaiseNotImpl, _wrap

_keywords = {}  # set of Python keywords
for name in """
 and assert break class continue def del elif else except exec
 finally for from global if import in is lambda not
 or pass print raise return try while
 """.split():
    _keywords[name] = 1


class SourceCodeContainer:
    def __init__(
        self,
        text,
        fileName="<Remove Me!>",
        sourceContext=0,
        startLineNumber=0,
        site=None,
        debugDocument=None,
    ):
        self.sourceContext = sourceContext  # The source context added by a smart host.
        self.text = text
        if text:
            self._buildlines()
        self.nextLineNo = 0
        self.fileName = fileName
        self.codeContexts = {}
        self.site = site
        self.startLineNumber = startLineNumber
        self.debugDocument = None

    def _Close(self):
        self.text = self.lines = self.lineOffsets = None
        self.codeContexts = None
        self.debugDocument = None
        self.site = None
        self.sourceContext = None

    def GetText(self):
        return self.text

    def GetName(self, dnt):
        assert 0, "You must subclass this"

    def GetFileName(self):
        return self.fileName

    def GetPositionOfLine(self, cLineNumber):
        self.GetText()  # Prime us.
        try:
            return self.lineOffsets[cLineNumber]
        except IndexError:
            raise Exception(scode=winerror.S_FALSE)

    def GetLineOfPosition(self, charPos):
        self.GetText()  # Prime us.
        lastOffset = 0
        lineNo = 0
        for lineOffset in self.lineOffsets[1:]:
            if lineOffset > charPos:
                break
            lastOffset = lineOffset
            lineNo = lineNo + 1
        else:  # for not broken.
            #                       print "Cant find", charPos, "in", self.lineOffsets
            raise Exception(scode=winerror.S_FALSE)
        #               print "GLOP ret=",lineNo,       (charPos-lastOffset)
        return lineNo, (charPos - lastOffset)

    def GetNextLine(self):
        if self.nextLineNo >= len(self.lines):
            self.nextLineNo = 0  # auto-reset.
            return ""
        rc = self.lines[self.nextLineNo]
        self.nextLineNo = self.nextLineNo + 1
        return rc

    def GetLine(self, num):
        self.GetText()  # Prime us.
        return self.lines[num]

    def GetNumChars(self):
        return len(self.GetText())

    def GetNumLines(self):
        self.GetText()  # Prime us.
        return len(self.lines)

    def _buildline(self, pos):
        i = self.text.find("\n", pos)
        if i < 0:
            newpos = len(self.text)
        else:
            newpos = i + 1
        r = self.text[pos:newpos]
        return r, newpos

    def _buildlines(self):
        self.lines = []
        self.lineOffsets = [0]
        line, pos = self._buildline(0)
        while line:
            self.lines.append(line)
            self.lineOffsets.append(pos)
            line, pos = self._buildline(pos)

    def _ProcessToken(self, type, token, spos, epos, line):
        srow, scol = spos
        erow, ecol = epos
        self.GetText()  # Prime us.
        linenum = srow - 1  # Lines zero based for us too.
        realCharPos = self.lineOffsets[linenum] + scol
        numskipped = realCharPos - self.lastPos
        if numskipped == 0:
            pass
        elif numskipped == 1:
            self.attrs.append(axdebug.SOURCETEXT_ATTR_COMMENT)
        else:
            self.attrs.append((axdebug.SOURCETEXT_ATTR_COMMENT, numskipped))
        kwSize = len(token)
        self.lastPos = realCharPos + kwSize
        attr = 0

        if type == tokenize.NAME:
            if token in _keywords:
                attr = axdebug.SOURCETEXT_ATTR_KEYWORD
        elif type == tokenize.STRING:
            attr = axdebug.SOURCETEXT_ATTR_STRING
        elif type == tokenize.NUMBER:
            attr = axdebug.SOURCETEXT_ATTR_NUMBER
        elif type == tokenize.OP:
            attr = axdebug.SOURCETEXT_ATTR_OPERATOR
        elif type == tokenize.COMMENT:
            attr = axdebug.SOURCETEXT_ATTR_COMMENT
        # else attr remains zero...
        if kwSize == 0:
            pass
        elif kwSize == 1:
            self.attrs.append(attr)
        else:
            self.attrs.append((attr, kwSize))

    def GetSyntaxColorAttributes(self):
        self.lastPos = 0
        self.attrs = []
        try:
            tokenize.tokenize(self.GetNextLine, self._ProcessToken)
        except tokenize.TokenError:
            pass  # Ignore - will cause all subsequent text to be commented.
        numAtEnd = len(self.GetText()) - self.lastPos
        if numAtEnd:
            self.attrs.append((axdebug.SOURCETEXT_ATTR_COMMENT, numAtEnd))
        return self.attrs

    # We also provide and manage DebugDocumentContext objects
    def _MakeDebugCodeContext(self, lineNo, charPos, len):
        return _wrap(
            contexts.DebugCodeContext(lineNo, charPos, len, self, self.site),
            axdebug.IID_IDebugCodeContext,
        )

    # Make a context at the given position.  It should take up the entire context.
    def _MakeContextAtPosition(self, charPos):
        lineNo, offset = self.GetLineOfPosition(charPos)
        try:
            endPos = self.GetPositionOfLine(lineNo + 1)
        except:
            endPos = charPos
        codecontext = self._MakeDebugCodeContext(lineNo, charPos, endPos - charPos)
        return codecontext

    # Returns a DebugCodeContext.  debugDocument can be None for smart hosts.
    def GetCodeContextAtPosition(self, charPos):
        #               trace("GetContextOfPos", charPos, maxChars)
        # Convert to line number.
        lineNo, offset = self.GetLineOfPosition(charPos)
        charPos = self.GetPositionOfLine(lineNo)
        try:
            cc = self.codeContexts[charPos]
        #                       trace(" GetContextOfPos using existing")
        except KeyError:
            cc = self._MakeContextAtPosition(charPos)
            self.codeContexts[charPos] = cc
        return cc


class SourceModuleContainer(SourceCodeContainer):
    def __init__(self, module):
        self.module = module
        if hasattr(module, "__file__"):
            fname = self.module.__file__
            # Check for .pyc or .pyo or even .pys!
            if fname[-1] in ["O", "o", "C", "c", "S", "s"]:
                fname = fname[:-1]
            try:
                fname = win32api.GetFullPathName(fname)
            except win32api.error:
                pass
        else:
            if module.__name__ == "__main__" and len(sys.argv) > 0:
                fname = sys.argv[0]
            else:
                fname = "<Unknown!>"
        SourceCodeContainer.__init__(self, None, fname)

    def GetText(self):
        if self.text is None:
            fname = self.GetFileName()
            if fname:
                try:
                    self.text = open(fname, "r").read()
                except IOError as details:
                    self.text = "# Exception opening file\n# %s" % (repr(details))
            else:
                self.text = "# No file available for module '%s'" % (self.module)
            self._buildlines()
        return self.text

    def GetName(self, dnt):
        name = self.module.__name__
        try:
            fname = win32api.GetFullPathName(self.module.__file__)
        except win32api.error:
            fname = self.module.__file__
        except AttributeError:
            fname = name
        if dnt == axdebug.DOCUMENTNAMETYPE_APPNODE:
            return name.split(".")[-1]
        elif dnt == axdebug.DOCUMENTNAMETYPE_TITLE:
            return fname
        elif dnt == axdebug.DOCUMENTNAMETYPE_FILE_TAIL:
            return os.path.split(fname)[1]
        elif dnt == axdebug.DOCUMENTNAMETYPE_URL:
            return "file:%s" % fname
        else:
            raise Exception(scode=winerror.E_UNEXPECTED)


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    import ttest

    sc = SourceModuleContainer(ttest)
    #       sc = SourceCodeContainer(open(sys.argv[1], "rb").read(), sys.argv[1])
    attrs = sc.GetSyntaxColorAttributes()
    attrlen = 0
    for attr in attrs:
        if type(attr) == type(()):
            attrlen = attrlen + attr[1]
        else:
            attrlen = attrlen + 1
    text = sc.GetText()
    if attrlen != len(text):
        print("Lengths dont match!!! (%d/%d)" % (attrlen, len(text)))

    #       print "Attributes:"
    #       print attrs
    print("GetLineOfPos=", sc.GetLineOfPosition(0))
    print("GetLineOfPos=", sc.GetLineOfPosition(4))
    print("GetLineOfPos=", sc.GetLineOfPosition(10))
