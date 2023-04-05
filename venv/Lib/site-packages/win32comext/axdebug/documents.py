""" Management of documents for AXDebugging.
"""


import pythoncom
import win32api
from win32com.server.exception import Exception
from win32com.server.util import unwrap

from . import axdebug, codecontainer, contexts, gateways
from .util import RaiseNotImpl, _wrap, _wrap_remove, trace

# def trace(*args):
#       pass


def GetGoodFileName(fname):
    if fname[0] != "<":
        return win32api.GetFullPathName(fname)
    return fname


class DebugDocumentProvider(gateways.DebugDocumentProvider):
    def __init__(self, doc):
        self.doc = doc

    def GetName(self, dnt):
        return self.doc.GetName(dnt)

    def GetDocumentClassId(self):
        return self.doc.GetDocumentClassId()

    def GetDocument(self):
        return self.doc


class DebugDocumentText(
    gateways.DebugDocumentInfo, gateways.DebugDocumentText, gateways.DebugDocument
):
    _com_interfaces_ = (
        gateways.DebugDocumentInfo._com_interfaces_
        + gateways.DebugDocumentText._com_interfaces_
        + gateways.DebugDocument._com_interfaces_
    )
    _public_methods_ = (
        gateways.DebugDocumentInfo._public_methods_
        + gateways.DebugDocumentText._public_methods_
        + gateways.DebugDocument._public_methods_
    )

    # A class which implements a DebugDocumentText, using the functionality
    # provided by a codeContainer
    def __init__(self, codeContainer):
        gateways.DebugDocumentText.__init__(self)
        gateways.DebugDocumentInfo.__init__(self)
        gateways.DebugDocument.__init__(self)
        self.codeContainer = codeContainer

    def _Close(self):
        self.docContexts = None
        #               self.codeContainer._Close()
        self.codeContainer = None

    # IDebugDocumentInfo
    def GetName(self, dnt):
        return self.codeContainer.GetName(dnt)

    def GetDocumentClassId(self):
        return "{DF630910-1C1D-11d0-AE36-8C0F5E000000}"

    # IDebugDocument has no methods!
    #

    # IDebugDocumentText methods.
    # def GetDocumentAttributes
    def GetSize(self):
        #               trace("GetSize")
        return self.codeContainer.GetNumLines(), self.codeContainer.GetNumChars()

    def GetPositionOfLine(self, cLineNumber):
        return self.codeContainer.GetPositionOfLine(cLineNumber)

    def GetLineOfPosition(self, charPos):
        return self.codeContainer.GetLineOfPosition(charPos)

    def GetText(self, charPos, maxChars, wantAttr):
        # Get all the attributes, else the tokenizer will get upset.
        # XXX - not yet!
        #               trace("GetText", charPos, maxChars, wantAttr)
        cont = self.codeContainer
        attr = cont.GetSyntaxColorAttributes()
        return cont.GetText(), attr

    def GetPositionOfContext(self, context):
        trace("GetPositionOfContext", context)
        context = unwrap(context)
        return context.offset, context.length

    # Return a DebugDocumentContext.
    def GetContextOfPosition(self, charPos, maxChars):
        # Make one
        doc = _wrap(self, axdebug.IID_IDebugDocument)
        rc = self.codeContainer.GetCodeContextAtPosition(charPos)
        return rc.QueryInterface(axdebug.IID_IDebugDocumentContext)


class CodeContainerProvider:
    """An abstract Python class which provides code containers!

    Given a Python file name (as the debugger knows it by) this will
    return a CodeContainer interface suitable for use.

    This provides a simple base imlpementation that simply supports
    a dictionary of nodes and providers.
    """

    def __init__(self):
        self.ccsAndNodes = {}

    def AddCodeContainer(self, cc, node=None):
        fname = GetGoodFileName(cc.fileName)
        self.ccsAndNodes[fname] = cc, node

    def FromFileName(self, fname):
        cc, node = self.ccsAndNodes.get(GetGoodFileName(fname), (None, None))
        #               if cc is None:
        #                       print "FromFileName for %s returning None" % fname
        return cc

    def Close(self):
        for cc, node in self.ccsAndNodes.values():
            try:
                # Must close the node before closing the provider
                # as node may make calls on provider (eg Reset breakpoints etc)
                if node is not None:
                    node.Close()
                cc._Close()
            except pythoncom.com_error:
                pass
        self.ccsAndNodes = {}
