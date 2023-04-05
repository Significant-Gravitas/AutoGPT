import os
import string
import sys

import pythoncom
import win32api
from win32com.axdebug import (
    adb,
    axdebug,
    codecontainer,
    contexts,
    documents,
    expressions,
    gateways,
)
from win32com.axdebug.util import _wrap, _wrap_remove, trace
from win32com.axscript import axscript

currentDebugger = None


class ModuleTreeNode:
    """Helper class for building a module tree"""

    def __init__(self, module):
        modName = module.__name__
        self.moduleName = modName
        self.module = module
        self.realNode = None
        self.cont = codecontainer.SourceModuleContainer(module)

    def __repr__(self):
        return "<ModuleTreeNode wrapping %s>" % (self.module)

    def Attach(self, parentRealNode):
        self.realNode.Attach(parentRealNode)

    def Close(self):
        self.module = None
        self.cont = None
        self.realNode = None


def BuildModule(module, built_nodes, rootNode, create_node_fn, create_node_args):
    if module:
        keep = module.__name__
        keep = keep and (built_nodes.get(module) is None)
        if keep and hasattr(module, "__file__"):
            keep = string.lower(os.path.splitext(module.__file__)[1]) not in [
                ".pyd",
                ".dll",
            ]
    #               keep = keep and module.__name__=='__main__'
    if module and keep:
        #        print "keeping", module.__name__
        node = ModuleTreeNode(module)
        built_nodes[module] = node
        realNode = create_node_fn(*(node,) + create_node_args)
        node.realNode = realNode

        # Split into parent nodes.
        parts = string.split(module.__name__, ".")
        if parts[-1][:8] == "__init__":
            parts = parts[:-1]
        parent = string.join(parts[:-1], ".")
        parentNode = rootNode
        if parent:
            parentModule = sys.modules[parent]
            BuildModule(
                parentModule, built_nodes, rootNode, create_node_fn, create_node_args
            )
            if parentModule in built_nodes:
                parentNode = built_nodes[parentModule].realNode
        node.Attach(parentNode)


def RefreshAllModules(builtItems, rootNode, create_node, create_node_args):
    for module in list(sys.modules.values()):
        BuildModule(module, builtItems, rootNode, create_node, create_node_args)


# realNode = pdm.CreateDebugDocumentHelper(None) # DebugDocumentHelper node?
# app.CreateApplicationNode() # doc provider node.


class CodeContainerProvider(documents.CodeContainerProvider):
    def __init__(self, axdebugger):
        self.axdebugger = axdebugger
        documents.CodeContainerProvider.__init__(self)
        self.currentNumModules = len(sys.modules)
        self.nodes = {}
        self.axdebugger.RefreshAllModules(self.nodes, self)

    def FromFileName(self, fname):
        ### It appears we cant add modules during a debug session!
        #               if self.currentNumModules != len(sys.modules):
        #                       self.axdebugger.RefreshAllModules(self.nodes, self)
        #                       self.currentNumModules = len(sys.modules)
        #               for key in self.ccsAndNodes.keys():
        #                       print "File:", key
        return documents.CodeContainerProvider.FromFileName(self, fname)

    def Close(self):
        documents.CodeContainerProvider.Close(self)
        self.axdebugger = None
        print("Closing %d nodes" % (len(self.nodes)))
        for node in self.nodes.values():
            node.Close()
        self.nodes = {}


class OriginalInterfaceMaker:
    def MakeInterfaces(self, pdm):
        app = self.pdm.CreateApplication()
        self.cookie = pdm.AddApplication(app)
        root = app.GetRootNode()
        return app, root

    def CloseInterfaces(self, pdm):
        pdm.RemoveApplication(self.cookie)


class SimpleHostStyleInterfaceMaker:
    def MakeInterfaces(self, pdm):
        app = pdm.GetDefaultApplication()
        root = app.GetRootNode()
        return app, root

    def CloseInterfaces(self, pdm):
        pass


class AXDebugger:
    def __init__(self, interfaceMaker=None, processName=None):
        if processName is None:
            processName = "Python Process"
        if interfaceMaker is None:
            interfaceMaker = SimpleHostStyleInterfaceMaker()

        self.pydebugger = adb.Debugger()

        self.pdm = pythoncom.CoCreateInstance(
            axdebug.CLSID_ProcessDebugManager,
            None,
            pythoncom.CLSCTX_ALL,
            axdebug.IID_IProcessDebugManager,
        )

        self.app, self.root = interfaceMaker.MakeInterfaces(self.pdm)
        self.app.SetName(processName)
        self.interfaceMaker = interfaceMaker

        expressionProvider = _wrap(
            expressions.ProvideExpressionContexts(),
            axdebug.IID_IProvideExpressionContexts,
        )
        self.expressionCookie = self.app.AddGlobalExpressionContextProvider(
            expressionProvider
        )

        contProvider = CodeContainerProvider(self)
        self.pydebugger.AttachApp(self.app, contProvider)

    def Break(self):
        # Get the frame we start debugging from - this is the frame 1 level up
        try:
            1 + ""
        except:
            frame = sys.exc_info()[2].tb_frame.f_back

        # Get/create the debugger, and tell it to break.
        self.app.StartDebugSession()
        #               self.app.CauseBreak()

        self.pydebugger.SetupAXDebugging(None, frame)
        self.pydebugger.set_trace()

    def Close(self):
        self.pydebugger.ResetAXDebugging()
        self.interfaceMaker.CloseInterfaces(self.pdm)
        self.pydebugger.CloseApp()
        self.app.RemoveGlobalExpressionContextProvider(self.expressionCookie)
        self.expressionCookie = None

        self.pdm = None
        self.app = None
        self.pydebugger = None
        self.root = None

    def RefreshAllModules(self, nodes, containerProvider):
        RefreshAllModules(
            nodes, self.root, self.CreateApplicationNode, (containerProvider,)
        )

    def CreateApplicationNode(self, node, containerProvider):
        realNode = self.app.CreateApplicationNode()

        document = documents.DebugDocumentText(node.cont)
        document = _wrap(document, axdebug.IID_IDebugDocument)

        node.cont.debugDocument = document

        provider = documents.DebugDocumentProvider(document)
        provider = _wrap(provider, axdebug.IID_IDebugDocumentProvider)
        realNode.SetDocumentProvider(provider)

        containerProvider.AddCodeContainer(node.cont, realNode)
        return realNode


def _GetCurrentDebugger():
    global currentDebugger
    if currentDebugger is None:
        currentDebugger = AXDebugger()
    return currentDebugger


def Break():
    _GetCurrentDebugger().Break()


brk = Break
set_trace = Break


def dosomethingelse():
    a = 2
    b = "Hi there"


def dosomething():
    a = 1
    b = 2
    dosomethingelse()


def test():
    Break()
    input("Waiting...")
    dosomething()
    print("Done")


if __name__ == "__main__":
    print("About to test the debugging interfaces!")
    test()
    print(
        " %d/%d com objects still alive"
        % (pythoncom._GetInterfaceCount(), pythoncom._GetGatewayCount())
    )
