# basic module browser.

# usage:
# >>> import browser
# >>> browser.Browse()
# or
# >>> browser.Browse(your_module)
import sys
import types

import __main__
import win32ui
from pywin.mfc import dialog

from . import hierlist

special_names = ["__doc__", "__name__", "__self__"]


#
# HierList items
class HLIPythonObject(hierlist.HierListItem):
    def __init__(self, myobject=None, name=None):
        hierlist.HierListItem.__init__(self)
        self.myobject = myobject
        self.knownExpandable = None
        if name:
            self.name = name
        else:
            try:
                self.name = myobject.__name__
            except (AttributeError, TypeError):
                try:
                    r = repr(myobject)
                    if len(r) > 20:
                        r = r[:20] + "..."
                    self.name = r
                except (AttributeError, TypeError):
                    self.name = "???"

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        try:
            type = self.GetHLIType()
        except:
            type = "Generic"
        return (
            "HLIPythonObject("
            + type
            + ") - name: "
            + self.name
            + " object: "
            + repr(self.myobject)
        )

    def GetText(self):
        try:
            return str(self.name) + " (" + self.GetHLIType() + ")"
        except AttributeError:
            return str(self.name) + " = " + repr(self.myobject)

    def InsertDocString(self, lst):
        ob = None
        try:
            ob = self.myobject.__doc__
        except (AttributeError, TypeError):
            pass
        # I don't quite grok descriptors enough to know how to
        # best hook them up. Eg:
        # >>> object.__getattribute__.__class__.__doc__
        # <attribute '__doc__' of 'wrapper_descriptor' objects>
        if ob and isinstance(ob, str):
            lst.insert(0, HLIDocString(ob, "Doc"))

    def GetSubList(self):
        ret = []
        try:
            for key, ob in self.myobject.__dict__.items():
                if key not in special_names:
                    ret.append(MakeHLI(ob, key))
        except (AttributeError, TypeError):
            pass
        try:
            for name in self.myobject.__methods__:
                ret.append(HLIMethod(name))  # no MakeHLI, as cant auto detect
        except (AttributeError, TypeError):
            pass
        try:
            for member in self.myobject.__members__:
                if not member in special_names:
                    ret.append(MakeHLI(getattr(self.myobject, member), member))
        except (AttributeError, TypeError):
            pass
        ret.sort()
        self.InsertDocString(ret)
        return ret

    # if the has a dict, it is expandable.
    def IsExpandable(self):
        if self.knownExpandable is None:
            self.knownExpandable = self.CalculateIsExpandable()
        return self.knownExpandable

    def CalculateIsExpandable(self):
        if hasattr(self.myobject, "__doc__"):
            return 1
        try:
            for key in self.myobject.__dict__.keys():
                if key not in special_names:
                    return 1
        except (AttributeError, TypeError):
            pass
        try:
            self.myobject.__methods__
            return 1
        except (AttributeError, TypeError):
            pass
        try:
            for item in self.myobject.__members__:
                if item not in special_names:
                    return 1
        except (AttributeError, TypeError):
            pass
        return 0

    def GetBitmapColumn(self):
        if self.IsExpandable():
            return 0
        else:
            return 4

    def TakeDefaultAction(self):
        ShowObject(self.myobject, self.name)


class HLIDocString(HLIPythonObject):
    def GetHLIType(self):
        return "DocString"

    def GetText(self):
        return self.myobject.strip()

    def IsExpandable(self):
        return 0

    def GetBitmapColumn(self):
        return 6


class HLIModule(HLIPythonObject):
    def GetHLIType(self):
        return "Module"


class HLIFrame(HLIPythonObject):
    def GetHLIType(self):
        return "Stack Frame"


class HLITraceback(HLIPythonObject):
    def GetHLIType(self):
        return "Traceback"


class HLIClass(HLIPythonObject):
    def GetHLIType(self):
        return "Class"

    def GetSubList(self):
        ret = []
        for base in self.myobject.__bases__:
            ret.append(MakeHLI(base, "Base class: " + base.__name__))
        ret = ret + HLIPythonObject.GetSubList(self)
        return ret


class HLIMethod(HLIPythonObject):
    # myobject is just a string for methods.
    def GetHLIType(self):
        return "Method"

    def GetText(self):
        return "Method: " + self.myobject + "()"


class HLICode(HLIPythonObject):
    def GetHLIType(self):
        return "Code"

    def IsExpandable(self):
        return self.myobject

    def GetSubList(self):
        ret = []
        ret.append(MakeHLI(self.myobject.co_consts, "Constants (co_consts)"))
        ret.append(MakeHLI(self.myobject.co_names, "Names (co_names)"))
        ret.append(MakeHLI(self.myobject.co_filename, "Filename (co_filename)"))
        ret.append(MakeHLI(self.myobject.co_argcount, "Number of args (co_argcount)"))
        ret.append(MakeHLI(self.myobject.co_varnames, "Param names (co_varnames)"))

        return ret


class HLIInstance(HLIPythonObject):
    def GetHLIType(self):
        return "Instance"

    def GetText(self):
        return (
            str(self.name)
            + " (Instance of class "
            + str(self.myobject.__class__.__name__)
            + ")"
        )

    def IsExpandable(self):
        return 1

    def GetSubList(self):
        ret = []
        ret.append(MakeHLI(self.myobject.__class__))
        ret = ret + HLIPythonObject.GetSubList(self)
        return ret


class HLIBuiltinFunction(HLIPythonObject):
    def GetHLIType(self):
        return "Builtin Function"


class HLIFunction(HLIPythonObject):
    def GetHLIType(self):
        return "Function"

    def IsExpandable(self):
        return 1

    def GetSubList(self):
        ret = []
        # 		ret.append( MakeHLI( self.myobject.func_argcount, "Arg Count" ))
        try:
            ret.append(MakeHLI(self.myobject.func_argdefs, "Arg Defs"))
        except AttributeError:
            pass
        try:
            code = self.myobject.__code__
            globs = self.myobject.__globals__
        except AttributeError:
            # must be py2.5 or earlier...
            code = self.myobject.func_code
            globs = self.myobject.func_globals
        ret.append(MakeHLI(code, "Code"))
        ret.append(MakeHLI(globs, "Globals"))
        self.InsertDocString(ret)
        return ret


class HLISeq(HLIPythonObject):
    def GetHLIType(self):
        return "Sequence (abstract!)"

    def IsExpandable(self):
        return len(self.myobject) > 0

    def GetSubList(self):
        ret = []
        pos = 0
        for item in self.myobject:
            ret.append(MakeHLI(item, "[" + str(pos) + "]"))
            pos = pos + 1
        self.InsertDocString(ret)
        return ret


class HLIList(HLISeq):
    def GetHLIType(self):
        return "List"


class HLITuple(HLISeq):
    def GetHLIType(self):
        return "Tuple"


class HLIDict(HLIPythonObject):
    def GetHLIType(self):
        return "Dict"

    def IsExpandable(self):
        try:
            self.myobject.__doc__
            return 1
        except (AttributeError, TypeError):
            return len(self.myobject) > 0

    def GetSubList(self):
        ret = []
        keys = list(self.myobject.keys())
        keys.sort()
        for key in keys:
            ob = self.myobject[key]
            ret.append(MakeHLI(ob, str(key)))
        self.InsertDocString(ret)
        return ret


# In Python 1.6, strings and Unicode have builtin methods, but we dont really want to see these
class HLIString(HLIPythonObject):
    def IsExpandable(self):
        return 0


TypeMap = {
    type: HLIClass,
    types.FunctionType: HLIFunction,
    tuple: HLITuple,
    dict: HLIDict,
    list: HLIList,
    types.ModuleType: HLIModule,
    types.CodeType: HLICode,
    types.BuiltinFunctionType: HLIBuiltinFunction,
    types.FrameType: HLIFrame,
    types.TracebackType: HLITraceback,
    str: HLIString,
    int: HLIPythonObject,
    bool: HLIPythonObject,
    float: HLIPythonObject,
}


def MakeHLI(ob, name=None):
    try:
        cls = TypeMap[type(ob)]
    except KeyError:
        # hrmph - this check gets more and more bogus as Python
        # improves.  Its possible we should just *always* use
        # HLIInstance?
        if hasattr(ob, "__class__"):  # 'new style' class
            cls = HLIInstance
        else:
            cls = HLIPythonObject
    return cls(ob, name)


#########################################
#
# Dialog related.


class DialogShowObject(dialog.Dialog):
    def __init__(self, object, title):
        self.object = object
        self.title = title
        dialog.Dialog.__init__(self, win32ui.IDD_LARGE_EDIT)

    def OnInitDialog(self):
        import re

        self.SetWindowText(self.title)
        self.edit = self.GetDlgItem(win32ui.IDC_EDIT1)
        try:
            strval = str(self.object)
        except:
            t, v, tb = sys.exc_info()
            strval = "Exception getting object value\n\n%s:%s" % (t, v)
            tb = None
        strval = re.sub("\n", "\r\n", strval)
        self.edit.ReplaceSel(strval)


def ShowObject(object, title):
    dlg = DialogShowObject(object, title)
    dlg.DoModal()


# And some mods for a sizable dialog from Sam Rushing!
import commctrl
import win32api
import win32con


class dynamic_browser(dialog.Dialog):
    style = win32con.WS_OVERLAPPEDWINDOW | win32con.WS_VISIBLE
    cs = (
        win32con.WS_CHILD
        | win32con.WS_VISIBLE
        | commctrl.TVS_HASLINES
        | commctrl.TVS_LINESATROOT
        | commctrl.TVS_HASBUTTONS
    )

    dt = [
        ["Python Object Browser", (0, 0, 200, 200), style, None, (8, "MS Sans Serif")],
        ["SysTreeView32", None, win32ui.IDC_LIST1, (0, 0, 200, 200), cs],
    ]

    def __init__(self, hli_root):
        dialog.Dialog.__init__(self, self.dt)
        self.hier_list = hierlist.HierListWithItems(hli_root, win32ui.IDB_BROWSER_HIER)
        self.HookMessage(self.on_size, win32con.WM_SIZE)

    def OnInitDialog(self):
        self.hier_list.HierInit(self)
        return dialog.Dialog.OnInitDialog(self)

    def OnOK(self):
        self.hier_list.HierTerm()
        self.hier_list = None
        return self._obj_.OnOK()

    def OnCancel(self):
        self.hier_list.HierTerm()
        self.hier_list = None
        return self._obj_.OnCancel()

    def on_size(self, params):
        lparam = params[3]
        w = win32api.LOWORD(lparam)
        h = win32api.HIWORD(lparam)
        self.GetDlgItem(win32ui.IDC_LIST1).MoveWindow((0, 0, w, h))


def Browse(ob=__main__):
    "Browse the argument, or the main dictionary"
    root = MakeHLI(ob, "root")
    if not root.IsExpandable():
        raise TypeError(
            "Browse() argument must have __dict__ attribute, or be a Browser supported type"
        )

    dlg = dynamic_browser(root)
    dlg.CreateWindow()
    return dlg


#
#
# Classes for using the browser in an MDI window, rather than a dialog
#
from pywin.mfc import docview


class BrowserTemplate(docview.DocTemplate):
    def __init__(self):
        docview.DocTemplate.__init__(
            self, win32ui.IDR_PYTHONTYPE, BrowserDocument, None, BrowserView
        )

    def OpenObject(self, root):  # Use this instead of OpenDocumentFile.
        # Look for existing open document
        for doc in self.GetDocumentList():
            if doc.root == root:
                doc.GetFirstView().ActivateFrame()
                return doc
        # not found - new one.
        doc = BrowserDocument(self, root)
        frame = self.CreateNewFrame(doc)
        doc.OnNewDocument()
        self.InitialUpdateFrame(frame, doc, 1)
        return doc


class BrowserDocument(docview.Document):
    def __init__(self, template, root):
        docview.Document.__init__(self, template)
        self.root = root
        self.SetTitle("Browser: " + root.name)

    def OnOpenDocument(self, name):
        raise TypeError("This template can not open files")
        return 0


class BrowserView(docview.TreeView):
    def OnInitialUpdate(self):
        import commctrl

        rc = self._obj_.OnInitialUpdate()
        list = hierlist.HierListWithItems(
            self.GetDocument().root,
            win32ui.IDB_BROWSER_HIER,
            win32ui.AFX_IDW_PANE_FIRST,
        )
        list.HierInit(self.GetParent())
        list.SetStyle(
            commctrl.TVS_HASLINES | commctrl.TVS_LINESATROOT | commctrl.TVS_HASBUTTONS
        )
        return rc


template = None


def MakeTemplate():
    global template
    if template is None:
        template = (
            BrowserTemplate()
        )  # win32ui.IDR_PYTHONTYPE, BrowserDocument, None, BrowserView)


def BrowseMDI(ob=__main__):
    """Browse an object using an MDI window."""

    MakeTemplate()
    root = MakeHLI(ob, repr(ob))
    if not root.IsExpandable():
        raise TypeError(
            "Browse() argument must have __dict__ attribute, or be a Browser supported type"
        )

    template.OpenObject(root)
