"""A utility for browsing COM objects.

 Usage:

  Command Prompt

    Use the command *"python.exe combrowse.py"*.  This will display
    display a fairly small, modal dialog.

  Pythonwin

    Use the "Run Script" menu item, and this will create the browser in an
    MDI window.  This window can be fully resized.

 Details

   This module allows browsing of registered Type Libraries, COM categories,
   and running COM objects.  The display is similar to the Pythonwin object
   browser, and displays the objects in a hierarchical window.

   Note that this module requires the win32ui (ie, Pythonwin) distribution to
   work.

"""
import sys

import pythoncom
import win32api
import win32con
import win32ui
from pywin.tools import browser
from win32com.client import util


class HLIRoot(browser.HLIPythonObject):
    def __init__(self, title):
        super().__init__(name=title)

    def GetSubList(self):
        return [
            HLIHeadingCategory(),
            HLI_IEnumMoniker(
                pythoncom.GetRunningObjectTable().EnumRunning(), "Running Objects"
            ),
            HLIHeadingRegisterdTypeLibs(),
        ]

    def __cmp__(self, other):
        return cmp(self.name, other.name)


class HLICOM(browser.HLIPythonObject):
    def GetText(self):
        return self.name

    def CalculateIsExpandable(self):
        return 1


class HLICLSID(HLICOM):
    def __init__(self, myobject, name=None):
        if type(myobject) == type(""):
            myobject = pythoncom.MakeIID(myobject)
        if name is None:
            try:
                name = pythoncom.ProgIDFromCLSID(myobject)
            except pythoncom.com_error:
                name = str(myobject)
            name = "IID: " + name
        HLICOM.__init__(self, myobject, name)

    def CalculateIsExpandable(self):
        return 0

    def GetSubList(self):
        return []


class HLI_Interface(HLICOM):
    pass


class HLI_Enum(HLI_Interface):
    def GetBitmapColumn(self):
        return 0  # Always a folder.

    def CalculateIsExpandable(self):
        if self.myobject is not None:
            rc = len(self.myobject.Next(1)) > 0
            self.myobject.Reset()
        else:
            rc = 0
        return rc

    pass


class HLI_IEnumMoniker(HLI_Enum):
    def GetSubList(self):
        ctx = pythoncom.CreateBindCtx()
        ret = []
        for mon in util.Enumerator(self.myobject):
            ret.append(HLI_IMoniker(mon, mon.GetDisplayName(ctx, None)))
        return ret


class HLI_IMoniker(HLI_Interface):
    def GetSubList(self):
        ret = []
        ret.append(browser.MakeHLI(self.myobject.Hash(), "Hash Value"))
        subenum = self.myobject.Enum(1)
        ret.append(HLI_IEnumMoniker(subenum, "Sub Monikers"))
        return ret


class HLIHeadingCategory(HLICOM):
    "A tree heading for registered categories"

    def GetText(self):
        return "Registered Categories"

    def GetSubList(self):
        catinf = pythoncom.CoCreateInstance(
            pythoncom.CLSID_StdComponentCategoriesMgr,
            None,
            pythoncom.CLSCTX_INPROC,
            pythoncom.IID_ICatInformation,
        )
        enum = util.Enumerator(catinf.EnumCategories())
        ret = []
        try:
            for catid, lcid, desc in enum:
                ret.append(HLICategory((catid, lcid, desc)))
        except pythoncom.com_error:
            # Registered categories occasionally seem to give spurious errors.
            pass  # Use what we already have.
        return ret


class HLICategory(HLICOM):
    "An actual Registered Category"

    def GetText(self):
        desc = self.myobject[2]
        if not desc:
            desc = "(unnamed category)"
        return desc

    def GetSubList(self):
        win32ui.DoWaitCursor(1)
        catid, lcid, desc = self.myobject
        catinf = pythoncom.CoCreateInstance(
            pythoncom.CLSID_StdComponentCategoriesMgr,
            None,
            pythoncom.CLSCTX_INPROC,
            pythoncom.IID_ICatInformation,
        )
        ret = []
        for clsid in util.Enumerator(catinf.EnumClassesOfCategories((catid,), ())):
            ret.append(HLICLSID(clsid))
        win32ui.DoWaitCursor(0)

        return ret


class HLIHelpFile(HLICOM):
    def CalculateIsExpandable(self):
        return 0

    def GetText(self):
        import os

        fname, ctx = self.myobject
        base = os.path.split(fname)[1]
        return "Help reference in %s" % (base)

    def TakeDefaultAction(self):
        fname, ctx = self.myobject
        if ctx:
            cmd = win32con.HELP_CONTEXT
        else:
            cmd = win32con.HELP_FINDER
        win32api.WinHelp(win32ui.GetMainFrame().GetSafeHwnd(), fname, cmd, ctx)

    def GetBitmapColumn(self):
        return 6


class HLIRegisteredTypeLibrary(HLICOM):
    def GetSubList(self):
        import os

        clsidstr, versionStr = self.myobject
        collected = []
        helpPath = ""
        key = win32api.RegOpenKey(
            win32con.HKEY_CLASSES_ROOT, "TypeLib\\%s\\%s" % (clsidstr, versionStr)
        )
        win32ui.DoWaitCursor(1)
        try:
            num = 0
            while 1:
                try:
                    subKey = win32api.RegEnumKey(key, num)
                except win32api.error:
                    break
                hSubKey = win32api.RegOpenKey(key, subKey)
                try:
                    value, typ = win32api.RegQueryValueEx(hSubKey, None)
                    if typ == win32con.REG_EXPAND_SZ:
                        value = win32api.ExpandEnvironmentStrings(value)
                except win32api.error:
                    value = ""
                if subKey == "HELPDIR":
                    helpPath = value
                elif subKey == "Flags":
                    flags = value
                else:
                    try:
                        lcid = int(subKey)
                        lcidkey = win32api.RegOpenKey(key, subKey)
                        # Enumerate the platforms
                        lcidnum = 0
                        while 1:
                            try:
                                platform = win32api.RegEnumKey(lcidkey, lcidnum)
                            except win32api.error:
                                break
                            try:
                                hplatform = win32api.RegOpenKey(lcidkey, platform)
                                fname, typ = win32api.RegQueryValueEx(hplatform, None)
                                if typ == win32con.REG_EXPAND_SZ:
                                    fname = win32api.ExpandEnvironmentStrings(fname)
                            except win32api.error:
                                fname = ""
                            collected.append((lcid, platform, fname))
                            lcidnum = lcidnum + 1
                        win32api.RegCloseKey(lcidkey)
                    except ValueError:
                        pass
                num = num + 1
        finally:
            win32ui.DoWaitCursor(0)
            win32api.RegCloseKey(key)
        # Now, loop over my collected objects, adding a TypeLib and a HelpFile
        ret = []
        #               if helpPath: ret.append(browser.MakeHLI(helpPath, "Help Path"))
        ret.append(HLICLSID(clsidstr))
        for lcid, platform, fname in collected:
            extraDescs = []
            if platform != "win32":
                extraDescs.append(platform)
            if lcid:
                extraDescs.append("locale=%s" % lcid)
            extraDesc = ""
            if extraDescs:
                extraDesc = " (%s)" % ", ".join(extraDescs)
            ret.append(HLITypeLib(fname, "Type Library" + extraDesc))
        ret.sort()
        return ret


class HLITypeLibEntry(HLICOM):
    def GetText(self):
        tlb, index = self.myobject
        name, doc, ctx, helpFile = tlb.GetDocumentation(index)
        try:
            typedesc = HLITypeKinds[tlb.GetTypeInfoType(index)][1]
        except KeyError:
            typedesc = "Unknown!"
        return name + " - " + typedesc

    def GetSubList(self):
        tlb, index = self.myobject
        name, doc, ctx, helpFile = tlb.GetDocumentation(index)
        ret = []
        if doc:
            ret.append(browser.HLIDocString(doc, "Doc"))
        if helpFile:
            ret.append(HLIHelpFile((helpFile, ctx)))
        return ret


class HLICoClass(HLITypeLibEntry):
    def GetSubList(self):
        ret = HLITypeLibEntry.GetSubList(self)
        tlb, index = self.myobject
        typeinfo = tlb.GetTypeInfo(index)
        attr = typeinfo.GetTypeAttr()
        for j in range(attr[8]):
            flags = typeinfo.GetImplTypeFlags(j)
            refType = typeinfo.GetRefTypeInfo(typeinfo.GetRefTypeOfImplType(j))
            refAttr = refType.GetTypeAttr()
            ret.append(
                browser.MakeHLI(refAttr[0], "Name=%s, Flags = %d" % (refAttr[0], flags))
            )
        return ret


class HLITypeLibMethod(HLITypeLibEntry):
    def __init__(self, ob, name=None):
        self.entry_type = "Method"
        HLITypeLibEntry.__init__(self, ob, name)

    def GetSubList(self):
        ret = HLITypeLibEntry.GetSubList(self)
        tlb, index = self.myobject
        typeinfo = tlb.GetTypeInfo(index)
        attr = typeinfo.GetTypeAttr()
        for i in range(attr[7]):
            ret.append(HLITypeLibProperty((typeinfo, i)))
        for i in range(attr[6]):
            ret.append(HLITypeLibFunction((typeinfo, i)))
        return ret


class HLITypeLibEnum(HLITypeLibEntry):
    def __init__(self, myitem):
        typelib, index = myitem
        typeinfo = typelib.GetTypeInfo(index)
        self.id = typeinfo.GetVarDesc(index)[0]
        name = typeinfo.GetNames(self.id)[0]
        HLITypeLibEntry.__init__(self, myitem, name)

    def GetText(self):
        return self.name + " - Enum/Module"

    def GetSubList(self):
        ret = []
        typelib, index = self.myobject
        typeinfo = typelib.GetTypeInfo(index)
        attr = typeinfo.GetTypeAttr()
        for j in range(attr[7]):
            vdesc = typeinfo.GetVarDesc(j)
            name = typeinfo.GetNames(vdesc[0])[0]
            ret.append(browser.MakeHLI(vdesc[1], name))
        return ret


class HLITypeLibProperty(HLICOM):
    def __init__(self, myitem):
        typeinfo, index = myitem
        self.id = typeinfo.GetVarDesc(index)[0]
        name = typeinfo.GetNames(self.id)[0]
        HLICOM.__init__(self, myitem, name)

    def GetText(self):
        return self.name + " - Property"

    def GetSubList(self):
        ret = []
        typeinfo, index = self.myobject
        names = typeinfo.GetNames(self.id)
        if len(names) > 1:
            ret.append(browser.MakeHLI(names[1:], "Named Params"))
        vd = typeinfo.GetVarDesc(index)
        ret.append(browser.MakeHLI(self.id, "Dispatch ID"))
        ret.append(browser.MakeHLI(vd[1], "Value"))
        ret.append(browser.MakeHLI(vd[2], "Elem Desc"))
        ret.append(browser.MakeHLI(vd[3], "Var Flags"))
        ret.append(browser.MakeHLI(vd[4], "Var Kind"))
        return ret


class HLITypeLibFunction(HLICOM):
    funckinds = {
        pythoncom.FUNC_VIRTUAL: "Virtual",
        pythoncom.FUNC_PUREVIRTUAL: "Pure Virtual",
        pythoncom.FUNC_STATIC: "Static",
        pythoncom.FUNC_DISPATCH: "Dispatch",
    }
    invokekinds = {
        pythoncom.INVOKE_FUNC: "Function",
        pythoncom.INVOKE_PROPERTYGET: "Property Get",
        pythoncom.INVOKE_PROPERTYPUT: "Property Put",
        pythoncom.INVOKE_PROPERTYPUTREF: "Property Put by reference",
    }
    funcflags = [
        (pythoncom.FUNCFLAG_FRESTRICTED, "Restricted"),
        (pythoncom.FUNCFLAG_FSOURCE, "Source"),
        (pythoncom.FUNCFLAG_FBINDABLE, "Bindable"),
        (pythoncom.FUNCFLAG_FREQUESTEDIT, "Request Edit"),
        (pythoncom.FUNCFLAG_FDISPLAYBIND, "Display Bind"),
        (pythoncom.FUNCFLAG_FDEFAULTBIND, "Default Bind"),
        (pythoncom.FUNCFLAG_FHIDDEN, "Hidden"),
        (pythoncom.FUNCFLAG_FUSESGETLASTERROR, "Uses GetLastError"),
    ]

    vartypes = {
        pythoncom.VT_EMPTY: "Empty",
        pythoncom.VT_NULL: "NULL",
        pythoncom.VT_I2: "Integer 2",
        pythoncom.VT_I4: "Integer 4",
        pythoncom.VT_R4: "Real 4",
        pythoncom.VT_R8: "Real 8",
        pythoncom.VT_CY: "CY",
        pythoncom.VT_DATE: "Date",
        pythoncom.VT_BSTR: "String",
        pythoncom.VT_DISPATCH: "IDispatch",
        pythoncom.VT_ERROR: "Error",
        pythoncom.VT_BOOL: "BOOL",
        pythoncom.VT_VARIANT: "Variant",
        pythoncom.VT_UNKNOWN: "IUnknown",
        pythoncom.VT_DECIMAL: "Decimal",
        pythoncom.VT_I1: "Integer 1",
        pythoncom.VT_UI1: "Unsigned integer 1",
        pythoncom.VT_UI2: "Unsigned integer 2",
        pythoncom.VT_UI4: "Unsigned integer 4",
        pythoncom.VT_I8: "Integer 8",
        pythoncom.VT_UI8: "Unsigned integer 8",
        pythoncom.VT_INT: "Integer",
        pythoncom.VT_UINT: "Unsigned integer",
        pythoncom.VT_VOID: "Void",
        pythoncom.VT_HRESULT: "HRESULT",
        pythoncom.VT_PTR: "Pointer",
        pythoncom.VT_SAFEARRAY: "SafeArray",
        pythoncom.VT_CARRAY: "C Array",
        pythoncom.VT_USERDEFINED: "User Defined",
        pythoncom.VT_LPSTR: "Pointer to string",
        pythoncom.VT_LPWSTR: "Pointer to Wide String",
        pythoncom.VT_FILETIME: "File time",
        pythoncom.VT_BLOB: "Blob",
        pythoncom.VT_STREAM: "IStream",
        pythoncom.VT_STORAGE: "IStorage",
        pythoncom.VT_STORED_OBJECT: "Stored object",
        pythoncom.VT_STREAMED_OBJECT: "Streamed object",
        pythoncom.VT_BLOB_OBJECT: "Blob object",
        pythoncom.VT_CF: "CF",
        pythoncom.VT_CLSID: "CLSID",
    }

    type_flags = [
        (pythoncom.VT_VECTOR, "Vector"),
        (pythoncom.VT_ARRAY, "Array"),
        (pythoncom.VT_BYREF, "ByRef"),
        (pythoncom.VT_RESERVED, "Reserved"),
    ]

    def __init__(self, myitem):
        typeinfo, index = myitem
        self.id = typeinfo.GetFuncDesc(index)[0]
        name = typeinfo.GetNames(self.id)[0]
        HLICOM.__init__(self, myitem, name)

    def GetText(self):
        return self.name + " - Function"

    def MakeReturnTypeName(self, typ):
        justtyp = typ & pythoncom.VT_TYPEMASK
        try:
            typname = self.vartypes[justtyp]
        except KeyError:
            typname = "?Bad type?"
        for flag, desc in self.type_flags:
            if flag & typ:
                typname = "%s(%s)" % (desc, typname)
        return typname

    def MakeReturnType(self, returnTypeDesc):
        if type(returnTypeDesc) == type(()):
            first = returnTypeDesc[0]
            result = self.MakeReturnType(first)
            if first != pythoncom.VT_USERDEFINED:
                result = result + " " + self.MakeReturnType(returnTypeDesc[1])
            return result
        else:
            return self.MakeReturnTypeName(returnTypeDesc)

    def GetSubList(self):
        ret = []
        typeinfo, index = self.myobject
        names = typeinfo.GetNames(self.id)
        ret.append(browser.MakeHLI(self.id, "Dispatch ID"))
        if len(names) > 1:
            ret.append(browser.MakeHLI(", ".join(names[1:]), "Named Params"))
        fd = typeinfo.GetFuncDesc(index)
        if fd[1]:
            ret.append(browser.MakeHLI(fd[1], "Possible result values"))
        if fd[8]:
            typ, flags, default = fd[8]
            val = self.MakeReturnType(typ)
            if flags:
                val = "%s (Flags=%d, default=%s)" % (val, flags, default)
            ret.append(browser.MakeHLI(val, "Return Type"))

        for argDesc in fd[2]:
            typ, flags, default = argDesc
            val = self.MakeReturnType(typ)
            if flags:
                val = "%s (Flags=%d)" % (val, flags)
            if default is not None:
                val = "%s (Default=%s)" % (val, default)
            ret.append(browser.MakeHLI(val, "Argument"))

        try:
            fkind = self.funckinds[fd[3]]
        except KeyError:
            fkind = "Unknown"
        ret.append(browser.MakeHLI(fkind, "Function Kind"))
        try:
            ikind = self.invokekinds[fd[4]]
        except KeyError:
            ikind = "Unknown"
        ret.append(browser.MakeHLI(ikind, "Invoke Kind"))
        # 5 = call conv
        # 5 = offset vtbl
        ret.append(browser.MakeHLI(fd[6], "Number Optional Params"))
        flagDescs = []
        for flag, desc in self.funcflags:
            if flag & fd[9]:
                flagDescs.append(desc)
        if flagDescs:
            ret.append(browser.MakeHLI(", ".join(flagDescs), "Function Flags"))
        return ret


HLITypeKinds = {
    pythoncom.TKIND_ENUM: (HLITypeLibEnum, "Enumeration"),
    pythoncom.TKIND_RECORD: (HLITypeLibEntry, "Record"),
    pythoncom.TKIND_MODULE: (HLITypeLibEnum, "Module"),
    pythoncom.TKIND_INTERFACE: (HLITypeLibMethod, "Interface"),
    pythoncom.TKIND_DISPATCH: (HLITypeLibMethod, "Dispatch"),
    pythoncom.TKIND_COCLASS: (HLICoClass, "CoClass"),
    pythoncom.TKIND_ALIAS: (HLITypeLibEntry, "Alias"),
    pythoncom.TKIND_UNION: (HLITypeLibEntry, "Union"),
}


class HLITypeLib(HLICOM):
    def GetSubList(self):
        ret = []
        ret.append(browser.MakeHLI(self.myobject, "Filename"))
        try:
            tlb = pythoncom.LoadTypeLib(self.myobject)
        except pythoncom.com_error:
            return [browser.MakeHLI("%s can not be loaded" % self.myobject)]

        for i in range(tlb.GetTypeInfoCount()):
            try:
                ret.append(HLITypeKinds[tlb.GetTypeInfoType(i)][0]((tlb, i)))
            except pythoncom.com_error:
                ret.append(browser.MakeHLI("The type info can not be loaded!"))
        ret.sort()
        return ret


class HLIHeadingRegisterdTypeLibs(HLICOM):
    "A tree heading for registered type libraries"

    def GetText(self):
        return "Registered Type Libraries"

    def GetSubList(self):
        # Explicit lookup in the registry.
        ret = []
        key = win32api.RegOpenKey(win32con.HKEY_CLASSES_ROOT, "TypeLib")
        win32ui.DoWaitCursor(1)
        try:
            num = 0
            while 1:
                try:
                    keyName = win32api.RegEnumKey(key, num)
                except win32api.error:
                    break
                # Enumerate all version info
                subKey = win32api.RegOpenKey(key, keyName)
                name = None
                try:
                    subNum = 0
                    bestVersion = 0.0
                    while 1:
                        try:
                            versionStr = win32api.RegEnumKey(subKey, subNum)
                        except win32api.error:
                            break
                        try:
                            versionFlt = float(versionStr)
                        except ValueError:
                            versionFlt = 0  # ????
                        if versionFlt > bestVersion:
                            bestVersion = versionFlt
                            name = win32api.RegQueryValue(subKey, versionStr)
                        subNum = subNum + 1
                finally:
                    win32api.RegCloseKey(subKey)
                if name is not None:
                    ret.append(HLIRegisteredTypeLibrary((keyName, versionStr), name))
                num = num + 1
        finally:
            win32api.RegCloseKey(key)
            win32ui.DoWaitCursor(0)
        ret.sort()
        return ret


def main(modal=True, mdi=False):
    from pywin.tools import hierlist

    root = HLIRoot("COM Browser")
    if mdi and "pywin.framework.app" in sys.modules:
        # do it in a MDI window
        browser.MakeTemplate()
        browser.template.OpenObject(root)
    else:
        dlg = browser.dynamic_browser(root)
        if modal:
            dlg.DoModal()
        else:
            dlg.CreateWindow()
            dlg.ShowWindow()


if __name__ == "__main__":
    main(modal=win32api.GetConsoleTitle())

    ni = pythoncom._GetInterfaceCount()
    ng = pythoncom._GetGatewayCount()
    if ni or ng:
        print("Warning - exiting with %d/%d objects alive" % (ni, ng))
