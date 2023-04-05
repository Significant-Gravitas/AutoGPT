"""Support for dynamic COM client support.

Introduction
 Dynamic COM client support is the ability to use a COM server without
 prior knowledge of the server.  This can be used to talk to almost all
 COM servers, including much of MS Office.

 In general, you should not use this module directly - see below.

Example
 >>> import win32com.client
 >>> xl = win32com.client.Dispatch("Excel.Application")
 # The line above invokes the functionality of this class.
 # xl is now an object we can use to talk to Excel.
 >>> xl.Visible = 1 # The Excel window becomes visible.

"""
import traceback
import types

import pythoncom  # Needed as code we eval() references it.
import win32com.client
import winerror
from pywintypes import IIDType

from . import build

debugging = 0  # General debugging
debugging_attr = 0  # Debugging dynamic attribute lookups.

LCID = 0x0

# These errors generally mean the property or method exists,
# but can't be used in this context - eg, property instead of a method, etc.
# Used to determine if we have a real error or not.
ERRORS_BAD_CONTEXT = [
    winerror.DISP_E_MEMBERNOTFOUND,
    winerror.DISP_E_BADPARAMCOUNT,
    winerror.DISP_E_PARAMNOTOPTIONAL,
    winerror.DISP_E_TYPEMISMATCH,
    winerror.E_INVALIDARG,
]

ALL_INVOKE_TYPES = [
    pythoncom.INVOKE_PROPERTYGET,
    pythoncom.INVOKE_PROPERTYPUT,
    pythoncom.INVOKE_PROPERTYPUTREF,
    pythoncom.INVOKE_FUNC,
]


def debug_print(*args):
    if debugging:
        for arg in args:
            print(arg, end=" ")
        print()


def debug_attr_print(*args):
    if debugging_attr:
        for arg in args:
            print(arg, end=" ")
        print()


def MakeMethod(func, inst, cls):
    return types.MethodType(func, inst)


# get the type objects for IDispatch and IUnknown
PyIDispatchType = pythoncom.TypeIIDs[pythoncom.IID_IDispatch]
PyIUnknownType = pythoncom.TypeIIDs[pythoncom.IID_IUnknown]

_GoodDispatchTypes = (str, IIDType)
_defaultDispatchItem = build.DispatchItem


def _GetGoodDispatch(IDispatch, clsctx=pythoncom.CLSCTX_SERVER):
    # quick return for most common case
    if isinstance(IDispatch, PyIDispatchType):
        return IDispatch
    if isinstance(IDispatch, _GoodDispatchTypes):
        try:
            IDispatch = pythoncom.connect(IDispatch)
        except pythoncom.ole_error:
            IDispatch = pythoncom.CoCreateInstance(
                IDispatch, None, clsctx, pythoncom.IID_IDispatch
            )
    else:
        # may already be a wrapped class.
        IDispatch = getattr(IDispatch, "_oleobj_", IDispatch)
    return IDispatch


def _GetGoodDispatchAndUserName(IDispatch, userName, clsctx):
    # Get a dispatch object, and a 'user name' (ie, the name as
    # displayed to the user in repr() etc.
    if userName is None:
        if isinstance(IDispatch, str):
            userName = IDispatch
        ## ??? else userName remains None ???
    else:
        userName = str(userName)
    return (_GetGoodDispatch(IDispatch, clsctx), userName)


def _GetDescInvokeType(entry, invoke_type):
    # determine the wFlags argument passed as input to IDispatch::Invoke
    # Only ever called by __getattr__ and __setattr__ from dynamic objects!
    # * `entry` is a MapEntry with whatever typeinfo we have about the property we are getting/setting.
    # * `invoke_type` is either INVOKE_PROPERTYGET | INVOKE_PROPERTYSET and really just
    #   means "called by __getattr__" or "called by __setattr__"
    if not entry or not entry.desc:
        return invoke_type

    if entry.desc.desckind == pythoncom.DESCKIND_VARDESC:
        return invoke_type

    # So it's a FUNCDESC - just use what it specifies.
    return entry.desc.invkind


def Dispatch(
    IDispatch,
    userName=None,
    createClass=None,
    typeinfo=None,
    UnicodeToString=None,
    clsctx=pythoncom.CLSCTX_SERVER,
):
    assert UnicodeToString is None, "this is deprecated and will go away"
    IDispatch, userName = _GetGoodDispatchAndUserName(IDispatch, userName, clsctx)
    if createClass is None:
        createClass = CDispatch
    lazydata = None
    try:
        if typeinfo is None:
            typeinfo = IDispatch.GetTypeInfo()
        if typeinfo is not None:
            try:
                # try for a typecomp
                typecomp = typeinfo.GetTypeComp()
                lazydata = typeinfo, typecomp
            except pythoncom.com_error:
                pass
    except pythoncom.com_error:
        typeinfo = None
    olerepr = MakeOleRepr(IDispatch, typeinfo, lazydata)
    return createClass(IDispatch, olerepr, userName, lazydata=lazydata)


def MakeOleRepr(IDispatch, typeinfo, typecomp):
    olerepr = None
    if typeinfo is not None:
        try:
            attr = typeinfo.GetTypeAttr()
            # If the type info is a special DUAL interface, magically turn it into
            # a DISPATCH typeinfo.
            if (
                attr[5] == pythoncom.TKIND_INTERFACE
                and attr[11] & pythoncom.TYPEFLAG_FDUAL
            ):
                # Get corresponding Disp interface;
                # -1 is a special value which does this for us.
                href = typeinfo.GetRefTypeOfImplType(-1)
                typeinfo = typeinfo.GetRefTypeInfo(href)
                attr = typeinfo.GetTypeAttr()
            if typecomp is None:
                olerepr = build.DispatchItem(typeinfo, attr, None, 0)
            else:
                olerepr = build.LazyDispatchItem(attr, None)
        except pythoncom.ole_error:
            pass
    if olerepr is None:
        olerepr = build.DispatchItem()
    return olerepr


def DumbDispatch(
    IDispatch,
    userName=None,
    createClass=None,
    UnicodeToString=None,
    clsctx=pythoncom.CLSCTX_SERVER,
):
    "Dispatch with no type info"
    assert UnicodeToString is None, "this is deprecated and will go away"
    IDispatch, userName = _GetGoodDispatchAndUserName(IDispatch, userName, clsctx)
    if createClass is None:
        createClass = CDispatch
    return createClass(IDispatch, build.DispatchItem(), userName)


class CDispatch:
    def __init__(
        self, IDispatch, olerepr, userName=None, UnicodeToString=None, lazydata=None
    ):
        assert UnicodeToString is None, "this is deprecated and will go away"
        if userName is None:
            userName = "<unknown>"
        self.__dict__["_oleobj_"] = IDispatch
        self.__dict__["_username_"] = userName
        self.__dict__["_olerepr_"] = olerepr
        self.__dict__["_mapCachedItems_"] = {}
        self.__dict__["_builtMethods_"] = {}
        self.__dict__["_enum_"] = None
        self.__dict__["_unicode_to_string_"] = None
        self.__dict__["_lazydata_"] = lazydata

    def __call__(self, *args):
        "Provide 'default dispatch' COM functionality - allow instance to be called"
        if self._olerepr_.defaultDispatchName:
            invkind, dispid = self._find_dispatch_type_(
                self._olerepr_.defaultDispatchName
            )
        else:
            invkind, dispid = (
                pythoncom.DISPATCH_METHOD | pythoncom.DISPATCH_PROPERTYGET,
                pythoncom.DISPID_VALUE,
            )
        if invkind is not None:
            allArgs = (dispid, LCID, invkind, 1) + args
            return self._get_good_object_(
                self._oleobj_.Invoke(*allArgs), self._olerepr_.defaultDispatchName, None
            )
        raise TypeError("This dispatch object does not define a default method")

    def __bool__(self):
        return True  # ie "if object:" should always be "true" - without this, __len__ is tried.
        # _Possibly_ want to defer to __len__ if available, but Im not sure this is
        # desirable???

    def __repr__(self):
        return "<COMObject %s>" % (self._username_)

    def __str__(self):
        # __str__ is used when the user does "print object", so we gracefully
        # fall back to the __repr__ if the object has no default method.
        try:
            return str(self.__call__())
        except pythoncom.com_error as details:
            if details.hresult not in ERRORS_BAD_CONTEXT:
                raise
            return self.__repr__()

    def __dir__(self):
        lst = list(self.__dict__.keys()) + dir(self.__class__) + self._dir_ole_()
        try:
            lst += [p.Name for p in self.Properties_]
        except AttributeError:
            pass
        return list(set(lst))

    def _dir_ole_(self):
        items_dict = {}
        for iTI in range(0, self._oleobj_.GetTypeInfoCount()):
            typeInfo = self._oleobj_.GetTypeInfo(iTI)
            self._UpdateWithITypeInfo_(items_dict, typeInfo)
        return list(items_dict.keys())

    def _UpdateWithITypeInfo_(self, items_dict, typeInfo):
        typeInfos = [typeInfo]
        # suppress IDispatch and IUnknown methods
        inspectedIIDs = {pythoncom.IID_IDispatch: None}

        while len(typeInfos) > 0:
            typeInfo = typeInfos.pop()
            typeAttr = typeInfo.GetTypeAttr()

            if typeAttr.iid not in inspectedIIDs:
                inspectedIIDs[typeAttr.iid] = None
                for iFun in range(0, typeAttr.cFuncs):
                    funDesc = typeInfo.GetFuncDesc(iFun)
                    funName = typeInfo.GetNames(funDesc.memid)[0]
                    if funName not in items_dict:
                        items_dict[funName] = None

                # Inspect the type info of all implemented types
                # E.g. IShellDispatch5 implements IShellDispatch4 which implements IShellDispatch3 ...
                for iImplType in range(0, typeAttr.cImplTypes):
                    iRefType = typeInfo.GetRefTypeOfImplType(iImplType)
                    refTypeInfo = typeInfo.GetRefTypeInfo(iRefType)
                    typeInfos.append(refTypeInfo)

    # Delegate comparison to the oleobjs, as they know how to do identity.
    def __eq__(self, other):
        other = getattr(other, "_oleobj_", other)
        return self._oleobj_ == other

    def __ne__(self, other):
        other = getattr(other, "_oleobj_", other)
        return self._oleobj_ != other

    def __int__(self):
        return int(self.__call__())

    def __len__(self):
        invkind, dispid = self._find_dispatch_type_("Count")
        if invkind:
            return self._oleobj_.Invoke(dispid, LCID, invkind, 1)
        raise TypeError("This dispatch object does not define a Count method")

    def _NewEnum(self):
        try:
            invkind = pythoncom.DISPATCH_METHOD | pythoncom.DISPATCH_PROPERTYGET
            enum = self._oleobj_.InvokeTypes(
                pythoncom.DISPID_NEWENUM, LCID, invkind, (13, 10), ()
            )
        except pythoncom.com_error:
            return None  # no enumerator for this object.
        from . import util

        return util.WrapEnum(enum, None)

    def __getitem__(self, index):  # syver modified
        # Improved __getitem__ courtesy Syver Enstad
        # Must check _NewEnum before Item, to ensure b/w compat.
        if isinstance(index, int):
            if self.__dict__["_enum_"] is None:
                self.__dict__["_enum_"] = self._NewEnum()
            if self.__dict__["_enum_"] is not None:
                return self._get_good_object_(self._enum_.__getitem__(index))
        # See if we have an "Item" method/property we can use (goes hand in hand with Count() above!)
        invkind, dispid = self._find_dispatch_type_("Item")
        if invkind is not None:
            return self._get_good_object_(
                self._oleobj_.Invoke(dispid, LCID, invkind, 1, index)
            )
        raise TypeError("This object does not support enumeration")

    def __setitem__(self, index, *args):
        # XXX - todo - We should support calling Item() here too!
        # 		print "__setitem__ with", index, args
        if self._olerepr_.defaultDispatchName:
            invkind, dispid = self._find_dispatch_type_(
                self._olerepr_.defaultDispatchName
            )
        else:
            invkind, dispid = (
                pythoncom.DISPATCH_PROPERTYPUT | pythoncom.DISPATCH_PROPERTYPUTREF,
                pythoncom.DISPID_VALUE,
            )
        if invkind is not None:
            allArgs = (dispid, LCID, invkind, 0, index) + args
            return self._get_good_object_(
                self._oleobj_.Invoke(*allArgs), self._olerepr_.defaultDispatchName, None
            )
        raise TypeError("This dispatch object does not define a default method")

    def _find_dispatch_type_(self, methodName):
        if methodName in self._olerepr_.mapFuncs:
            item = self._olerepr_.mapFuncs[methodName]
            return item.desc[4], item.dispid

        if methodName in self._olerepr_.propMapGet:
            item = self._olerepr_.propMapGet[methodName]
            return item.desc[4], item.dispid

        try:
            dispid = self._oleobj_.GetIDsOfNames(0, methodName)
        except:  ### what error?
            return None, None
        return pythoncom.DISPATCH_METHOD | pythoncom.DISPATCH_PROPERTYGET, dispid

    def _ApplyTypes_(self, dispid, wFlags, retType, argTypes, user, resultCLSID, *args):
        result = self._oleobj_.InvokeTypes(
            *(dispid, LCID, wFlags, retType, argTypes) + args
        )
        return self._get_good_object_(result, user, resultCLSID)

    def _wrap_dispatch_(
        self, ob, userName=None, returnCLSID=None, UnicodeToString=None
    ):
        # Given a dispatch object, wrap it in a class
        assert UnicodeToString is None, "this is deprecated and will go away"
        return Dispatch(ob, userName)

    def _get_good_single_object_(self, ob, userName=None, ReturnCLSID=None):
        if isinstance(ob, PyIDispatchType):
            # make a new instance of (probably this) class.
            return self._wrap_dispatch_(ob, userName, ReturnCLSID)
        if isinstance(ob, PyIUnknownType):
            try:
                ob = ob.QueryInterface(pythoncom.IID_IDispatch)
            except pythoncom.com_error:
                # It is an IUnknown, but not an IDispatch, so just let it through.
                return ob
            return self._wrap_dispatch_(ob, userName, ReturnCLSID)
        return ob

    def _get_good_object_(self, ob, userName=None, ReturnCLSID=None):
        """Given an object (usually the retval from a method), make it a good object to return.
        Basically checks if it is a COM object, and wraps it up.
        Also handles the fact that a retval may be a tuple of retvals"""
        if ob is None:  # Quick exit!
            return None
        elif isinstance(ob, tuple):
            return tuple(
                map(
                    lambda o, s=self, oun=userName, rc=ReturnCLSID: s._get_good_single_object_(
                        o, oun, rc
                    ),
                    ob,
                )
            )
        else:
            return self._get_good_single_object_(ob)

    def _make_method_(self, name):
        "Make a method object - Assumes in olerepr funcmap"
        methodName = build.MakePublicAttributeName(name)  # translate keywords etc.
        methodCodeList = self._olerepr_.MakeFuncMethod(
            self._olerepr_.mapFuncs[name], methodName, 0
        )
        methodCode = "\n".join(methodCodeList)
        try:
            # 			print "Method code for %s is:\n" % self._username_, methodCode
            # 			self._print_details_()
            codeObject = compile(methodCode, "<COMObject %s>" % self._username_, "exec")
            # Exec the code object
            tempNameSpace = {}
            # "Dispatch" in the exec'd code is win32com.client.Dispatch, not ours.
            globNameSpace = globals().copy()
            globNameSpace["Dispatch"] = win32com.client.Dispatch
            exec(
                codeObject, globNameSpace, tempNameSpace
            )  # self.__dict__, self.__dict__
            name = methodName
            # Save the function in map.
            fn = self._builtMethods_[name] = tempNameSpace[name]
            newMeth = MakeMethod(fn, self, self.__class__)
            return newMeth
        except:
            debug_print("Error building OLE definition for code ", methodCode)
            traceback.print_exc()
        return None

    def _Release_(self):
        """Cleanup object - like a close - to force cleanup when you dont
        want to rely on Python's reference counting."""
        for childCont in self._mapCachedItems_.values():
            childCont._Release_()
        self._mapCachedItems_ = {}
        if self._oleobj_:
            self._oleobj_.Release()
            self.__dict__["_oleobj_"] = None
        if self._olerepr_:
            self.__dict__["_olerepr_"] = None
        self._enum_ = None

    def _proc_(self, name, *args):
        """Call the named method as a procedure, rather than function.
        Mainly used by Word.Basic, which whinges about such things."""
        try:
            item = self._olerepr_.mapFuncs[name]
            dispId = item.dispid
            return self._get_good_object_(
                self._oleobj_.Invoke(*(dispId, LCID, item.desc[4], 0) + (args))
            )
        except KeyError:
            raise AttributeError(name)

    def _print_details_(self):
        "Debug routine - dumps what it knows about an object."
        print("AxDispatch container", self._username_)
        try:
            print("Methods:")
            for method in self._olerepr_.mapFuncs.keys():
                print("\t", method)
            print("Props:")
            for prop, entry in self._olerepr_.propMap.items():
                print("\t%s = 0x%x - %s" % (prop, entry.dispid, repr(entry)))
            print("Get Props:")
            for prop, entry in self._olerepr_.propMapGet.items():
                print("\t%s = 0x%x - %s" % (prop, entry.dispid, repr(entry)))
            print("Put Props:")
            for prop, entry in self._olerepr_.propMapPut.items():
                print("\t%s = 0x%x - %s" % (prop, entry.dispid, repr(entry)))
        except:
            traceback.print_exc()

    def __LazyMap__(self, attr):
        try:
            if self._LazyAddAttr_(attr):
                debug_attr_print(
                    "%s.__LazyMap__(%s) added something" % (self._username_, attr)
                )
                return 1
        except AttributeError:
            return 0

    # Using the typecomp, lazily create a new attribute definition.
    def _LazyAddAttr_(self, attr):
        if self._lazydata_ is None:
            return 0
        res = 0
        typeinfo, typecomp = self._lazydata_
        olerepr = self._olerepr_
        # We need to explicitly check each invoke type individually - simply
        # specifying '0' will bind to "any member", which may not be the one
        # we are actually after (ie, we may be after prop_get, but returned
        # the info for the prop_put.)
        for i in ALL_INVOKE_TYPES:
            try:
                x, t = typecomp.Bind(attr, i)
                # Support 'Get' and 'Set' properties - see
                # bug 1587023
                if x == 0 and attr[:3] in ("Set", "Get"):
                    x, t = typecomp.Bind(attr[3:], i)
                if x == pythoncom.DESCKIND_FUNCDESC:  # it's a FUNCDESC
                    r = olerepr._AddFunc_(typeinfo, t, 0)
                elif x == pythoncom.DESCKIND_VARDESC:  # it's a VARDESC
                    r = olerepr._AddVar_(typeinfo, t, 0)
                else:  # not found or TYPEDESC/IMPLICITAPP
                    r = None
                if not r is None:
                    key, map = r[0], r[1]
                    item = map[key]
                    if map == olerepr.propMapPut:
                        olerepr._propMapPutCheck_(key, item)
                    elif map == olerepr.propMapGet:
                        olerepr._propMapGetCheck_(key, item)
                    res = 1
            except:
                pass
        return res

    def _FlagAsMethod(self, *methodNames):
        """Flag these attribute names as being methods.
        Some objects do not correctly differentiate methods and
        properties, leading to problems when calling these methods.

        Specifically, trying to say: ob.SomeFunc()
        may yield an exception "None object is not callable"
        In this case, an attempt to fetch the *property* has worked
        and returned None, rather than indicating it is really a method.
        Calling: ob._FlagAsMethod("SomeFunc")
        should then allow this to work.
        """
        for name in methodNames:
            details = build.MapEntry(self.__AttrToID__(name), (name,))
            self._olerepr_.mapFuncs[name] = details

    def __AttrToID__(self, attr):
        debug_attr_print(
            "Calling GetIDsOfNames for property %s in Dispatch container %s"
            % (attr, self._username_)
        )
        return self._oleobj_.GetIDsOfNames(0, attr)

    def __getattr__(self, attr):
        if attr == "__iter__":
            # We can't handle this as a normal method, as if the attribute
            # exists, then it must return an iterable object.
            try:
                invkind = pythoncom.DISPATCH_METHOD | pythoncom.DISPATCH_PROPERTYGET
                enum = self._oleobj_.InvokeTypes(
                    pythoncom.DISPID_NEWENUM, LCID, invkind, (13, 10), ()
                )
            except pythoncom.com_error:
                raise AttributeError("This object can not function as an iterator")

            # We must return a callable object.
            class Factory:
                def __init__(self, ob):
                    self.ob = ob

                def __call__(self):
                    import win32com.client.util

                    return win32com.client.util.Iterator(self.ob)

            return Factory(enum)

        if attr.startswith("_") and attr.endswith("_"):  # Fast-track.
            raise AttributeError(attr)
        # If a known method, create new instance and return.
        try:
            return MakeMethod(self._builtMethods_[attr], self, self.__class__)
        except KeyError:
            pass
        # XXX - Note that we current are case sensitive in the method.
        # debug_attr_print("GetAttr called for %s on DispatchContainer %s" % (attr,self._username_))
        # First check if it is in the method map.  Note that an actual method
        # must not yet exist, (otherwise we would not be here).  This
        # means we create the actual method object - which also means
        # this code will never be asked for that method name again.
        if attr in self._olerepr_.mapFuncs:
            return self._make_method_(attr)

        # Delegate to property maps/cached items
        retEntry = None
        if self._olerepr_ and self._oleobj_:
            # first check general property map, then specific "put" map.
            retEntry = self._olerepr_.propMap.get(attr)
            if retEntry is None:
                retEntry = self._olerepr_.propMapGet.get(attr)
            # Not found so far - See what COM says.
            if retEntry is None:
                try:
                    if self.__LazyMap__(attr):
                        if attr in self._olerepr_.mapFuncs:
                            return self._make_method_(attr)
                        retEntry = self._olerepr_.propMap.get(attr)
                        if retEntry is None:
                            retEntry = self._olerepr_.propMapGet.get(attr)
                    if retEntry is None:
                        retEntry = build.MapEntry(self.__AttrToID__(attr), (attr,))
                except pythoncom.ole_error:
                    pass  # No prop by that name - retEntry remains None.

        if retEntry is not None:  # see if in my cache
            try:
                ret = self._mapCachedItems_[retEntry.dispid]
                debug_attr_print("Cached items has attribute!", ret)
                return ret
            except (KeyError, AttributeError):
                debug_attr_print("Attribute %s not in cache" % attr)

        # If we are still here, and have a retEntry, get the OLE item
        if retEntry is not None:
            invoke_type = _GetDescInvokeType(retEntry, pythoncom.INVOKE_PROPERTYGET)
            debug_attr_print(
                "Getting property Id 0x%x from OLE object" % retEntry.dispid
            )
            try:
                ret = self._oleobj_.Invoke(retEntry.dispid, 0, invoke_type, 1)
            except pythoncom.com_error as details:
                if details.hresult in ERRORS_BAD_CONTEXT:
                    # May be a method.
                    self._olerepr_.mapFuncs[attr] = retEntry
                    return self._make_method_(attr)
                raise
            debug_attr_print("OLE returned ", ret)
            return self._get_good_object_(ret)

        # no where else to look.
        raise AttributeError("%s.%s" % (self._username_, attr))

    def __setattr__(self, attr, value):
        if (
            attr in self.__dict__
        ):  # Fast-track - if already in our dict, just make the assignment.
            # XXX - should maybe check method map - if someone assigns to a method,
            # it could mean something special (not sure what, tho!)
            self.__dict__[attr] = value
            return
        # Allow property assignment.
        debug_attr_print(
            "SetAttr called for %s.%s=%s on DispatchContainer"
            % (self._username_, attr, repr(value))
        )

        if self._olerepr_:
            # Check the "general" property map.
            if attr in self._olerepr_.propMap:
                entry = self._olerepr_.propMap[attr]
                invoke_type = _GetDescInvokeType(entry, pythoncom.INVOKE_PROPERTYPUT)
                self._oleobj_.Invoke(entry.dispid, 0, invoke_type, 0, value)
                return
            # Check the specific "put" map.
            if attr in self._olerepr_.propMapPut:
                entry = self._olerepr_.propMapPut[attr]
                invoke_type = _GetDescInvokeType(entry, pythoncom.INVOKE_PROPERTYPUT)
                self._oleobj_.Invoke(entry.dispid, 0, invoke_type, 0, value)
                return

        # Try the OLE Object
        if self._oleobj_:
            if self.__LazyMap__(attr):
                # Check the "general" property map.
                if attr in self._olerepr_.propMap:
                    entry = self._olerepr_.propMap[attr]
                    invoke_type = _GetDescInvokeType(
                        entry, pythoncom.INVOKE_PROPERTYPUT
                    )
                    self._oleobj_.Invoke(entry.dispid, 0, invoke_type, 0, value)
                    return
                # Check the specific "put" map.
                if attr in self._olerepr_.propMapPut:
                    entry = self._olerepr_.propMapPut[attr]
                    invoke_type = _GetDescInvokeType(
                        entry, pythoncom.INVOKE_PROPERTYPUT
                    )
                    self._oleobj_.Invoke(entry.dispid, 0, invoke_type, 0, value)
                    return
            try:
                entry = build.MapEntry(self.__AttrToID__(attr), (attr,))
            except pythoncom.com_error:
                # No attribute of that name
                entry = None
            if entry is not None:
                try:
                    invoke_type = _GetDescInvokeType(
                        entry, pythoncom.INVOKE_PROPERTYPUT
                    )
                    self._oleobj_.Invoke(entry.dispid, 0, invoke_type, 0, value)
                    self._olerepr_.propMap[attr] = entry
                    debug_attr_print(
                        "__setattr__ property %s (id=0x%x) in Dispatch container %s"
                        % (attr, entry.dispid, self._username_)
                    )
                    return
                except pythoncom.com_error:
                    pass
        raise AttributeError(
            "Property '%s.%s' can not be set." % (self._username_, attr)
        )
