"""Policies 

Note that Dispatchers are now implemented in "dispatcher.py", but
are still documented here.

Policies

 A policy is an object which manages the interaction between a public 
 Python object, and COM .  In simple terms, the policy object is the 
 object which is actually called by COM, and it invokes the requested 
 method, fetches/sets the requested property, etc.  See the 
 @win32com.server.policy.CreateInstance@ method for a description of
 how a policy is specified or created.

 Exactly how a policy determines which underlying object method/property 
 is obtained is up to the policy.  A few policies are provided, but you 
 can build your own.  See each policy class for a description of how it 
 implements its policy.

 There is a policy that allows the object to specify exactly which 
 methods and properties will be exposed.  There is also a policy that 
 will dynamically expose all Python methods and properties - even those 
 added after the object has been instantiated.

Dispatchers

 A Dispatcher is a level in front of a Policy.  A dispatcher is the 
 thing which actually receives the COM calls, and passes them to the 
 policy object (which in turn somehow does something with the wrapped 
 object).

 It is important to note that a policy does not need to have a dispatcher.
 A dispatcher has the same interface as a policy, and simply steps in its 
 place, delegating to the real policy.  The primary use for a Dispatcher 
 is to support debugging when necessary, but without imposing overheads 
 when not (ie, by not using a dispatcher at all).

 There are a few dispatchers provided - "tracing" dispatchers which simply 
 prints calls and args (including a variation which uses 
 win32api.OutputDebugString), and a "debugger" dispatcher, which can 
 invoke the debugger when necessary.

Error Handling

 It is important to realise that the caller of these interfaces may
 not be Python.  Therefore, general Python exceptions and tracebacks aren't 
 much use.

 In general, there is an Exception class that should be raised, to allow 
 the framework to extract rich COM type error information.

 The general rule is that the **only** exception returned from Python COM 
 Server code should be an Exception instance.  Any other Python exception 
 should be considered an implementation bug in the server (if not, it 
 should be handled, and an appropriate Exception instance raised).  Any 
 other exception is considered "unexpected", and a dispatcher may take 
 special action (see Dispatchers above)

 Occasionally, the implementation will raise the policy.error error.  
 This usually means there is a problem in the implementation that the 
 Python programmer should fix.

 For example, if policy is asked to wrap an object which it can not 
 support (because, eg, it does not provide _public_methods_ or _dynamic_) 
 then policy.error will be raised, indicating it is a Python programmers 
 problem, rather than a COM error.
 
"""
__author__ = "Greg Stein and Mark Hammond"

import sys
import types

import pythoncom
import pywintypes
import win32api
import win32con
import winerror

# Import a few important constants to speed lookups.
from pythoncom import (
    DISPATCH_METHOD,
    DISPATCH_PROPERTYGET,
    DISPATCH_PROPERTYPUT,
    DISPATCH_PROPERTYPUTREF,
    DISPID_COLLECT,
    DISPID_CONSTRUCTOR,
    DISPID_DESTRUCTOR,
    DISPID_EVALUATE,
    DISPID_NEWENUM,
    DISPID_PROPERTYPUT,
    DISPID_STARTENUM,
    DISPID_UNKNOWN,
    DISPID_VALUE,
)

S_OK = 0

# Few more globals to speed things.
IDispatchType = pythoncom.TypeIIDs[pythoncom.IID_IDispatch]
IUnknownType = pythoncom.TypeIIDs[pythoncom.IID_IUnknown]

from .exception import COMException

error = __name__ + " error"

regSpec = "CLSID\\%s\\PythonCOM"
regPolicy = "CLSID\\%s\\PythonCOMPolicy"
regDispatcher = "CLSID\\%s\\PythonCOMDispatcher"
regAddnPath = "CLSID\\%s\\PythonCOMPath"


def CreateInstance(clsid, reqIID):
    """Create a new instance of the specified IID

    The COM framework **always** calls this function to create a new
    instance for the specified CLSID.  This function looks up the
    registry for the name of a policy, creates the policy, and asks the
    policy to create the specified object by calling the _CreateInstance_ method.

    Exactly how the policy creates the instance is up to the policy.  See the
    specific policy documentation for more details.
    """
    # First see is sys.path should have something on it.
    try:
        addnPaths = win32api.RegQueryValue(
            win32con.HKEY_CLASSES_ROOT, regAddnPath % clsid
        ).split(";")
        for newPath in addnPaths:
            if newPath not in sys.path:
                sys.path.insert(0, newPath)
    except win32api.error:
        pass
    try:
        policy = win32api.RegQueryValue(win32con.HKEY_CLASSES_ROOT, regPolicy % clsid)
        policy = resolve_func(policy)
    except win32api.error:
        policy = DefaultPolicy

    try:
        dispatcher = win32api.RegQueryValue(
            win32con.HKEY_CLASSES_ROOT, regDispatcher % clsid
        )
        if dispatcher:
            dispatcher = resolve_func(dispatcher)
    except win32api.error:
        dispatcher = None

    if dispatcher:
        retObj = dispatcher(policy, None)
    else:
        retObj = policy(None)
    return retObj._CreateInstance_(clsid, reqIID)


class BasicWrapPolicy:
    """The base class of policies.

    Normally not used directly (use a child class, instead)

    This policy assumes we are wrapping another object
    as the COM server.  This supports the delegation of the core COM entry points
    to either the wrapped object, or to a child class.

    This policy supports the following special attributes on the wrapped object

    _query_interface_ -- A handler which can respond to the COM 'QueryInterface' call.
    _com_interfaces_ -- An optional list of IIDs which the interface will assume are
        valid for the object.
    _invoke_ -- A handler which can respond to the COM 'Invoke' call.  If this attribute
        is not provided, then the default policy implementation is used.  If this attribute
        does exist, it is responsible for providing all required functionality - ie, the
        policy _invoke_ method is not invoked at all (and nor are you able to call it!)
    _getidsofnames_ -- A handler which can respond to the COM 'GetIDsOfNames' call.  If this attribute
        is not provided, then the default policy implementation is used.  If this attribute
        does exist, it is responsible for providing all required functionality - ie, the
        policy _getidsofnames_ method is not invoked at all (and nor are you able to call it!)

    IDispatchEx functionality:

    _invokeex_ -- Very similar to _invoke_, except slightly different arguments are used.
        And the result is just the _real_ result (rather than the (hresult, argErr, realResult)
        tuple that _invoke_ uses.
        This is the new, prefered handler (the default _invoke_ handler simply called _invokeex_)
    _getdispid_ -- Very similar to _getidsofnames_, except slightly different arguments are used,
        and only 1 property at a time can be fetched (which is all we support in getidsofnames anyway!)
        This is the new, prefered handler (the default _invoke_ handler simply called _invokeex_)
    _getnextdispid_- uses self._name_to_dispid_ to enumerate the DISPIDs
    """

    def __init__(self, object):
        """Initialise the policy object

        Params:

        object -- The object to wrap.  May be None *iff* @BasicWrapPolicy._CreateInstance_@ will be
        called immediately after this to setup a brand new object
        """
        if object is not None:
            self._wrap_(object)

    def _CreateInstance_(self, clsid, reqIID):
        """Creates a new instance of a **wrapped** object

        This method looks up a "@win32com.server.policy.regSpec@" % clsid entry
        in the registry (using @DefaultPolicy@)
        """
        try:
            classSpec = win32api.RegQueryValue(
                win32con.HKEY_CLASSES_ROOT, regSpec % clsid
            )
        except win32api.error:
            raise error(
                "The object is not correctly registered - %s key can not be read"
                % (regSpec % clsid)
            )
        myob = call_func(classSpec)
        self._wrap_(myob)
        try:
            return pythoncom.WrapObject(self, reqIID)
        except pythoncom.com_error as xxx_todo_changeme:
            (hr, desc, exc, arg) = xxx_todo_changeme.args
            from win32com.util import IIDToInterfaceName

            desc = (
                "The object '%r' was created, but does not support the "
                "interface '%s'(%s): %s"
                % (myob, IIDToInterfaceName(reqIID), reqIID, desc)
            )
            raise pythoncom.com_error(hr, desc, exc, arg)

    def _wrap_(self, object):
        """Wraps up the specified object.

        This function keeps a reference to the passed
        object, and may interogate it to determine how to respond to COM requests, etc.
        """
        # We "clobber" certain of our own methods with ones
        # provided by the wrapped object, iff they exist.
        self._name_to_dispid_ = {}
        ob = self._obj_ = object
        if hasattr(ob, "_query_interface_"):
            self._query_interface_ = ob._query_interface_

        if hasattr(ob, "_invoke_"):
            self._invoke_ = ob._invoke_

        if hasattr(ob, "_invokeex_"):
            self._invokeex_ = ob._invokeex_

        if hasattr(ob, "_getidsofnames_"):
            self._getidsofnames_ = ob._getidsofnames_

        if hasattr(ob, "_getdispid_"):
            self._getdispid_ = ob._getdispid_

        # Allow for override of certain special attributes.
        if hasattr(ob, "_com_interfaces_"):
            self._com_interfaces_ = []
            # Allow interfaces to be specified by name.
            for i in ob._com_interfaces_:
                if type(i) != pywintypes.IIDType:
                    # Prolly a string!
                    if i[0] != "{":
                        i = pythoncom.InterfaceNames[i]
                    else:
                        i = pythoncom.MakeIID(i)
                self._com_interfaces_.append(i)
        else:
            self._com_interfaces_ = []

    # "QueryInterface" handling.
    def _QueryInterface_(self, iid):
        """The main COM entry-point for QueryInterface.

        This checks the _com_interfaces_ attribute and if the interface is not specified
        there, it calls the derived helper _query_interface_
        """
        if iid in self._com_interfaces_:
            return 1
        return self._query_interface_(iid)

    def _query_interface_(self, iid):
        """Called if the object does not provide the requested interface in _com_interfaces_,
        and does not provide a _query_interface_ handler.

        Returns a result to the COM framework indicating the interface is not supported.
        """
        return 0

    # "Invoke" handling.
    def _Invoke_(self, dispid, lcid, wFlags, args):
        """The main COM entry-point for Invoke.

        This calls the _invoke_ helper.
        """
        # Translate a possible string dispid to real dispid.
        if type(dispid) == type(""):
            try:
                dispid = self._name_to_dispid_[dispid.lower()]
            except KeyError:
                raise COMException(
                    scode=winerror.DISP_E_MEMBERNOTFOUND, desc="Member not found"
                )
        return self._invoke_(dispid, lcid, wFlags, args)

    def _invoke_(self, dispid, lcid, wFlags, args):
        # Delegates to the _invokeex_ implementation.  This allows
        # a custom policy to define _invokeex_, and automatically get _invoke_ too.
        return S_OK, -1, self._invokeex_(dispid, lcid, wFlags, args, None, None)

    # "GetIDsOfNames" handling.
    def _GetIDsOfNames_(self, names, lcid):
        """The main COM entry-point for GetIDsOfNames.

        This checks the validity of the arguments, and calls the _getidsofnames_ helper.
        """
        if len(names) > 1:
            raise COMException(
                scode=winerror.DISP_E_INVALID,
                desc="Cannot support member argument names",
            )
        return self._getidsofnames_(names, lcid)

    def _getidsofnames_(self, names, lcid):
        ### note: lcid is being ignored...
        return (self._getdispid_(names[0], 0),)

    # IDispatchEx support for policies.  Most of the IDispathEx functionality
    # by default will raise E_NOTIMPL.  Thus it is not necessary for derived
    # policies to explicitely implement all this functionality just to not implement it!

    def _GetDispID_(self, name, fdex):
        return self._getdispid_(name, fdex)

    def _getdispid_(self, name, fdex):
        try:
            ### TODO - look at the fdex flags!!!
            return self._name_to_dispid_[name.lower()]
        except KeyError:
            raise COMException(scode=winerror.DISP_E_UNKNOWNNAME)

    # "InvokeEx" handling.
    def _InvokeEx_(self, dispid, lcid, wFlags, args, kwargs, serviceProvider):
        """The main COM entry-point for InvokeEx.

        This calls the _invokeex_ helper.
        """
        # Translate a possible string dispid to real dispid.
        if type(dispid) == type(""):
            try:
                dispid = self._name_to_dispid_[dispid.lower()]
            except KeyError:
                raise COMException(
                    scode=winerror.DISP_E_MEMBERNOTFOUND, desc="Member not found"
                )
        return self._invokeex_(dispid, lcid, wFlags, args, kwargs, serviceProvider)

    def _invokeex_(self, dispid, lcid, wFlags, args, kwargs, serviceProvider):
        """A stub for _invokeex_ - should never be called.

        Simply raises an exception.
        """
        # Base classes should override this method (and not call the base)
        raise error("This class does not provide _invokeex_ semantics")

    def _DeleteMemberByName_(self, name, fdex):
        return self._deletememberbyname_(name, fdex)

    def _deletememberbyname_(self, name, fdex):
        raise COMException(scode=winerror.E_NOTIMPL)

    def _DeleteMemberByDispID_(self, id):
        return self._deletememberbydispid(id)

    def _deletememberbydispid_(self, id):
        raise COMException(scode=winerror.E_NOTIMPL)

    def _GetMemberProperties_(self, id, fdex):
        return self._getmemberproperties_(id, fdex)

    def _getmemberproperties_(self, id, fdex):
        raise COMException(scode=winerror.E_NOTIMPL)

    def _GetMemberName_(self, dispid):
        return self._getmembername_(dispid)

    def _getmembername_(self, dispid):
        raise COMException(scode=winerror.E_NOTIMPL)

    def _GetNextDispID_(self, fdex, dispid):
        return self._getnextdispid_(fdex, dispid)

    def _getnextdispid_(self, fdex, dispid):
        ids = list(self._name_to_dispid_.values())
        ids.sort()
        if DISPID_STARTENUM in ids:
            ids.remove(DISPID_STARTENUM)
        if dispid == DISPID_STARTENUM:
            return ids[0]
        else:
            try:
                return ids[ids.index(dispid) + 1]
            except ValueError:  # dispid not in list?
                raise COMException(scode=winerror.E_UNEXPECTED)
            except IndexError:  # No more items
                raise COMException(scode=winerror.S_FALSE)

    def _GetNameSpaceParent_(self):
        return self._getnamespaceparent()

    def _getnamespaceparent_(self):
        raise COMException(scode=winerror.E_NOTIMPL)


class MappedWrapPolicy(BasicWrapPolicy):
    """Wraps an object using maps to do its magic

    This policy wraps up a Python object, using a number of maps
    which translate from a Dispatch ID and flags, into an object to call/getattr, etc.

    It is the responsibility of derived classes to determine exactly how the
    maps are filled (ie, the derived classes determine the map filling policy.

    This policy supports the following special attributes on the wrapped object

    _dispid_to_func_/_dispid_to_get_/_dispid_to_put_ -- These are dictionaries
      (keyed by integer dispid, values are string attribute names) which the COM
      implementation uses when it is processing COM requests.  Note that the implementation
      uses this dictionary for its own purposes - not a copy - which means the contents of
      these dictionaries will change as the object is used.

    """

    def _wrap_(self, object):
        BasicWrapPolicy._wrap_(self, object)
        ob = self._obj_
        if hasattr(ob, "_dispid_to_func_"):
            self._dispid_to_func_ = ob._dispid_to_func_
        else:
            self._dispid_to_func_ = {}
        if hasattr(ob, "_dispid_to_get_"):
            self._dispid_to_get_ = ob._dispid_to_get_
        else:
            self._dispid_to_get_ = {}
        if hasattr(ob, "_dispid_to_put_"):
            self._dispid_to_put_ = ob._dispid_to_put_
        else:
            self._dispid_to_put_ = {}

    def _getmembername_(self, dispid):
        if dispid in self._dispid_to_func_:
            return self._dispid_to_func_[dispid]
        elif dispid in self._dispid_to_get_:
            return self._dispid_to_get_[dispid]
        elif dispid in self._dispid_to_put_:
            return self._dispid_to_put_[dispid]
        else:
            raise COMException(scode=winerror.DISP_E_MEMBERNOTFOUND)


class DesignatedWrapPolicy(MappedWrapPolicy):
    """A policy which uses a mapping to link functions and dispid

     A MappedWrappedPolicy which allows the wrapped object to specify, via certain
     special named attributes, exactly which methods and properties are exposed.

     All a wrapped object need do is provide the special attributes, and the policy
     will handle everything else.

     Attributes:

     _public_methods_ -- Required, unless a typelib GUID is given -- A list
                  of strings, which must be the names of methods the object
                  provides.  These methods will be exposed and callable
                  from other COM hosts.
     _public_attrs_ A list of strings, which must be the names of attributes on the object.
                  These attributes will be exposed and readable and possibly writeable from other COM hosts.
     _readonly_attrs_ -- A list of strings, which must also appear in _public_attrs.  These
                  attributes will be readable, but not writable, by other COM hosts.
     _value_ -- A method that will be called if the COM host requests the "default" method
                  (ie, calls Invoke with dispid==DISPID_VALUE)
     _NewEnum -- A method that will be called if the COM host requests an enumerator on the
                  object (ie, calls Invoke with dispid==DISPID_NEWENUM.)
                  It is the responsibility of the method to ensure the returned
                  object conforms to the required Enum interface.

    _typelib_guid_ -- The GUID of the typelibrary with interface definitions we use.
    _typelib_version_ -- A tuple of (major, minor) with a default of 1,1
    _typelib_lcid_ -- The LCID of the typelib, default = LOCALE_USER_DEFAULT

     _Evaluate -- Dunno what this means, except the host has called Invoke with dispid==DISPID_EVALUATE!
                  See the COM documentation for details.
    """

    def _wrap_(self, ob):
        # If we have nominated universal interfaces to support, load them now
        tlb_guid = getattr(ob, "_typelib_guid_", None)
        if tlb_guid is not None:
            tlb_major, tlb_minor = getattr(ob, "_typelib_version_", (1, 0))
            tlb_lcid = getattr(ob, "_typelib_lcid_", 0)
            from win32com import universal

            # XXX - what if the user wants to implement interfaces from multiple
            # typelibs?
            # Filter out all 'normal' IIDs (ie, IID objects and strings starting with {
            interfaces = [
                i
                for i in getattr(ob, "_com_interfaces_", [])
                if type(i) != pywintypes.IIDType and not i.startswith("{")
            ]
            universal_data = universal.RegisterInterfaces(
                tlb_guid, tlb_lcid, tlb_major, tlb_minor, interfaces
            )
        else:
            universal_data = []
        MappedWrapPolicy._wrap_(self, ob)
        if not hasattr(ob, "_public_methods_") and not hasattr(ob, "_typelib_guid_"):
            raise error(
                "Object does not support DesignatedWrapPolicy, as it does not have either _public_methods_ or _typelib_guid_ attributes."
            )

        # Copy existing _dispid_to_func_ entries to _name_to_dispid_
        for dispid, name in self._dispid_to_func_.items():
            self._name_to_dispid_[name.lower()] = dispid
        for dispid, name in self._dispid_to_get_.items():
            self._name_to_dispid_[name.lower()] = dispid
        for dispid, name in self._dispid_to_put_.items():
            self._name_to_dispid_[name.lower()] = dispid

        # Patch up the universal stuff.
        for dispid, invkind, name in universal_data:
            self._name_to_dispid_[name.lower()] = dispid
            if invkind == DISPATCH_METHOD:
                self._dispid_to_func_[dispid] = name
            elif invkind in (DISPATCH_PROPERTYPUT, DISPATCH_PROPERTYPUTREF):
                self._dispid_to_put_[dispid] = name
            elif invkind == DISPATCH_PROPERTYGET:
                self._dispid_to_get_[dispid] = name
            else:
                raise ValueError("unexpected invkind: %d (%s)" % (invkind, name))

        # look for reserved methods
        if hasattr(ob, "_value_"):
            self._dispid_to_get_[DISPID_VALUE] = "_value_"
            self._dispid_to_put_[DISPID_PROPERTYPUT] = "_value_"
        if hasattr(ob, "_NewEnum"):
            self._name_to_dispid_["_newenum"] = DISPID_NEWENUM
            self._dispid_to_func_[DISPID_NEWENUM] = "_NewEnum"
        if hasattr(ob, "_Evaluate"):
            self._name_to_dispid_["_evaluate"] = DISPID_EVALUATE
            self._dispid_to_func_[DISPID_EVALUATE] = "_Evaluate"

        next_dispid = self._allocnextdispid(999)
        # note: funcs have precedence over attrs (install attrs first)
        if hasattr(ob, "_public_attrs_"):
            if hasattr(ob, "_readonly_attrs_"):
                readonly = ob._readonly_attrs_
            else:
                readonly = []
            for name in ob._public_attrs_:
                dispid = self._name_to_dispid_.get(name.lower())
                if dispid is None:
                    dispid = next_dispid
                    self._name_to_dispid_[name.lower()] = dispid
                    next_dispid = self._allocnextdispid(next_dispid)
                self._dispid_to_get_[dispid] = name
                if name not in readonly:
                    self._dispid_to_put_[dispid] = name
        for name in getattr(ob, "_public_methods_", []):
            dispid = self._name_to_dispid_.get(name.lower())
            if dispid is None:
                dispid = next_dispid
                self._name_to_dispid_[name.lower()] = dispid
                next_dispid = self._allocnextdispid(next_dispid)
            self._dispid_to_func_[dispid] = name
        self._typeinfos_ = None  # load these on demand.

    def _build_typeinfos_(self):
        # Can only ever be one for now.
        tlb_guid = getattr(self._obj_, "_typelib_guid_", None)
        if tlb_guid is None:
            return []
        tlb_major, tlb_minor = getattr(self._obj_, "_typelib_version_", (1, 0))
        tlb = pythoncom.LoadRegTypeLib(tlb_guid, tlb_major, tlb_minor)
        typecomp = tlb.GetTypeComp()
        # Not 100% sure what semantics we should use for the default interface.
        # Look for the first name in _com_interfaces_ that exists in the typelib.
        for iname in self._obj_._com_interfaces_:
            try:
                type_info, type_comp = typecomp.BindType(iname)
                if type_info is not None:
                    return [type_info]
            except pythoncom.com_error:
                pass
        return []

    def _GetTypeInfoCount_(self):
        if self._typeinfos_ is None:
            self._typeinfos_ = self._build_typeinfos_()
        return len(self._typeinfos_)

    def _GetTypeInfo_(self, index, lcid):
        if self._typeinfos_ is None:
            self._typeinfos_ = self._build_typeinfos_()
        if index < 0 or index >= len(self._typeinfos_):
            raise COMException(scode=winerror.DISP_E_BADINDEX)
        return 0, self._typeinfos_[index]

    def _allocnextdispid(self, last_dispid):
        while 1:
            last_dispid = last_dispid + 1
            if (
                last_dispid not in self._dispid_to_func_
                and last_dispid not in self._dispid_to_get_
                and last_dispid not in self._dispid_to_put_
            ):
                return last_dispid

    def _invokeex_(self, dispid, lcid, wFlags, args, kwArgs, serviceProvider):
        ### note: lcid is being ignored...

        if wFlags & DISPATCH_METHOD:
            try:
                funcname = self._dispid_to_func_[dispid]
            except KeyError:
                if not wFlags & DISPATCH_PROPERTYGET:
                    raise COMException(
                        scode=winerror.DISP_E_MEMBERNOTFOUND
                    )  # not found
            else:
                try:
                    func = getattr(self._obj_, funcname)
                except AttributeError:
                    # May have a dispid, but that doesnt mean we have the function!
                    raise COMException(scode=winerror.DISP_E_MEMBERNOTFOUND)
                # Should check callable here
                try:
                    return func(*args)
                except TypeError as v:
                    # Particularly nasty is "wrong number of args" type error
                    # This helps you see what 'func' and 'args' actually is
                    if str(v).find("arguments") >= 0:
                        print(
                            "** TypeError %s calling function %r(%r)" % (v, func, args)
                        )
                    raise

        if wFlags & DISPATCH_PROPERTYGET:
            try:
                name = self._dispid_to_get_[dispid]
            except KeyError:
                raise COMException(scode=winerror.DISP_E_MEMBERNOTFOUND)  # not found
            retob = getattr(self._obj_, name)
            if type(retob) == types.MethodType:  # a method as a property - call it.
                retob = retob(*args)
            return retob

        if wFlags & (DISPATCH_PROPERTYPUT | DISPATCH_PROPERTYPUTREF):  ### correct?
            try:
                name = self._dispid_to_put_[dispid]
            except KeyError:
                raise COMException(scode=winerror.DISP_E_MEMBERNOTFOUND)  # read-only
            # If we have a method of that name (ie, a property get function), and
            # we have an equiv. property set function, use that instead.
            if (
                type(getattr(self._obj_, name, None)) == types.MethodType
                and type(getattr(self._obj_, "Set" + name, None)) == types.MethodType
            ):
                fn = getattr(self._obj_, "Set" + name)
                fn(*args)
            else:
                # just set the attribute
                setattr(self._obj_, name, args[0])
            return

        raise COMException(scode=winerror.E_INVALIDARG, desc="invalid wFlags")


class EventHandlerPolicy(DesignatedWrapPolicy):
    """The default policy used by event handlers in the win32com.client package.

    In addition to the base policy, this provides argument conversion semantics for
    params
      * dispatch params are converted to dispatch objects.
      * Unicode objects are converted to strings (1.5.2 and earlier)

    NOTE: Later, we may allow the object to override this process??
    """

    def _transform_args_(self, args, kwArgs, dispid, lcid, wFlags, serviceProvider):
        ret = []
        for arg in args:
            arg_type = type(arg)
            if arg_type == IDispatchType:
                import win32com.client

                arg = win32com.client.Dispatch(arg)
            elif arg_type == IUnknownType:
                try:
                    import win32com.client

                    arg = win32com.client.Dispatch(
                        arg.QueryInterface(pythoncom.IID_IDispatch)
                    )
                except pythoncom.error:
                    pass  # Keep it as IUnknown
            ret.append(arg)
        return tuple(ret), kwArgs

    def _invokeex_(self, dispid, lcid, wFlags, args, kwArgs, serviceProvider):
        # transform the args.
        args, kwArgs = self._transform_args_(
            args, kwArgs, dispid, lcid, wFlags, serviceProvider
        )
        return DesignatedWrapPolicy._invokeex_(
            self, dispid, lcid, wFlags, args, kwArgs, serviceProvider
        )


class DynamicPolicy(BasicWrapPolicy):
    """A policy which dynamically (ie, at run-time) determines public interfaces.

    A dynamic policy is used to dynamically dispatch methods and properties to the
    wrapped object.  The list of objects and properties does not need to be known in
    advance, and methods or properties added to the wrapped object after construction
    are also handled.

    The wrapped object must provide the following attributes:

    _dynamic_ -- A method that will be called whenever an invoke on the object
           is called.  The method is called with the name of the underlying method/property
           (ie, the mapping of dispid to/from name has been resolved.)  This name property
           may also be '_value_' to indicate the default, and '_NewEnum' to indicate a new
           enumerator is requested.

    """

    def _wrap_(self, object):
        BasicWrapPolicy._wrap_(self, object)
        if not hasattr(self._obj_, "_dynamic_"):
            raise error("Object does not support Dynamic COM Policy")
        self._next_dynamic_ = self._min_dynamic_ = 1000
        self._dyn_dispid_to_name_ = {
            DISPID_VALUE: "_value_",
            DISPID_NEWENUM: "_NewEnum",
        }

    def _getdispid_(self, name, fdex):
        # TODO - Look at fdex flags.
        lname = name.lower()
        try:
            return self._name_to_dispid_[lname]
        except KeyError:
            dispid = self._next_dynamic_ = self._next_dynamic_ + 1
            self._name_to_dispid_[lname] = dispid
            self._dyn_dispid_to_name_[dispid] = name  # Keep case in this map...
            return dispid

    def _invoke_(self, dispid, lcid, wFlags, args):
        return S_OK, -1, self._invokeex_(dispid, lcid, wFlags, args, None, None)

    def _invokeex_(self, dispid, lcid, wFlags, args, kwargs, serviceProvider):
        ### note: lcid is being ignored...
        ### note: kwargs is being ignored...
        ### note: serviceProvider is being ignored...
        ### there might be assigned DISPID values to properties, too...
        try:
            name = self._dyn_dispid_to_name_[dispid]
        except KeyError:
            raise COMException(
                scode=winerror.DISP_E_MEMBERNOTFOUND, desc="Member not found"
            )
        return self._obj_._dynamic_(name, lcid, wFlags, args)


DefaultPolicy = DesignatedWrapPolicy


def resolve_func(spec):
    """Resolve a function by name

    Given a function specified by 'module.function', return a callable object
    (ie, the function itself)
    """
    try:
        idx = spec.rindex(".")
        mname = spec[:idx]
        fname = spec[idx + 1 :]
        # Dont attempt to optimize by looking in sys.modules,
        # as another thread may also be performing the import - this
        # way we take advantage of the built-in import lock.
        module = _import_module(mname)
        return getattr(module, fname)
    except ValueError:  # No "." in name - assume in this module
        return globals()[spec]


def call_func(spec, *args):
    """Call a function specified by name.

    Call a function specified by 'module.function' and return the result.
    """

    return resolve_func(spec)(*args)


def _import_module(mname):
    """Import a module just like the 'import' statement.

    Having this function is much nicer for importing arbitrary modules than
    using the 'exec' keyword.  It is more efficient and obvious to the reader.
    """
    __import__(mname)
    # Eeek - result of _import_ is "win32com" - not "win32com.a.b.c"
    # Get the full module from sys.modules
    return sys.modules[mname]


#######
#
# Temporary hacks until all old code moves.
#
# These have been moved to a new source file, but some code may
# still reference them here.  These will end up being removed.
try:
    from .dispatcher import DispatcherTrace, DispatcherWin32trace
except ImportError:  # Quite likely a frozen executable that doesnt need dispatchers
    pass
