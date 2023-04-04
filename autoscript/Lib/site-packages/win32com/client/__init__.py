# This module exists to create the "best" dispatch object for a given
# object.  If "makepy" support for a given object is detected, it is
# used, otherwise a dynamic dispatch object.

# Note that if the unknown dispatch object then returns a known
# dispatch object, the known class will be used.  This contrasts
# with dynamic.Dispatch behaviour, where dynamic objects are always used.

import sys

import pythoncom
import pywintypes

from . import dynamic, gencache

_PyIDispatchType = pythoncom.TypeIIDs[pythoncom.IID_IDispatch]


def __WrapDispatch(
    dispatch,
    userName=None,
    resultCLSID=None,
    typeinfo=None,
    UnicodeToString=None,
    clsctx=pythoncom.CLSCTX_SERVER,
    WrapperClass=None,
):
    """
    Helper function to return a makepy generated class for a CLSID if it exists,
    otherwise cope by using CDispatch.
    """
    assert UnicodeToString is None, "this is deprecated and will go away"
    if resultCLSID is None:
        try:
            typeinfo = dispatch.GetTypeInfo()
            if (
                typeinfo is not None
            ):  # Some objects return NULL, some raise exceptions...
                resultCLSID = str(typeinfo.GetTypeAttr()[0])
        except (pythoncom.com_error, AttributeError):
            pass
    if resultCLSID is not None:
        from . import gencache

        # Attempt to load generated module support
        # This may load the module, and make it available
        klass = gencache.GetClassForCLSID(resultCLSID)
        if klass is not None:
            return klass(dispatch)

    # Return a "dynamic" object - best we can do!
    if WrapperClass is None:
        WrapperClass = CDispatch
    return dynamic.Dispatch(dispatch, userName, WrapperClass, typeinfo, clsctx=clsctx)


def GetObject(Pathname=None, Class=None, clsctx=None):
    """
    Mimic VB's GetObject() function.

    ob = GetObject(Class = "ProgID") or GetObject(Class = clsid) will
    connect to an already running instance of the COM object.

    ob = GetObject(r"c:\blah\blah\foo.xls") (aka the COM moniker syntax)
    will return a ready to use Python wrapping of the required COM object.

    Note: You must specifiy one or the other of these arguments. I know
    this isn't pretty, but it is what VB does. Blech. If you don't
    I'll throw ValueError at you. :)

    This will most likely throw pythoncom.com_error if anything fails.
    """
    if clsctx is None:
        clsctx = pythoncom.CLSCTX_ALL

    if (Pathname is None and Class is None) or (
        Pathname is not None and Class is not None
    ):
        raise ValueError(
            "You must specify a value for Pathname or Class, but not both."
        )

    if Class is not None:
        return GetActiveObject(Class, clsctx)
    else:
        return Moniker(Pathname, clsctx)


def GetActiveObject(Class, clsctx=pythoncom.CLSCTX_ALL):
    """
    Python friendly version of GetObject's ProgID/CLSID functionality.
    """
    resultCLSID = pywintypes.IID(Class)
    dispatch = pythoncom.GetActiveObject(resultCLSID)
    dispatch = dispatch.QueryInterface(pythoncom.IID_IDispatch)
    return __WrapDispatch(dispatch, Class, resultCLSID=resultCLSID, clsctx=clsctx)


def Moniker(Pathname, clsctx=pythoncom.CLSCTX_ALL):
    """
    Python friendly version of GetObject's moniker functionality.
    """
    moniker, i, bindCtx = pythoncom.MkParseDisplayName(Pathname)
    dispatch = moniker.BindToObject(bindCtx, None, pythoncom.IID_IDispatch)
    return __WrapDispatch(dispatch, Pathname, clsctx=clsctx)


def Dispatch(
    dispatch,
    userName=None,
    resultCLSID=None,
    typeinfo=None,
    UnicodeToString=None,
    clsctx=pythoncom.CLSCTX_SERVER,
):
    """Creates a Dispatch based COM object."""
    assert UnicodeToString is None, "this is deprecated and will go away"
    dispatch, userName = dynamic._GetGoodDispatchAndUserName(dispatch, userName, clsctx)
    return __WrapDispatch(dispatch, userName, resultCLSID, typeinfo, clsctx=clsctx)


def DispatchEx(
    clsid,
    machine=None,
    userName=None,
    resultCLSID=None,
    typeinfo=None,
    UnicodeToString=None,
    clsctx=None,
):
    """Creates a Dispatch based COM object on a specific machine."""
    assert UnicodeToString is None, "this is deprecated and will go away"
    # If InProc is registered, DCOM will use it regardless of the machine name
    # (and regardless of the DCOM config for the object.)  So unless the user
    # specifies otherwise, we exclude inproc apps when a remote machine is used.
    if clsctx is None:
        clsctx = pythoncom.CLSCTX_SERVER
        if machine is not None:
            clsctx = clsctx & ~pythoncom.CLSCTX_INPROC
    if machine is None:
        serverInfo = None
    else:
        serverInfo = (machine,)
    if userName is None:
        userName = clsid
    dispatch = pythoncom.CoCreateInstanceEx(
        clsid, None, clsctx, serverInfo, (pythoncom.IID_IDispatch,)
    )[0]
    return Dispatch(dispatch, userName, resultCLSID, typeinfo, clsctx=clsctx)


class CDispatch(dynamic.CDispatch):
    """
    The dynamic class used as a last resort.
    The purpose of this overriding of dynamic.CDispatch is to perpetuate the policy
    of using the makepy generated wrapper Python class instead of dynamic.CDispatch
    if/when possible.
    """

    def _wrap_dispatch_(
        self, ob, userName=None, returnCLSID=None, UnicodeToString=None
    ):
        assert UnicodeToString is None, "this is deprecated and will go away"
        return Dispatch(ob, userName, returnCLSID, None)

    def __dir__(self):
        return dynamic.CDispatch.__dir__(self)


def CastTo(ob, target, typelib=None):
    """'Cast' a COM object to another interface"""
    # todo - should support target being an IID
    mod = None
    if (
        typelib is not None
    ):  # caller specified target typelib (TypelibSpec). See e.g. selecttlb.EnumTlbs().
        mod = gencache.MakeModuleForTypelib(
            typelib.clsid, typelib.lcid, int(typelib.major, 16), int(typelib.minor, 16)
        )
        if not hasattr(mod, target):
            raise ValueError(
                "The interface name '%s' does not appear in the "
                "specified library %r" % (target, typelib.ver_desc)
            )

    elif hasattr(target, "index"):  # string like
        # for now, we assume makepy for this to work.
        if "CLSID" not in ob.__class__.__dict__:
            # Eeek - no makepy support - try and build it.
            ob = gencache.EnsureDispatch(ob)
        if "CLSID" not in ob.__class__.__dict__:
            raise ValueError("Must be a makepy-able object for this to work")
        clsid = ob.CLSID
        # Lots of hoops to support "demand-build" - ie, generating
        # code for an interface first time it is used.  We assume the
        # interface name exists in the same library as the object.
        # This is generally the case - only referenced typelibs may be
        # a problem, and we can handle that later.  Maybe <wink>
        # So get the generated module for the library itself, then
        # find the interface CLSID there.
        mod = gencache.GetModuleForCLSID(clsid)
        # Get the 'root' module.
        mod = gencache.GetModuleForTypelib(
            mod.CLSID, mod.LCID, mod.MajorVersion, mod.MinorVersion
        )
        # Find the CLSID of the target
        target_clsid = mod.NamesToIIDMap.get(target)
        if target_clsid is None:
            raise ValueError(
                "The interface name '%s' does not appear in the "
                "same library as object '%r'" % (target, ob)
            )
        mod = gencache.GetModuleForCLSID(target_clsid)
    if mod is not None:
        target_class = getattr(mod, target)
        # resolve coclass to interface
        target_class = getattr(target_class, "default_interface", target_class)
        return target_class(ob)  # auto QI magic happens
    raise ValueError


class Constants:
    """A container for generated COM constants."""

    def __init__(self):
        self.__dicts__ = []  # A list of dictionaries

    def __getattr__(self, a):
        for d in self.__dicts__:
            if a in d:
                return d[a]
        raise AttributeError(a)


# And create an instance.
constants = Constants()


# A helpers for DispatchWithEvents - this becomes __setattr__ for the
# temporary class.
def _event_setattr_(self, attr, val):
    try:
        # Does the COM object have an attribute of this name?
        self.__class__.__bases__[0].__setattr__(self, attr, val)
    except AttributeError:
        # Otherwise just stash it away in the instance.
        self.__dict__[attr] = val


# An instance of this "proxy" is created to break the COM circular references
# that exist (ie, when we connect to the COM events, COM keeps a reference
# to the object.  Thus, the Event connection must be manually broken before
# our object can die.  This solves the problem by manually breaking the connection
# to the real object as the proxy dies.
class EventsProxy:
    def __init__(self, ob):
        self.__dict__["_obj_"] = ob

    def __del__(self):
        try:
            # If there is a COM error on disconnection we should
            # just ignore it - object probably already shut down...
            self._obj_.close()
        except pythoncom.com_error:
            pass

    def __getattr__(self, attr):
        return getattr(self._obj_, attr)

    def __setattr__(self, attr, val):
        setattr(self._obj_, attr, val)


def DispatchWithEvents(clsid, user_event_class):
    """Create a COM object that can fire events to a user defined class.
    clsid -- The ProgID or CLSID of the object to create.
    user_event_class -- A Python class object that responds to the events.

    This requires makepy support for the COM object being created.  If
    this support does not exist it will be automatically generated by
    this function.  If the object does not support makepy, a TypeError
    exception will be raised.

    The result is a class instance that both represents the COM object
    and handles events from the COM object.

    It is important to note that the returned instance is not a direct
    instance of the user_event_class, but an instance of a temporary
    class object that derives from three classes:
    * The makepy generated class for the COM object
    * The makepy generated class for the COM events
    * The user_event_class as passed to this function.

    If this is not suitable, see the getevents function for an alternative
    technique of handling events.

    Object Lifetimes:  Whenever the object returned from this function is
    cleaned-up by Python, the events will be disconnected from
    the COM object.  This is almost always what should happen,
    but see the documentation for getevents() for more details.

    Example:

    >>> class IEEvents:
    ...    def OnVisible(self, visible):
    ...       print "Visible changed:", visible
    ...
    >>> ie = DispatchWithEvents("InternetExplorer.Application", IEEvents)
    >>> ie.Visible = 1
    Visible changed: 1
    >>>
    """
    # Create/Get the object.
    disp = Dispatch(clsid)
    if not disp.__class__.__dict__.get(
        "CLSID"
    ):  # Eeek - no makepy support - try and build it.
        try:
            ti = disp._oleobj_.GetTypeInfo()
            disp_clsid = ti.GetTypeAttr()[0]
            tlb, index = ti.GetContainingTypeLib()
            tla = tlb.GetLibAttr()
            gencache.EnsureModule(tla[0], tla[1], tla[3], tla[4], bValidateFile=0)
            # Get the class from the module.
            disp_class = gencache.GetClassForProgID(str(disp_clsid))
        except pythoncom.com_error:
            raise TypeError(
                "This COM object can not automate the makepy process - please run makepy manually for this object"
            )
    else:
        disp_class = disp.__class__
    # If the clsid was an object, get the clsid
    clsid = disp_class.CLSID
    # Create a new class that derives from 3 classes - the dispatch class, the event sink class and the user class.
    # XXX - we are still "classic style" classes in py2x, so we need can't yet
    # use 'type()' everywhere - revisit soon, as py2x will move to new-style too...
    try:
        from types import ClassType as new_type
    except ImportError:
        new_type = type  # py3k
    events_class = getevents(clsid)
    if events_class is None:
        raise ValueError("This COM object does not support events.")
    result_class = new_type(
        "COMEventClass",
        (disp_class, events_class, user_event_class),
        {"__setattr__": _event_setattr_},
    )
    instance = result_class(
        disp._oleobj_
    )  # This only calls the first base class __init__.
    events_class.__init__(instance, instance)
    if hasattr(user_event_class, "__init__"):
        user_event_class.__init__(instance)
    return EventsProxy(instance)


def WithEvents(disp, user_event_class):
    """Similar to DispatchWithEvents - except that the returned
    object is *not* also usable as the original Dispatch object - that is
    the returned object is not dispatchable.

    The difference is best summarised by example.

    >>> class IEEvents:
    ...    def OnVisible(self, visible):
    ...       print "Visible changed:", visible
    ...
    >>> ie = Dispatch("InternetExplorer.Application")
    >>> ie_events = WithEvents(ie, IEEvents)
    >>> ie.Visible = 1
    Visible changed: 1

    Compare with the code sample for DispatchWithEvents, where you get a
    single object that is both the interface and the event handler.  Note that
    the event handler instance will *not* be able to use 'self.' to refer to
    IE's methods and properties.

    This is mainly useful where using DispatchWithEvents causes
    circular reference problems that the simple proxy doesn't deal with
    """
    disp = Dispatch(disp)
    if not disp.__class__.__dict__.get(
        "CLSID"
    ):  # Eeek - no makepy support - try and build it.
        try:
            ti = disp._oleobj_.GetTypeInfo()
            disp_clsid = ti.GetTypeAttr()[0]
            tlb, index = ti.GetContainingTypeLib()
            tla = tlb.GetLibAttr()
            gencache.EnsureModule(tla[0], tla[1], tla[3], tla[4], bValidateFile=0)
            # Get the class from the module.
            disp_class = gencache.GetClassForProgID(str(disp_clsid))
        except pythoncom.com_error:
            raise TypeError(
                "This COM object can not automate the makepy process - please run makepy manually for this object"
            )
    else:
        disp_class = disp.__class__
    # Get the clsid
    clsid = disp_class.CLSID
    # Create a new class that derives from 2 classes - the event sink
    # class and the user class.
    try:
        from types import ClassType as new_type
    except ImportError:
        new_type = type  # py3k
    events_class = getevents(clsid)
    if events_class is None:
        raise ValueError("This COM object does not support events.")
    result_class = new_type("COMEventClass", (events_class, user_event_class), {})
    instance = result_class(disp)  # This only calls the first base class __init__.
    if hasattr(user_event_class, "__init__"):
        user_event_class.__init__(instance)
    return instance


def getevents(clsid):
    """Determine the default outgoing interface for a class, given
    either a clsid or progid. It returns a class - you can
    conveniently derive your own handler from this class and implement
    the appropriate methods.

    This method relies on the classes produced by makepy. You must use
    either makepy or the gencache module to ensure that the
    appropriate support classes have been generated for the com server
    that you will be handling events from.

    Beware of COM circular references.  When the Events class is connected
    to the COM object, the COM object itself keeps a reference to the Python
    events class.  Thus, neither the Events instance or the COM object will
    ever die by themselves.  The 'close' method on the events instance
    must be called to break this chain and allow standard Python collection
    rules to manage object lifetimes.  Note that DispatchWithEvents() does
    work around this problem by the use of a proxy object, but if you use
    the getevents() function yourself, you must make your own arrangements
    to manage this circular reference issue.

    Beware of creating Python circular references: this will happen if your
    handler has a reference to an object that has a reference back to
    the event source. Call the 'close' method to break the chain.

    Example:

    >>>win32com.client.gencache.EnsureModule('{EAB22AC0-30C1-11CF-A7EB-0000C05BAE0B}',0,1,1)
    <module 'win32com.gen_py.....
    >>>
    >>> class InternetExplorerEvents(win32com.client.getevents("InternetExplorer.Application.1")):
    ...    def OnVisible(self, Visible):
    ...        print "Visibility changed: ", Visible
    ...
    >>>
    >>> ie=win32com.client.Dispatch("InternetExplorer.Application.1")
    >>> events=InternetExplorerEvents(ie)
    >>> ie.Visible=1
    Visibility changed:  1
    >>>
    """

    # find clsid given progid or clsid
    clsid = str(pywintypes.IID(clsid))
    # return default outgoing interface for that class
    klass = gencache.GetClassForCLSID(clsid)
    try:
        return klass.default_source
    except AttributeError:
        # See if we have a coclass for the interfaces.
        try:
            return gencache.GetClassForCLSID(klass.coclass_clsid).default_source
        except AttributeError:
            return None


# A Record object, as used by the COM struct support
def Record(name, object):
    """Creates a new record object, given the name of the record,
    and an object from the same type library.

    Example usage would be:
      app = win32com.client.Dispatch("Some.Application")
      point = win32com.client.Record("SomeAppPoint", app)
      point.x = 0
      point.y = 0
      app.MoveTo(point)
    """
    # XXX - to do - probably should allow "object" to already be a module object.
    from . import gencache

    object = gencache.EnsureDispatch(object)
    module = sys.modules[object.__class__.__module__]
    # to allow us to work correctly with "demand generated" code,
    # we must use the typelib CLSID to obtain the module
    # (otherwise we get the sub-module for the object, which
    # does not hold the records)
    # thus, package may be module, or may be module's parent if demand generated.
    package = gencache.GetModuleForTypelib(
        module.CLSID, module.LCID, module.MajorVersion, module.MinorVersion
    )
    try:
        struct_guid = package.RecordMap[name]
    except KeyError:
        raise ValueError(
            "The structure '%s' is not defined in module '%s'" % (name, package)
        )
    return pythoncom.GetRecordFromGuids(
        module.CLSID, module.MajorVersion, module.MinorVersion, module.LCID, struct_guid
    )


############################################
# The base of all makepy generated classes
############################################
class DispatchBaseClass:
    def __init__(self, oobj=None):
        if oobj is None:
            oobj = pythoncom.new(self.CLSID)
        elif isinstance(oobj, DispatchBaseClass):
            try:
                oobj = oobj._oleobj_.QueryInterface(
                    self.CLSID, pythoncom.IID_IDispatch
                )  # Must be a valid COM instance
            except pythoncom.com_error as details:
                import winerror

                # Some stupid objects fail here, even tho it is _already_ IDispatch!!??
                # Eg, Lotus notes.
                # So just let it use the existing object if E_NOINTERFACE
                if details.hresult != winerror.E_NOINTERFACE:
                    raise
                oobj = oobj._oleobj_
        self.__dict__["_oleobj_"] = oobj  # so we dont call __setattr__

    def __dir__(self):
        lst = (
            list(self.__dict__.keys())
            + dir(self.__class__)
            + list(self._prop_map_get_.keys())
            + list(self._prop_map_put_.keys())
        )
        try:
            lst += [p.Name for p in self.Properties_]
        except AttributeError:
            pass
        return list(set(lst))

    # Provide a prettier name than the CLSID
    def __repr__(self):
        # Need to get the docstring for the module for this class.
        try:
            mod_doc = sys.modules[self.__class__.__module__].__doc__
            if mod_doc:
                mod_name = "win32com.gen_py." + mod_doc
            else:
                mod_name = sys.modules[self.__class__.__module__].__name__
        except KeyError:
            mod_name = "win32com.gen_py.unknown"
        return "<%s.%s instance at 0x%s>" % (
            mod_name,
            self.__class__.__name__,
            id(self),
        )

    # Delegate comparison to the oleobjs, as they know how to do identity.
    def __eq__(self, other):
        other = getattr(other, "_oleobj_", other)
        return self._oleobj_ == other

    def __ne__(self, other):
        other = getattr(other, "_oleobj_", other)
        return self._oleobj_ != other

    def _ApplyTypes_(self, dispid, wFlags, retType, argTypes, user, resultCLSID, *args):
        return self._get_good_object_(
            self._oleobj_.InvokeTypes(dispid, 0, wFlags, retType, argTypes, *args),
            user,
            resultCLSID,
        )

    def __getattr__(self, attr):
        args = self._prop_map_get_.get(attr)
        if args is None:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (repr(self), attr)
            )
        return self._ApplyTypes_(*args)

    def __setattr__(self, attr, value):
        if attr in self.__dict__:
            self.__dict__[attr] = value
            return
        try:
            args, defArgs = self._prop_map_put_[attr]
        except KeyError:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (repr(self), attr)
            )
        self._oleobj_.Invoke(*(args + (value,) + defArgs))

    def _get_good_single_object_(self, obj, obUserName=None, resultCLSID=None):
        return _get_good_single_object_(obj, obUserName, resultCLSID)

    def _get_good_object_(self, obj, obUserName=None, resultCLSID=None):
        return _get_good_object_(obj, obUserName, resultCLSID)


# XXX - These should be consolidated with dynamic.py versions.
def _get_good_single_object_(obj, obUserName=None, resultCLSID=None):
    if _PyIDispatchType == type(obj):
        return Dispatch(obj, obUserName, resultCLSID)
    return obj


def _get_good_object_(obj, obUserName=None, resultCLSID=None):
    if obj is None:
        return None
    elif isinstance(obj, tuple):
        obUserNameTuple = (obUserName,) * len(obj)
        resultCLSIDTuple = (resultCLSID,) * len(obj)
        return tuple(map(_get_good_object_, obj, obUserNameTuple, resultCLSIDTuple))
    else:
        return _get_good_single_object_(obj, obUserName, resultCLSID)


class CoClassBaseClass:
    def __init__(self, oobj=None):
        if oobj is None:
            oobj = pythoncom.new(self.CLSID)
        dispobj = self.__dict__["_dispobj_"] = self.default_interface(oobj)
        # See comments below re the special methods.
        for maybe in [
            "__call__",
            "__str__",
            "__int__",
            "__iter__",
            "__len__",
            "__nonzero__",
        ]:
            if hasattr(dispobj, maybe):
                setattr(self, maybe, getattr(self, "__maybe" + maybe))

    def __repr__(self):
        return "<win32com.gen_py.%s.%s>" % (__doc__, self.__class__.__name__)

    def __getattr__(self, attr):
        d = self.__dict__["_dispobj_"]
        if d is not None:
            return getattr(d, attr)
        raise AttributeError(attr)

    def __setattr__(self, attr, value):
        if attr in self.__dict__:
            self.__dict__[attr] = value
            return
        try:
            d = self.__dict__["_dispobj_"]
            if d is not None:
                d.__setattr__(attr, value)
                return
        except AttributeError:
            pass
        self.__dict__[attr] = value

        # Special methods don't use __getattr__ etc, so explicitly delegate here.
        # Note however, that not all are safe to let bubble up - things like
        # `bool(ob)` will break if the object defines __int__ but then raises an
        # attribute error - eg, see #1753.
        # It depends on what the wrapped COM object actually defines whether these
        # will exist on the underlying object, so __init__ explicitly checks if they
        # do and if so, wires them up.

    def __maybe__call__(self, *args, **kwargs):
        return self.__dict__["_dispobj_"].__call__(*args, **kwargs)

    def __maybe__str__(self, *args):
        return self.__dict__["_dispobj_"].__str__(*args)

    def __maybe__int__(self, *args):
        return self.__dict__["_dispobj_"].__int__(*args)

    def __maybe__iter__(self):
        return self.__dict__["_dispobj_"].__iter__()

    def __maybe__len__(self):
        return self.__dict__["_dispobj_"].__len__()

    def __maybe__nonzero__(self):
        return self.__dict__["_dispobj_"].__nonzero__()


# A very simple VARIANT class.  Only to be used with poorly-implemented COM
# objects.  If an object accepts an arg which is a simple "VARIANT", but still
# is very pickly about the actual variant type (eg, isn't happy with a VT_I4,
# which it would get from a Python integer), you can use this to force a
# particular VT.
class VARIANT(object):
    def __init__(self, vt, value):
        self.varianttype = vt
        self._value = value

    # 'value' is a property so when set by pythoncom it gets any magic wrapping
    # which normally happens for result objects
    def _get_value(self):
        return self._value

    def _set_value(self, newval):
        self._value = _get_good_object_(newval)

    def _del_value(self):
        del self._value

    value = property(_get_value, _set_value, _del_value)

    def __repr__(self):
        return "win32com.client.VARIANT(%r, %r)" % (self.varianttype, self._value)
