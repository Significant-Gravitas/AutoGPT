# Code that packs and unpacks the Univgw structures.

# See if we have a special directory for the binaries (for developers)

import pythoncom
from win32com.client import gencache

com_error = pythoncom.com_error
_univgw = pythoncom._univgw


def RegisterInterfaces(typelibGUID, lcid, major, minor, interface_names=None):
    ret = []  # return a list of (dispid, funcname for our policy's benefit
    # First see if we have makepy support.  If so, we can probably satisfy the request without loading the typelib.
    try:
        mod = gencache.GetModuleForTypelib(typelibGUID, lcid, major, minor)
    except ImportError:
        mod = None
    if mod is None:
        import win32com.client.build

        # Load up the typelib and build (but don't cache) it now
        tlb = pythoncom.LoadRegTypeLib(typelibGUID, major, minor, lcid)
        typecomp_lib = tlb.GetTypeComp()
        if interface_names is None:
            interface_names = []
            for i in range(tlb.GetTypeInfoCount()):
                info = tlb.GetTypeInfo(i)
                doc = tlb.GetDocumentation(i)
                attr = info.GetTypeAttr()
                if attr.typekind == pythoncom.TKIND_INTERFACE or (
                    attr.typekind == pythoncom.TKIND_DISPATCH
                    and attr.wTypeFlags & pythoncom.TYPEFLAG_FDUAL
                ):
                    interface_names.append(doc[0])
        for name in interface_names:
            type_info, type_comp = typecomp_lib.BindType(
                name,
            )
            # Not sure why we don't get an exception here - BindType's C
            # impl looks correct..
            if type_info is None:
                raise ValueError("The interface '%s' can not be located" % (name,))
            # If we got back a Dispatch interface, convert to the real interface.
            attr = type_info.GetTypeAttr()
            if attr.typekind == pythoncom.TKIND_DISPATCH:
                refhtype = type_info.GetRefTypeOfImplType(-1)
                type_info = type_info.GetRefTypeInfo(refhtype)
                attr = type_info.GetTypeAttr()
            item = win32com.client.build.VTableItem(
                type_info, attr, type_info.GetDocumentation(-1)
            )
            _doCreateVTable(
                item.clsid, item.python_name, item.bIsDispatch, item.vtableFuncs
            )
            for info in item.vtableFuncs:
                names, dispid, desc = info
                invkind = desc[4]
                ret.append((dispid, invkind, names[0]))
    else:
        # Cool - can used cached info.
        if not interface_names:
            interface_names = list(mod.VTablesToClassMap.values())
        for name in interface_names:
            try:
                iid = mod.NamesToIIDMap[name]
            except KeyError:
                raise ValueError(
                    "Interface '%s' does not exist in this cached typelib" % (name,)
                )
            #            print "Processing interface", name
            sub_mod = gencache.GetModuleForCLSID(iid)
            is_dispatch = getattr(sub_mod, name + "_vtables_dispatch_", None)
            method_defs = getattr(sub_mod, name + "_vtables_", None)
            if is_dispatch is None or method_defs is None:
                raise ValueError("Interface '%s' is IDispatch only" % (name,))

            # And create the univgw defn
            _doCreateVTable(iid, name, is_dispatch, method_defs)
            for info in method_defs:
                names, dispid, desc = info
                invkind = desc[4]
                ret.append((dispid, invkind, names[0]))
    return ret


def _doCreateVTable(iid, interface_name, is_dispatch, method_defs):
    defn = Definition(iid, is_dispatch, method_defs)
    vtbl = _univgw.CreateVTable(defn, is_dispatch)
    _univgw.RegisterVTable(vtbl, iid, interface_name)


def _CalcTypeSize(typeTuple):
    t = typeTuple[0]
    if t & (pythoncom.VT_BYREF | pythoncom.VT_ARRAY):
        # Its a pointer.
        cb = _univgw.SizeOfVT(pythoncom.VT_PTR)[1]
    elif t == pythoncom.VT_RECORD:
        # Just because a type library uses records doesn't mean the user
        # is trying to.  We need to better place to warn about this, but it
        # isn't here.
        # try:
        #    import warnings
        #    warnings.warn("warning: records are known to not work for vtable interfaces")
        # except ImportError:
        #    print "warning: records are known to not work for vtable interfaces"
        cb = _univgw.SizeOfVT(pythoncom.VT_PTR)[1]
        # cb = typeInfo.GetTypeAttr().cbSizeInstance
    else:
        cb = _univgw.SizeOfVT(t)[1]
    return cb


class Arg:
    def __init__(self, arg_info, name=None):
        self.name = name
        self.vt, self.inOut, self.default, self.clsid = arg_info
        self.size = _CalcTypeSize(arg_info)
        # Offset from the beginning of the arguments of the stack.
        self.offset = 0


class Method:
    def __init__(self, method_info, isEventSink=0):
        all_names, dispid, desc = method_info
        name = all_names[0]
        names = all_names[1:]
        invkind = desc[4]
        arg_defs = desc[2]
        ret_def = desc[8]

        self.dispid = dispid
        self.invkind = invkind
        # We dont use this ATM.
        #        self.ret = Arg(ret_def)
        if isEventSink and name[:2] != "On":
            name = "On%s" % name
        self.name = name
        cbArgs = 0
        self.args = []
        for argDesc in arg_defs:
            arg = Arg(argDesc)
            arg.offset = cbArgs
            cbArgs = cbArgs + arg.size
            self.args.append(arg)
        self.cbArgs = cbArgs
        self._gw_in_args = self._GenerateInArgTuple()
        self._gw_out_args = self._GenerateOutArgTuple()

    def _GenerateInArgTuple(self):
        # Given a method, generate the in argument tuple
        l = []
        for arg in self.args:
            if arg.inOut & pythoncom.PARAMFLAG_FIN or arg.inOut == 0:
                l.append((arg.vt, arg.offset, arg.size))
        return tuple(l)

    def _GenerateOutArgTuple(self):
        # Given a method, generate the out argument tuple
        l = []
        for arg in self.args:
            if (
                arg.inOut & pythoncom.PARAMFLAG_FOUT
                or arg.inOut & pythoncom.PARAMFLAG_FRETVAL
                or arg.inOut == 0
            ):
                l.append((arg.vt, arg.offset, arg.size, arg.clsid))
        return tuple(l)


class Definition:
    def __init__(self, iid, is_dispatch, method_defs):
        self._iid = iid
        self._methods = []
        self._is_dispatch = is_dispatch
        for info in method_defs:
            entry = Method(info)
            self._methods.append(entry)

    def iid(self):
        return self._iid

    def vtbl_argsizes(self):
        return [m.cbArgs for m in self._methods]

    def vtbl_argcounts(self):
        return [len(m.args) for m in self._methods]

    def dispatch(
        self,
        ob,
        index,
        argPtr,
        ReadFromInTuple=_univgw.ReadFromInTuple,
        WriteFromOutTuple=_univgw.WriteFromOutTuple,
    ):
        "Dispatch a call to an interface method."
        meth = self._methods[index]
        # Infer S_OK if they don't return anything bizarre.
        hr = 0
        args = ReadFromInTuple(meth._gw_in_args, argPtr)
        # If ob is a dispatcher, ensure a policy
        ob = getattr(ob, "policy", ob)
        # Ensure the correct dispid is setup
        ob._dispid_to_func_[meth.dispid] = meth.name
        retVal = ob._InvokeEx_(meth.dispid, 0, meth.invkind, args, None, None)
        # None is an allowed return value stating that
        # the code doesn't want to touch any output arguments.
        if type(retVal) == tuple:  # Like pythoncom, we special case a tuple.
            # However, if they want to return a specific HRESULT,
            # then they have to return all of the out arguments
            # AND the HRESULT.
            if len(retVal) == len(meth._gw_out_args) + 1:
                hr = retVal[0]
                retVal = retVal[1:]
            else:
                raise TypeError(
                    "Expected %s return values, got: %s"
                    % (len(meth._gw_out_args) + 1, len(retVal))
                )
        else:
            retVal = [retVal]
            retVal.extend([None] * (len(meth._gw_out_args) - 1))
            retVal = tuple(retVal)
        WriteFromOutTuple(retVal, meth._gw_out_args, argPtr)
        return hr
