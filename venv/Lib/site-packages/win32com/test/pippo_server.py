# A little test server, complete with typelib, we can use for testing.
# Originally submitted with bug:
# [ 753154 ] memory leak wrapping object having _typelib_guid_ attribute
# but modified by mhammond for use as part of the test suite.
import os
import sys

import pythoncom
import win32com
import winerror
from win32com.server.util import wrap


class CPippo:
    #
    # COM declarations
    #
    _reg_clsid_ = "{1F0F75D6-BD63-41B9-9F88-2D9D2E1AA5C3}"
    _reg_desc_ = "Pippo Python test object"
    _reg_progid_ = "Python.Test.Pippo"
    # _reg_clsctx_ = pythoncom.CLSCTX_LOCAL_SERVER
    ###
    ### Link to typelib
    _typelib_guid_ = "{7783054E-9A20-4584-8C62-6ED2A08F6AC6}"
    _typelib_version_ = 1, 0
    _com_interfaces_ = ["IPippo"]

    def __init__(self):
        self.MyProp1 = 10

    def Method1(self):
        return wrap(CPippo())

    def Method2(self, in1, inout1):
        return in1, inout1 * 2

    def Method3(self, in1):
        # in1 will be a tuple, not a list.
        # Yet, we are not allowed to return a tuple, but need to convert it to a list first. (Bug?)
        return list(in1)


def BuildTypelib():
    from distutils.dep_util import newer

    this_dir = os.path.dirname(__file__)
    idl = os.path.abspath(os.path.join(this_dir, "pippo.idl"))
    tlb = os.path.splitext(idl)[0] + ".tlb"
    if newer(idl, tlb):
        print("Compiling %s" % (idl,))
        rc = os.system('midl "%s"' % (idl,))
        if rc:
            raise RuntimeError("Compiling MIDL failed!")
        # Can't work out how to prevent MIDL from generating the stubs.
        # just nuke them
        for fname in "dlldata.c pippo_i.c pippo_p.c pippo.h".split():
            os.remove(os.path.join(this_dir, fname))

    print("Registering %s" % (tlb,))
    tli = pythoncom.LoadTypeLib(tlb)
    pythoncom.RegisterTypeLib(tli, tlb)


def UnregisterTypelib():
    k = CPippo
    try:
        pythoncom.UnRegisterTypeLib(
            k._typelib_guid_,
            k._typelib_version_[0],
            k._typelib_version_[1],
            0,
            pythoncom.SYS_WIN32,
        )
        print("Unregistered typelib")
    except pythoncom.error as details:
        if details[0] == winerror.TYPE_E_REGISTRYACCESS:
            pass
        else:
            raise


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if "--unregister" in argv:
        # Unregister the type-libraries.
        UnregisterTypelib()
    else:
        # Build and register the type-libraries.
        BuildTypelib()
    import win32com.server.register

    win32com.server.register.UseCommandLine(CPippo)


if __name__ == "__main__":
    main(sys.argv)
