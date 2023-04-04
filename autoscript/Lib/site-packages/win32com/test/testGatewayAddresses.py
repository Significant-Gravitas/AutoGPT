# The purpose of this test is to ensure that the gateways objects
# do the right thing WRT COM rules about object identity etc.

# Also includes a basic test that we support inheritance correctly in
# gateway interfaces.

# For our test, we create an object of type IID_IPersistStorage
# This interface derives from IPersist.
# Therefore, QI's for IID_IDispatch, IID_IUnknown, IID_IPersist and
# IID_IPersistStorage should all return the same gateway object.
#
# In addition, the interface should only need to declare itself as
# using the IPersistStorage interface, and as the gateway derives
# from IPersist, it should automatically be available without declaration.
#
# We also create an object of type IID_I??, and perform a QI for it.
# We then jump through a number of hoops, ensuring that the objects
# returned by the QIs follow all the rules.
#
# Here is Gregs summary of the rules:
# 1) the set of supported interfaces is static and unchanging
# 2) symmetric: if you QI an interface for that interface, it succeeds
# 3) reflexive: if you QI against A for B, the new pointer must succeed
#   for a QI for A
# 4) transitive: if you QI for B, then QI that for C, then QI'ing A for C
#   must succeed
#
#
# Note that 1) Requires cooperation of the Python programmer.  The rule to keep is:
# "whenever you return an _object_ from _query_interface_(), you must return the
# same object each time for a given IID.  Note that you must return the same
# _wrapped_ object
# you
# The rest are tested here.


import pythoncom
from win32com.server.util import wrap

from .util import CheckClean

numErrors = 0


# Check that the 2 objects both have identical COM pointers.
def CheckSameCOMObject(ob1, ob2):
    addr1 = repr(ob1).split()[6][:-1]
    addr2 = repr(ob2).split()[6][:-1]
    return addr1 == addr2


# Check that the objects conform to COM identity rules.
def CheckObjectIdentity(ob1, ob2):
    u1 = ob1.QueryInterface(pythoncom.IID_IUnknown)
    u2 = ob2.QueryInterface(pythoncom.IID_IUnknown)
    return CheckSameCOMObject(u1, u2)


def FailObjectIdentity(ob1, ob2, when):
    if not CheckObjectIdentity(ob1, ob2):
        global numErrors
        numErrors = numErrors + 1
        print(when, "are not identical (%s, %s)" % (repr(ob1), repr(ob2)))


class Dummy:
    _public_methods_ = []  # We never attempt to make a call on this object.
    _com_interfaces_ = [pythoncom.IID_IPersistStorage]


class Dummy2:
    _public_methods_ = []  # We never attempt to make a call on this object.
    _com_interfaces_ = [
        pythoncom.IID_IPersistStorage,
        pythoncom.IID_IExternalConnection,
    ]


class DeletgatedDummy:
    _public_methods_ = []


class Dummy3:
    _public_methods_ = []  # We never attempt to make a call on this object.
    _com_interfaces_ = [pythoncom.IID_IPersistStorage]

    def _query_interface_(self, iid):
        if iid == pythoncom.IID_IExternalConnection:
            # This will NEVER work - can only wrap the object once!
            return wrap(DelegatedDummy())


def TestGatewayInheritance():
    # By default, wrap() creates and discards a temporary object.
    # This is not necessary, but just the current implementation of wrap.
    # As the object is correctly discarded, it doesnt affect this test.
    o = wrap(Dummy(), pythoncom.IID_IPersistStorage)
    o2 = o.QueryInterface(pythoncom.IID_IUnknown)
    FailObjectIdentity(o, o2, "IID_IPersistStorage->IID_IUnknown")

    o3 = o2.QueryInterface(pythoncom.IID_IDispatch)

    FailObjectIdentity(o2, o3, "IID_IUnknown->IID_IDispatch")
    FailObjectIdentity(o, o3, "IID_IPersistStorage->IID_IDispatch")

    o4 = o3.QueryInterface(pythoncom.IID_IPersistStorage)
    FailObjectIdentity(o, o4, "IID_IPersistStorage->IID_IPersistStorage(2)")
    FailObjectIdentity(o2, o4, "IID_IUnknown->IID_IPersistStorage(2)")
    FailObjectIdentity(o3, o4, "IID_IDispatch->IID_IPersistStorage(2)")

    o5 = o4.QueryInterface(pythoncom.IID_IPersist)
    FailObjectIdentity(o, o5, "IID_IPersistStorage->IID_IPersist")
    FailObjectIdentity(o2, o5, "IID_IUnknown->IID_IPersist")
    FailObjectIdentity(o3, o5, "IID_IDispatch->IID_IPersist")
    FailObjectIdentity(o4, o5, "IID_IPersistStorage(2)->IID_IPersist")


def TestMultiInterface():
    o = wrap(Dummy2(), pythoncom.IID_IPersistStorage)
    o2 = o.QueryInterface(pythoncom.IID_IExternalConnection)

    FailObjectIdentity(o, o2, "IID_IPersistStorage->IID_IExternalConnection")

    # Make the same QI again, to make sure it is stable.
    o22 = o.QueryInterface(pythoncom.IID_IExternalConnection)
    FailObjectIdentity(o, o22, "IID_IPersistStorage->IID_IExternalConnection")
    FailObjectIdentity(
        o2, o22, "IID_IPersistStorage->IID_IExternalConnection (stability)"
    )

    o3 = o2.QueryInterface(pythoncom.IID_IPersistStorage)
    FailObjectIdentity(o2, o3, "IID_IExternalConnection->IID_IPersistStorage")
    FailObjectIdentity(
        o, o3, "IID_IPersistStorage->IID_IExternalConnection->IID_IPersistStorage"
    )


def test():
    TestGatewayInheritance()
    TestMultiInterface()
    if numErrors == 0:
        print("Worked ok")
    else:
        print("There were", numErrors, "errors.")


if __name__ == "__main__":
    test()
    CheckClean()
