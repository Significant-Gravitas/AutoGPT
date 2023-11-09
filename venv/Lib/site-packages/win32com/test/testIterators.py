# Some raw iter tests.  Some "high-level" iterator tests can be found in
# testvb.py and testOutlook.py
import sys
import unittest

import pythoncom
import win32com.server.util
import win32com.test.util
from win32com.client import Dispatch
from win32com.client.gencache import EnsureDispatch


class _BaseTestCase(win32com.test.util.TestCase):
    def test_enumvariant_vb(self):
        ob, iter = self.iter_factory()
        got = []
        for v in iter:
            got.append(v)
        self.assertEqual(got, self.expected_data)

    def test_yield(self):
        ob, i = self.iter_factory()
        got = []
        for v in iter(i):
            got.append(v)
        self.assertEqual(got, self.expected_data)

    def _do_test_nonenum(self, object):
        try:
            for i in object:
                pass
            self.fail("Could iterate over a non-iterable object")
        except TypeError:
            pass  # this is expected.
        self.assertRaises(TypeError, iter, object)
        self.assertRaises(AttributeError, getattr, object, "next")

    def test_nonenum_wrapper(self):
        # Check our raw PyIDispatch
        ob = self.object._oleobj_
        try:
            for i in ob:
                pass
            self.fail("Could iterate over a non-iterable object")
        except TypeError:
            pass  # this is expected.
        self.assertRaises(TypeError, iter, ob)
        self.assertRaises(AttributeError, getattr, ob, "next")

        # And our Dispatch wrapper
        ob = self.object
        try:
            for i in ob:
                pass
            self.fail("Could iterate over a non-iterable object")
        except TypeError:
            pass  # this is expected.
        # Note that as our object may be dynamic, we *do* have a __getitem__
        # method, meaning we *can* call iter() on the object.  In this case
        # actual iteration is what fails.
        # So either the 'iter(); will raise a type error, or an attempt to
        # fetch it
        try:
            next(iter(ob))
            self.fail("Expected a TypeError fetching this iterator")
        except TypeError:
            pass
        # And it should never have a 'next' method
        self.assertRaises(AttributeError, getattr, ob, "next")


class VBTestCase(_BaseTestCase):
    def setUp(self):
        def factory():
            # Our VB test harness exposes a property with IEnumVariant.
            ob = self.object.EnumerableCollectionProperty
            for i in self.expected_data:
                ob.Add(i)
            # Get the raw IEnumVARIANT.
            invkind = pythoncom.DISPATCH_METHOD | pythoncom.DISPATCH_PROPERTYGET
            iter = ob._oleobj_.InvokeTypes(
                pythoncom.DISPID_NEWENUM, 0, invkind, (13, 10), ()
            )
            return ob, iter.QueryInterface(pythoncom.IID_IEnumVARIANT)

        # We *need* generated dispatch semantics, so dynamic __getitem__ etc
        # don't get in the way of our tests.
        self.object = EnsureDispatch("PyCOMVBTest.Tester")
        self.expected_data = [1, "Two", "3"]
        self.iter_factory = factory

    def tearDown(self):
        self.object = None


# Test our client semantics, but using a wrapped Python list object.
# This has the effect of re-using our client specific tests, but in this
# case is exercising the server side.
class SomeObject:
    _public_methods_ = ["GetCollection"]

    def __init__(self, data):
        self.data = data

    def GetCollection(self):
        return win32com.server.util.NewCollection(self.data)


class WrappedPythonCOMServerTestCase(_BaseTestCase):
    def setUp(self):
        def factory():
            ob = self.object.GetCollection()
            flags = pythoncom.DISPATCH_METHOD | pythoncom.DISPATCH_PROPERTYGET
            enum = ob._oleobj_.Invoke(pythoncom.DISPID_NEWENUM, 0, flags, 1)
            return ob, enum.QueryInterface(pythoncom.IID_IEnumVARIANT)

        self.expected_data = [1, "Two", 3]
        sv = win32com.server.util.wrap(SomeObject(self.expected_data))
        self.object = Dispatch(sv)
        self.iter_factory = factory

    def tearDown(self):
        self.object = None


def suite():
    # We dont want our base class run
    suite = unittest.TestSuite()
    for item in list(globals().values()):
        if (
            type(item) == type(unittest.TestCase)
            and issubclass(item, unittest.TestCase)
            and item != _BaseTestCase
        ):
            suite.addTest(unittest.makeSuite(item))
    return suite


if __name__ == "__main__":
    unittest.main(argv=sys.argv + ["suite"])
