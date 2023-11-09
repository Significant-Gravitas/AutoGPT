# Test code for a VB Program.
#
# This requires the PythonCOM VB Test Harness.
#

import sys
import traceback

import pythoncom
import win32com.client
import win32com.client.dynamic
import win32com.client.gencache
import winerror
from pywin32_testutil import str2memory
from win32com.server.util import NewCollection, wrap
from win32com.test import util

# for debugging
useDispatcher = None
##  import win32com.server.dispatcher
##  useDispatcher = win32com.server.dispatcher.DefaultDebugDispatcher

error = RuntimeError


# Set up a COM object that VB will do some callbacks on.  This is used
# to test byref params for gateway IDispatch.
class TestObject:
    _public_methods_ = [
        "CallbackVoidOneByRef",
        "CallbackResultOneByRef",
        "CallbackVoidTwoByRef",
        "CallbackString",
        "CallbackResultOneByRefButReturnNone",
        "CallbackVoidOneByRefButReturnNone",
        "CallbackArrayResult",
        "CallbackArrayResultOneArrayByRef",
        "CallbackArrayResultWrongSize",
    ]

    def CallbackVoidOneByRef(self, intVal):
        return intVal + 1

    def CallbackResultOneByRef(self, intVal):
        return intVal, intVal + 1

    def CallbackVoidTwoByRef(self, int1, int2):
        return int1 + int2, int1 - int2

    def CallbackString(self, strVal):
        return 0, strVal + " has visited Python"

    def CallbackArrayResult(self, arrayVal):
        ret = []
        for i in arrayVal:
            ret.append(i + 1)
        # returning as a list forces it be processed as a single result
        # (rather than a tuple, where it may be interpreted as
        # multiple results for byref unpacking)
        return ret

    def CallbackArrayResultWrongSize(self, arrayVal):
        return list(arrayVal[:-1])

    def CallbackArrayResultOneArrayByRef(self, arrayVal):
        ret = []
        for i in arrayVal:
            ret.append(i + 1)
        # See above for list processing.
        return list(arrayVal), ret

    def CallbackResultOneByRefButReturnNone(self, intVal):
        return

    def CallbackVoidOneByRefButReturnNone(self, intVal):
        return


def TestVB(vbtest, bUseGenerated):
    vbtest.LongProperty = -1
    if vbtest.LongProperty != -1:
        raise error("Could not set the long property correctly.")
    vbtest.IntProperty = 10
    if vbtest.IntProperty != 10:
        raise error("Could not set the integer property correctly.")
    vbtest.VariantProperty = 10
    if vbtest.VariantProperty != 10:
        raise error("Could not set the variant integer property correctly.")
    vbtest.VariantProperty = str2memory("raw\0data")
    if vbtest.VariantProperty != str2memory("raw\0data"):
        raise error("Could not set the variant buffer property correctly.")
    vbtest.StringProperty = "Hello from Python"
    if vbtest.StringProperty != "Hello from Python":
        raise error("Could not set the string property correctly.")
    vbtest.VariantProperty = "Hello from Python"
    if vbtest.VariantProperty != "Hello from Python":
        raise error("Could not set the variant string property correctly.")
    vbtest.VariantProperty = (1.0, 2.0, 3.0)
    if vbtest.VariantProperty != (1.0, 2.0, 3.0):
        raise error(
            "Could not set the variant property to an array of floats correctly - '%s'."
            % (vbtest.VariantProperty,)
        )

    TestArrays(vbtest, bUseGenerated)
    TestStructs(vbtest)
    TestCollections(vbtest)

    assert vbtest.TakeByValObject(vbtest) == vbtest

    # Python doesnt support PUTREF properties without a typeref
    # (although we could)
    if bUseGenerated:
        ob = vbtest.TakeByRefObject(vbtest)
        assert ob[0] == vbtest and ob[1] == vbtest

        # A property that only has PUTREF defined.
        vbtest.VariantPutref = vbtest
        if vbtest.VariantPutref._oleobj_ != vbtest._oleobj_:
            raise error("Could not set the VariantPutref property correctly.")
        # Cant test further types for this VariantPutref, as only
        # COM objects can be stored ByRef.

        # A "set" type property - only works for generated.
        # VB recognizes a collection via a few "private" interfaces that we
        # could later build support in for.
        #               vbtest.CollectionProperty = NewCollection((1,2,"3", "Four"))
        #               if vbtest.CollectionProperty != (1,2,"3", "Four"):
        #                       raise error("Could not set the Collection property correctly - got back " + str(vbtest.CollectionProperty))

        # These are sub's that have a single byref param
        # Result should be just the byref.
        if vbtest.IncrementIntegerParam(1) != 2:
            raise error("Could not pass an integer byref")

        # Sigh - we cant have *both* "ommited byref" and optional args
        # We really have to opt that args nominated as optional work as optional
        # rather than simply all byrefs working as optional.
        #               if vbtest.IncrementIntegerParam() != 1:
        #                       raise error("Could not pass an omitted integer byref")

        if vbtest.IncrementVariantParam(1) != 2:
            raise error(
                "Could not pass an int VARIANT byref:"
                + str(vbtest.IncrementVariantParam(1))
            )

        if vbtest.IncrementVariantParam(1.5) != 2.5:
            raise error("Could not pass a float VARIANT byref")

        # Can't test IncrementVariantParam with the param omitted as it
        # it not declared in the VB code as "Optional"
        callback_ob = wrap(TestObject(), useDispatcher=useDispatcher)
        vbtest.DoSomeCallbacks(callback_ob)

    ret = vbtest.PassIntByVal(1)
    if ret != 2:
        raise error("Could not increment the integer - " + str(ret))

    TestVBInterface(vbtest)
    # Python doesnt support byrefs without some sort of generated support.
    if bUseGenerated:
        # This is a VB function that takes a single byref
        # Hence 2 return values - function and byref.
        ret = vbtest.PassIntByRef(1)
        if ret != (1, 2):
            raise error("Could not increment the integer - " + str(ret))
        # Check you can leave a byref arg blank.


# see above
#               ret = vbtest.PassIntByRef()
#               if ret != (0,1):
#                       raise error("Could not increment the integer with default arg- "+str(ret))


def _DoTestCollection(vbtest, col_name, expected):
    # It sucks that some objects allow "Count()", but others "Count"
    def _getcount(ob):
        r = getattr(ob, "Count")
        if type(r) != type(0):
            return r()
        return r

    c = getattr(vbtest, col_name)
    check = []
    for item in c:
        check.append(item)
    if check != list(expected):
        raise error(
            "Collection %s didn't have %r (had %r)" % (col_name, expected, check)
        )
    # Just looping over the collection again works (ie, is restartable)
    check = []
    for item in c:
        check.append(item)
    if check != list(expected):
        raise error(
            "Collection 2nd time around %s didn't have %r (had %r)"
            % (col_name, expected, check)
        )
    # Check we can get it via iter()
    i = iter(getattr(vbtest, col_name))
    check = []
    for item in i:
        check.append(item)
    if check != list(expected):
        raise error(
            "Collection iterator %s didn't have %r 2nd time around (had %r)"
            % (col_name, expected, check)
        )
    # but an iterator is not restartable
    check = []
    for item in i:
        check.append(item)
    if check != []:
        raise error(
            "2nd time around Collection iterator %s wasn't empty (had %r)"
            % (col_name, check)
        )

    # Check len()==Count()
    c = getattr(vbtest, col_name)
    if len(c) != _getcount(c):
        raise error(
            "Collection %s __len__(%r) wasn't==Count(%r)"
            % (col_name, len(c), _getcount(c))
        )
    # Check we can do it with zero based indexing.
    c = getattr(vbtest, col_name)
    check = []
    for i in range(_getcount(c)):
        check.append(c[i])
    if check != list(expected):
        raise error(
            "Collection %s didn't have %r (had %r)" % (col_name, expected, check)
        )

    # Check we can do it with our old "Skip/Next" methods.
    c = getattr(vbtest, col_name)._NewEnum()
    check = []
    while 1:
        n = c.Next()
        if not n:
            break
        check.append(n[0])
    if check != list(expected):
        raise error(
            "Collection %s didn't have %r (had %r)" % (col_name, expected, check)
        )


def TestCollections(vbtest):
    _DoTestCollection(vbtest, "CollectionProperty", [1, "Two", "3"])
    # zero based indexing works for simple VB collections.
    if vbtest.CollectionProperty[0] != 1:
        raise error("The CollectionProperty[0] element was not the default value")

    _DoTestCollection(vbtest, "EnumerableCollectionProperty", [])
    vbtest.EnumerableCollectionProperty.Add(1)
    vbtest.EnumerableCollectionProperty.Add("Two")
    vbtest.EnumerableCollectionProperty.Add("3")
    _DoTestCollection(vbtest, "EnumerableCollectionProperty", [1, "Two", "3"])


def _DoTestArray(vbtest, data, expected_exception=None):
    try:
        vbtest.ArrayProperty = data
        if expected_exception is not None:
            raise error("Expected '%s'" % expected_exception)
    except expected_exception:
        return
    got = vbtest.ArrayProperty
    if got != data:
        raise error(
            "Could not set the array data correctly - got %r, expected %r" % (got, data)
        )


def TestArrays(vbtest, bUseGenerated):
    # Try and use a safe array (note that the VB code has this declared as a VARIANT
    # and I cant work out how to force it to use native arrays!
    # (NOTE Python will convert incoming arrays to tuples, so we pass a tuple, even tho
    # a list works fine - just makes it easier for us to compare the result!
    # Empty array
    _DoTestArray(vbtest, ())
    # Empty child array
    _DoTestArray(vbtest, ((), ()))
    # ints
    _DoTestArray(vbtest, tuple(range(1, 100)))
    # Floats
    _DoTestArray(vbtest, (1.0, 2.0, 3.0))
    # Strings.
    _DoTestArray(vbtest, tuple("Hello from Python".split()))
    # Date and Time?
    # COM objects.
    _DoTestArray(vbtest, (vbtest, vbtest))
    # Mixed
    _DoTestArray(vbtest, (1, 2.0, "3"))
    # Array alements containing other arrays
    _DoTestArray(vbtest, (1, (vbtest, vbtest), ("3", "4")))
    # Multi-dimensional
    _DoTestArray(vbtest, (((1, 2, 3), (4, 5, 6))))
    _DoTestArray(vbtest, (((vbtest, vbtest, vbtest), (vbtest, vbtest, vbtest))))
    # Another dimension!
    arrayData = (((1, 2), (3, 4), (5, 6)), ((7, 8), (9, 10), (11, 12)))
    arrayData = (
        ((vbtest, vbtest), (vbtest, vbtest), (vbtest, vbtest)),
        ((vbtest, vbtest), (vbtest, vbtest), (vbtest, vbtest)),
    )
    _DoTestArray(vbtest, arrayData)

    # Check that when a '__getitem__ that fails' object is the first item
    # in the structure, we don't mistake it for a sequence.
    _DoTestArray(vbtest, (vbtest, 2.0, "3"))
    _DoTestArray(vbtest, (1, 2.0, vbtest))

    # Pass arbitrarily sized arrays - these used to fail, but thanks to
    # Stefan Schukat, they now work!
    expected_exception = None
    arrayData = (((1, 2, 1), (3, 4), (5, 6)), ((7, 8), (9, 10), (11, 12)))
    _DoTestArray(vbtest, arrayData, expected_exception)
    arrayData = (((vbtest, vbtest),), ((vbtest,),))
    _DoTestArray(vbtest, arrayData, expected_exception)
    # Pass bad data - last item wrong size
    arrayData = (((1, 2), (3, 4), (5, 6, 8)), ((7, 8), (9, 10), (11, 12)))
    _DoTestArray(vbtest, arrayData, expected_exception)

    # byref safearray results with incorrect size.
    callback_ob = wrap(TestObject(), useDispatcher=useDispatcher)
    print("** Expecting a 'ValueError' exception to be printed next:")
    try:
        vbtest.DoCallbackSafeArraySizeFail(callback_ob)
    except pythoncom.com_error as exc:
        assert (
            exc.excepinfo[1] == "Python COM Server Internal Error"
        ), "Didnt get the correct exception - '%s'" % (exc,)

    if bUseGenerated:
        # This one is a bit strange!  The array param is "ByRef", as VB insists.
        # The function itself also _returns_ the arram param.
        # Therefore, Python sees _2_ result values - one for the result,
        # and one for the byref.
        testData = "Mark was here".split()
        resultData, byRefParam = vbtest.PassSAFEARRAY(testData)
        if testData != list(resultData):
            raise error(
                "The safe array data was not what we expected - got " + str(resultData)
            )
        if testData != list(byRefParam):
            raise error(
                "The safe array data was not what we expected - got " + str(byRefParam)
            )
        testData = [1.0, 2.0, 3.0]
        resultData, byRefParam = vbtest.PassSAFEARRAYVariant(testData)
        assert testData == list(byRefParam)
        assert testData == list(resultData)
        testData = ["hi", "from", "Python"]
        resultData, byRefParam = vbtest.PassSAFEARRAYVariant(testData)
        assert testData == list(byRefParam), "Expected '%s', got '%s'" % (
            testData,
            list(byRefParam),
        )
        assert testData == list(resultData), "Expected '%s', got '%s'" % (
            testData,
            list(resultData),
        )
        # This time, instead of an explicit str() for 1.5, we just
        # pass Unicode, so the result should compare equal
        testData = [1, 2.0, "3"]
        resultData, byRefParam = vbtest.PassSAFEARRAYVariant(testData)
        assert testData == list(byRefParam)
        assert testData == list(resultData)
    print("Array tests passed")


def TestStructs(vbtest):
    try:
        vbtest.IntProperty = "One"
        raise error("Should have failed by now")
    except pythoncom.com_error as exc:
        if exc.hresult != winerror.DISP_E_TYPEMISMATCH:
            raise error("Expected DISP_E_TYPEMISMATCH")

    s = vbtest.StructProperty
    if s.int_val != 99 or str(s.str_val) != "hello":
        raise error("The struct value was not correct")
    s.str_val = "Hi from Python"
    s.int_val = 11
    if s.int_val != 11 or str(s.str_val) != "Hi from Python":
        raise error("The struct value didnt persist!")

    if s.sub_val.int_val != 66 or str(s.sub_val.str_val) != "sub hello":
        raise error("The sub-struct value was not correct")
    sub = s.sub_val
    sub.int_val = 22
    if sub.int_val != 22:
        print(sub.int_val)
        raise error("The sub-struct value didnt persist!")

    if s.sub_val.int_val != 22:
        print(s.sub_val.int_val)
        raise error("The sub-struct value (re-fetched) didnt persist!")

    if (
        s.sub_val.array_val[0].int_val != 0
        or str(s.sub_val.array_val[0].str_val) != "zero"
    ):
        print(s.sub_val.array_val[0].int_val)
        raise error("The array element wasnt correct")
    s.sub_val.array_val[0].int_val = 99
    s.sub_val.array_val[1].int_val = 66
    if s.sub_val.array_val[0].int_val != 99 or s.sub_val.array_val[1].int_val != 66:
        print(s.sub_val.array_val[0].int_val)
        raise error("The array element didnt persist.")
    # Now pass the struct back to VB
    vbtest.StructProperty = s
    # And get it back again
    s = vbtest.StructProperty
    if s.int_val != 11 or str(s.str_val) != "Hi from Python":
        raise error("After sending to VB, the struct value didnt persist!")
    if s.sub_val.array_val[0].int_val != 99:
        raise error("After sending to VB, the struct array value didnt persist!")

    # Now do some object equality tests.
    assert s == s
    assert s != None
    if sys.version_info > (3, 0):
        try:
            s < None
            raise error("Expected type error")
        except TypeError:
            pass
        try:
            None < s
            raise error("Expected type error")
        except TypeError:
            pass
    assert s != s.sub_val
    import copy

    s2 = copy.copy(s)
    assert s is not s2
    assert s == s2
    s2.int_val = 123
    assert s != s2
    # Make sure everything works with functions
    s2 = vbtest.GetStructFunc()
    assert s == s2
    vbtest.SetStructSub(s2)

    # Create a new structure, and set its elements.
    s = win32com.client.Record("VBStruct", vbtest)
    assert s.int_val == 0, "new struct inst initialized correctly!"
    s.int_val = -1
    vbtest.SetStructSub(s)
    assert vbtest.GetStructFunc().int_val == -1, "new struct didnt make the round trip!"
    # Finally, test stand-alone structure arrays.
    s_array = vbtest.StructArrayProperty
    assert s_array is None, "Expected None from the uninitialized VB array"
    vbtest.MakeStructArrayProperty(3)
    s_array = vbtest.StructArrayProperty
    assert len(s_array) == 3
    for i in range(len(s_array)):
        assert s_array[i].int_val == i
        assert s_array[i].sub_val.int_val == i
        assert s_array[i].sub_val.array_val[0].int_val == i
        assert s_array[i].sub_val.array_val[1].int_val == i + 1
        assert s_array[i].sub_val.array_val[2].int_val == i + 2

    # Some error type checks.
    try:
        s.bad_attribute
        raise RuntimeError("Could get a bad attribute")
    except AttributeError:
        pass
    m = s.__members__
    assert (
        m[0] == "int_val"
        and m[1] == "str_val"
        and m[2] == "ob_val"
        and m[3] == "sub_val"
    ), m

    # Test attribute errors.
    try:
        s.foo
        raise RuntimeError("Expected attribute error")
    except AttributeError as exc:
        assert "foo" in str(exc), exc

    # test repr - it uses repr() of the sub-objects, so check it matches.
    expected = "com_struct(int_val=%r, str_val=%r, ob_val=%r, sub_val=%r)" % (
        s.int_val,
        s.str_val,
        s.ob_val,
        s.sub_val,
    )
    if repr(s) != expected:
        print("Expected repr:", expected)
        print("Actual repr  :", repr(s))
        raise RuntimeError("repr() of record object failed")

    print("Struct/Record tests passed")


def TestVBInterface(ob):
    t = ob.GetInterfaceTester(2)
    if t.getn() != 2:
        raise error("Initial value wrong")
    t.setn(3)
    if t.getn() != 3:
        raise error("New value wrong")


def TestObjectSemantics(ob):
    # a convenient place to test some of our equality semantics
    assert ob == ob._oleobj_
    assert not ob != ob._oleobj_
    # same test again, but lhs and rhs reversed.
    assert ob._oleobj_ == ob
    assert not ob._oleobj_ != ob
    # same tests but against different pointers.  COM identity rules should
    # still ensure all works
    assert ob._oleobj_ == ob._oleobj_.QueryInterface(pythoncom.IID_IUnknown)
    assert not ob._oleobj_ != ob._oleobj_.QueryInterface(pythoncom.IID_IUnknown)

    assert ob._oleobj_ != None
    assert None != ob._oleobj_
    assert ob != None
    assert None != ob
    if sys.version_info > (3, 0):
        try:
            ob < None
            raise error("Expected type error")
        except TypeError:
            pass
        try:
            None < ob
            raise error("Expected type error")
        except TypeError:
            pass

    assert ob._oleobj_.QueryInterface(pythoncom.IID_IUnknown) == ob._oleobj_
    assert not ob._oleobj_.QueryInterface(pythoncom.IID_IUnknown) != ob._oleobj_

    assert ob._oleobj_ == ob._oleobj_.QueryInterface(pythoncom.IID_IDispatch)
    assert not ob._oleobj_ != ob._oleobj_.QueryInterface(pythoncom.IID_IDispatch)

    assert ob._oleobj_.QueryInterface(pythoncom.IID_IDispatch) == ob._oleobj_
    assert not ob._oleobj_.QueryInterface(pythoncom.IID_IDispatch) != ob._oleobj_

    print("Object semantic tests passed")


def DoTestAll():
    o = win32com.client.Dispatch("PyCOMVBTest.Tester")
    TestObjectSemantics(o)
    TestVB(o, 1)

    o = win32com.client.dynamic.DumbDispatch("PyCOMVBTest.Tester")
    TestObjectSemantics(o)
    TestVB(o, 0)


def TestAll():
    # Import the type library for the test module.  Let the 'invalid clsid'
    # exception filter up, where the test runner will treat it as 'skipped'
    win32com.client.gencache.EnsureDispatch("PyCOMVBTest.Tester")

    if not __debug__:
        raise RuntimeError("This must be run in debug mode - we use assert!")
    try:
        DoTestAll()
        print("All tests appear to have worked!")
    except:
        # ?????
        print("TestAll() failed!!")
        traceback.print_exc()
        raise


# Make this test run under our test suite to leak tests etc work
def suite():
    import unittest

    test = util.CapturingFunctionTestCase(TestAll, description="VB tests")
    suite = unittest.TestSuite()
    suite.addTest(test)
    return suite


if __name__ == "__main__":
    util.testmain()
