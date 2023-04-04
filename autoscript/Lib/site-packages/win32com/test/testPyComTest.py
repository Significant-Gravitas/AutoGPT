# NOTE - Still seems to be a leak here somewhere
# gateway count doesnt hit zero.  Hence the print statements!

import sys

sys.coinit_flags = 0  # Must be free-threaded!
import datetime
import decimal
import os
import time

import pythoncom
import pywintypes
import win32api
import win32com
import win32com.client.connect
import win32timezone
import winerror
from pywin32_testutil import str2memory
from win32com.client import VARIANT, CastTo, DispatchBaseClass, constants
from win32com.test.util import CheckClean, RegisterPythonServer

importMsg = "**** PyCOMTest is not installed ***\n  PyCOMTest is a Python test specific COM client and server.\n  It is likely this server is not installed on this machine\n  To install the server, you must get the win32com sources\n  and build it using MS Visual C++"

error = Exception

# This test uses a Python implemented COM server - ensure correctly registered.
RegisterPythonServer(
    os.path.join(os.path.dirname(__file__), "..", "servers", "test_pycomtest.py"),
    "Python.Test.PyCOMTest",
)

from win32com.client import gencache

try:
    gencache.EnsureModule("{6BCDCB60-5605-11D0-AE5F-CADD4C000000}", 0, 1, 1)
except pythoncom.com_error:
    print("The PyCOMTest module can not be located or generated.")
    print(importMsg)
    raise RuntimeError(importMsg)

# We had a bg where RegisterInterfaces would fail if gencache had
# already been run - exercise that here
from win32com import universal

universal.RegisterInterfaces("{6BCDCB60-5605-11D0-AE5F-CADD4C000000}", 0, 1, 1)

verbose = 0


def check_get_set(func, arg):
    got = func(arg)
    if got != arg:
        raise error("%s failed - expected %r, got %r" % (func, arg, got))


def check_get_set_raises(exc, func, arg):
    try:
        got = func(arg)
    except exc as e:
        pass  # what we expect!
    else:
        raise error(
            "%s with arg %r didn't raise %s - returned %r" % (func, arg, exc, got)
        )


def progress(*args):
    if verbose:
        for arg in args:
            print(arg, end=" ")
        print()


def TestApplyResult(fn, args, result):
    try:
        fnName = str(fn).split()[1]
    except:
        fnName = str(fn)
    progress("Testing ", fnName)
    pref = "function " + fnName
    rc = fn(*args)
    if rc != result:
        raise error("%s failed - result not %r but %r" % (pref, result, rc))


def TestConstant(constName, pyConst):
    try:
        comConst = getattr(constants, constName)
    except:
        raise error("Constant %s missing" % (constName,))
    if comConst != pyConst:
        raise error(
            "Constant value wrong for %s - got %s, wanted %s"
            % (constName, comConst, pyConst)
        )


# Simple handler class.  This demo only fires one event.
class RandomEventHandler:
    def _Init(self):
        self.fireds = {}

    def OnFire(self, no):
        try:
            self.fireds[no] = self.fireds[no] + 1
        except KeyError:
            self.fireds[no] = 0

    def OnFireWithNamedParams(self, no, a_bool, out1, out2):
        # This test exists mainly to help with an old bug, where named
        # params would come in reverse.
        Missing = pythoncom.Missing
        if no is not Missing:
            # We know our impl called 'OnFire' with the same ID
            assert no in self.fireds
            assert no + 1 == out1, "expecting 'out1' param to be ID+1"
            assert no + 2 == out2, "expecting 'out2' param to be ID+2"
        # The middle must be a boolean.
        assert a_bool is Missing or type(a_bool) == bool, "middle param not a bool"
        return out1 + 2, out2 + 2

    def _DumpFireds(self):
        if not self.fireds:
            print("ERROR: Nothing was received!")
        for firedId, no in self.fireds.items():
            progress("ID %d fired %d times" % (firedId, no))


# A simple handler class that derives from object (ie, a "new style class") -
# only relevant for Python 2.x (ie, the 2 classes should be identical in 3.x)
class NewStyleRandomEventHandler(object):
    def _Init(self):
        self.fireds = {}

    def OnFire(self, no):
        try:
            self.fireds[no] = self.fireds[no] + 1
        except KeyError:
            self.fireds[no] = 0

    def OnFireWithNamedParams(self, no, a_bool, out1, out2):
        # This test exists mainly to help with an old bug, where named
        # params would come in reverse.
        Missing = pythoncom.Missing
        if no is not Missing:
            # We know our impl called 'OnFire' with the same ID
            assert no in self.fireds
            assert no + 1 == out1, "expecting 'out1' param to be ID+1"
            assert no + 2 == out2, "expecting 'out2' param to be ID+2"
        # The middle must be a boolean.
        assert a_bool is Missing or type(a_bool) == bool, "middle param not a bool"
        return out1 + 2, out2 + 2

    def _DumpFireds(self):
        if not self.fireds:
            print("ERROR: Nothing was received!")
        for firedId, no in self.fireds.items():
            progress("ID %d fired %d times" % (firedId, no))


# Test everything which can be tested using both the "dynamic" and "generated"
# COM objects (or when there are very subtle differences)
def TestCommon(o, is_generated):
    progress("Getting counter")
    counter = o.GetSimpleCounter()
    TestCounter(counter, is_generated)

    progress("Checking default args")
    rc = o.TestOptionals()
    if rc[:-1] != ("def", 0, 1) or abs(rc[-1] - 3.14) > 0.01:
        print(rc)
        raise error("Did not get the optional values correctly")
    rc = o.TestOptionals("Hi", 2, 3, 1.1)
    if rc[:-1] != ("Hi", 2, 3) or abs(rc[-1] - 1.1) > 0.01:
        print(rc)
        raise error("Did not get the specified optional values correctly")
    rc = o.TestOptionals2(0)
    if rc != (0, "", 1):
        print(rc)
        raise error("Did not get the optional2 values correctly")
    rc = o.TestOptionals2(1.1, "Hi", 2)
    if rc[1:] != ("Hi", 2) or abs(rc[0] - 1.1) > 0.01:
        print(rc)
        raise error("Did not get the specified optional2 values correctly")

    progress("Checking getting/passing IUnknown")
    check_get_set(o.GetSetUnknown, o)
    progress("Checking getting/passing IDispatch")
    # This might be called with either the interface or the CoClass - but these
    # functions always return from the interface.
    expected_class = o.__class__
    # CoClass instances have `default_interface`
    expected_class = getattr(expected_class, "default_interface", expected_class)
    if not isinstance(o.GetSetDispatch(o), expected_class):
        raise error("GetSetDispatch failed: %r" % (o.GetSetDispatch(o),))
    progress("Checking getting/passing IDispatch of known type")
    expected_class = o.__class__
    expected_class = getattr(expected_class, "default_interface", expected_class)
    if o.GetSetInterface(o).__class__ != expected_class:
        raise error("GetSetDispatch failed")

    progress("Checking misc args")
    check_get_set(o.GetSetVariant, 4)
    check_get_set(o.GetSetVariant, "foo")
    check_get_set(o.GetSetVariant, o)

    # signed/unsigned.
    check_get_set(o.GetSetInt, 0)
    check_get_set(o.GetSetInt, -1)
    check_get_set(o.GetSetInt, 1)

    check_get_set(o.GetSetUnsignedInt, 0)
    check_get_set(o.GetSetUnsignedInt, 1)
    check_get_set(o.GetSetUnsignedInt, 0x80000000)
    if o.GetSetUnsignedInt(-1) != 0xFFFFFFFF:
        # -1 is a special case - we accept a negative int (silently converting to
        # unsigned) but when getting it back we convert it to a long.
        raise error("unsigned -1 failed")

    check_get_set(o.GetSetLong, 0)
    check_get_set(o.GetSetLong, -1)
    check_get_set(o.GetSetLong, 1)

    check_get_set(o.GetSetUnsignedLong, 0)
    check_get_set(o.GetSetUnsignedLong, 1)
    check_get_set(o.GetSetUnsignedLong, 0x80000000)
    # -1 is a special case - see above.
    if o.GetSetUnsignedLong(-1) != 0xFFFFFFFF:
        raise error("unsigned -1 failed")

    # We want to explicitly test > 32 bits.  py3k has no 'maxint' and
    # 'maxsize+1' is no good on 64bit platforms as its 65 bits!
    big = 2147483647  # sys.maxint on py2k
    for l in big, big + 1, 1 << 65:
        check_get_set(o.GetSetVariant, l)

    progress("Checking structs")
    r = o.GetStruct()
    assert r.int_value == 99 and str(r.str_value) == "Hello from C++"
    assert o.DoubleString("foo") == "foofoo"

    progress("Checking var args")
    o.SetVarArgs("Hi", "There", "From", "Python", 1)
    if o.GetLastVarArgs() != ("Hi", "There", "From", "Python", 1):
        raise error("VarArgs failed -" + str(o.GetLastVarArgs()))

    progress("Checking arrays")
    l = []
    TestApplyResult(o.SetVariantSafeArray, (l,), len(l))
    l = [1, 2, 3, 4]
    TestApplyResult(o.SetVariantSafeArray, (l,), len(l))
    TestApplyResult(
        o.CheckVariantSafeArray,
        (
            (
                1,
                2,
                3,
                4,
            ),
        ),
        1,
    )

    # and binary
    TestApplyResult(o.SetBinSafeArray, (str2memory("foo\0bar"),), 7)

    progress("Checking properties")
    o.LongProp = 3
    if o.LongProp != 3 or o.IntProp != 3:
        raise error("Property value wrong - got %d/%d" % (o.LongProp, o.IntProp))
    o.LongProp = o.IntProp = -3
    if o.LongProp != -3 or o.IntProp != -3:
        raise error("Property value wrong - got %d/%d" % (o.LongProp, o.IntProp))
    # This number fits in an unsigned long.  Attempting to set it to a normal
    # long will involve overflow, which is to be expected. But we do
    # expect it to work in a property explicitly a VT_UI4.
    check = 3 * 10**9
    o.ULongProp = check
    if o.ULongProp != check:
        raise error(
            "Property value wrong - got %d (expected %d)" % (o.ULongProp, check)
        )

    TestApplyResult(o.Test, ("Unused", 99), 1)  # A bool function
    TestApplyResult(o.Test, ("Unused", -1), 1)  # A bool function
    TestApplyResult(o.Test, ("Unused", 1 == 1), 1)  # A bool function
    TestApplyResult(o.Test, ("Unused", 0), 0)
    TestApplyResult(o.Test, ("Unused", 1 == 0), 0)

    assert o.DoubleString("foo") == "foofoo"

    TestConstant("ULongTest1", 0xFFFFFFFF)
    TestConstant("ULongTest2", 0x7FFFFFFF)
    TestConstant("LongTest1", -0x7FFFFFFF)
    TestConstant("LongTest2", 0x7FFFFFFF)
    TestConstant("UCharTest", 255)
    TestConstant("CharTest", -1)
    # 'Hello World', but the 'r' is the "Registered" sign (\xae)
    TestConstant("StringTest", "Hello Wo\xaeld")

    progress("Checking dates and times")
    # For now *all* times passed must be tz-aware.
    now = win32timezone.now()
    # but conversion to and from a VARIANT loses sub-second...
    now = now.replace(microsecond=0)
    later = now + datetime.timedelta(seconds=1)
    TestApplyResult(o.EarliestDate, (now, later), now)

    # The below used to fail with `ValueError: microsecond must be in 0..999999` - see #1655
    # https://planetcalc.com/7027/ says that float is: Sun, 25 Mar 1951 7:23:49 am
    assert o.MakeDate(18712.308206013888) == datetime.datetime.fromisoformat(
        "1951-03-25 07:23:49+00:00"
    )

    progress("Checking currency")
    # currency.
    pythoncom.__future_currency__ = 1
    if o.CurrencyProp != 0:
        raise error("Expecting 0, got %r" % (o.CurrencyProp,))
    for val in ("1234.5678", "1234.56", "1234"):
        o.CurrencyProp = decimal.Decimal(val)
        if o.CurrencyProp != decimal.Decimal(val):
            raise error("%s got %r" % (val, o.CurrencyProp))
    v1 = decimal.Decimal("1234.5678")
    TestApplyResult(o.DoubleCurrency, (v1,), v1 * 2)

    v2 = decimal.Decimal("9012.3456")
    TestApplyResult(o.AddCurrencies, (v1, v2), v1 + v2)

    TestTrickyTypesWithVariants(o, is_generated)
    progress("Checking win32com.client.VARIANT")
    TestPyVariant(o, is_generated)


def TestTrickyTypesWithVariants(o, is_generated):
    # Test tricky stuff with type handling and generally only works with
    # "generated" support but can be worked around using VARIANT.
    if is_generated:
        got = o.TestByRefVariant(2)
    else:
        v = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_VARIANT, 2)
        o.TestByRefVariant(v)
        got = v.value
    if got != 4:
        raise error("TestByRefVariant failed")

    if is_generated:
        got = o.TestByRefString("Foo")
    else:
        v = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_BSTR, "Foo")
        o.TestByRefString(v)
        got = v.value
    if got != "FooFoo":
        raise error("TestByRefString failed")

    # check we can pass ints as a VT_UI1
    vals = [1, 2, 3, 4]
    if is_generated:
        arg = vals
    else:
        arg = VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_UI1, vals)
    TestApplyResult(o.SetBinSafeArray, (arg,), len(vals))

    # safearrays of doubles and floats
    vals = [0, 1.1, 2.2, 3.3]
    if is_generated:
        arg = vals
    else:
        arg = VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, vals)
    TestApplyResult(o.SetDoubleSafeArray, (arg,), len(vals))

    if is_generated:
        arg = vals
    else:
        arg = VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R4, vals)
    TestApplyResult(o.SetFloatSafeArray, (arg,), len(vals))

    vals = [1.1, 2.2, 3.3, 4.4]
    expected = (1.1 * 2, 2.2 * 2, 3.3 * 2, 4.4 * 2)
    if is_generated:
        TestApplyResult(o.ChangeDoubleSafeArray, (vals,), expected)
    else:
        arg = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_ARRAY | pythoncom.VT_R8, vals)
        o.ChangeDoubleSafeArray(arg)
        if arg.value != expected:
            raise error("ChangeDoubleSafeArray got the wrong value")

    if is_generated:
        got = o.DoubleInOutString("foo")
    else:
        v = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_BSTR, "foo")
        o.DoubleInOutString(v)
        got = v.value
    assert got == "foofoo", got

    val = decimal.Decimal("1234.5678")
    if is_generated:
        got = o.DoubleCurrencyByVal(val)
    else:
        v = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_CY, val)
        o.DoubleCurrencyByVal(v)
        got = v.value
    assert got == val * 2


def TestDynamic():
    progress("Testing Dynamic")
    import win32com.client.dynamic

    o = win32com.client.dynamic.DumbDispatch("PyCOMTest.PyCOMTest")
    TestCommon(o, False)

    counter = win32com.client.dynamic.DumbDispatch("PyCOMTest.SimpleCounter")
    TestCounter(counter, False)

    # Dynamic doesn't know this should be an int, so we get a COM
    # TypeMismatch error.
    try:
        check_get_set_raises(ValueError, o.GetSetInt, "foo")
        raise error("no exception raised")
    except pythoncom.com_error as exc:
        if exc.hresult != winerror.DISP_E_TYPEMISMATCH:
            raise

    arg1 = VARIANT(pythoncom.VT_R4 | pythoncom.VT_BYREF, 2.0)
    arg2 = VARIANT(pythoncom.VT_BOOL | pythoncom.VT_BYREF, True)
    arg3 = VARIANT(pythoncom.VT_I4 | pythoncom.VT_BYREF, 4)
    o.TestInOut(arg1, arg2, arg3)
    assert arg1.value == 4.0, arg1
    assert arg2.value == False
    assert arg3.value == 8

    # damn - props with params don't work for dynamic objects :(
    # o.SetParamProp(0, 1)
    # if o.ParamProp(0) != 1:
    #    raise RuntimeError, o.paramProp(0)


def TestGenerated():
    # Create an instance of the server.
    from win32com.client.gencache import EnsureDispatch

    o = EnsureDispatch("PyCOMTest.PyCOMTest")
    TestCommon(o, True)

    counter = EnsureDispatch("PyCOMTest.SimpleCounter")
    TestCounter(counter, True)

    # This dance lets us get a CoClass even though it's not explicitly registered.
    # This is `CoPyComTest`
    from win32com.client.CLSIDToClass import GetClass

    coclass_o = GetClass("{8EE0C520-5605-11D0-AE5F-CADD4C000000}")()
    TestCommon(coclass_o, True)

    # Test the regression reported in #1753
    assert bool(coclass_o)

    # This is `CoSimpleCounter` and the counter tests should work.
    coclass = GetClass("{B88DD310-BAE8-11D0-AE86-76F2C1000000}")()
    TestCounter(coclass, True)

    # XXX - this is failing in dynamic tests, but should work fine.
    i1, i2 = o.GetMultipleInterfaces()
    if not isinstance(i1, DispatchBaseClass) or not isinstance(i2, DispatchBaseClass):
        # Yay - is now an instance returned!
        raise error(
            "GetMultipleInterfaces did not return instances - got '%s', '%s'" % (i1, i2)
        )
    del i1
    del i2

    # Generated knows to only pass a 32bit int, so should fail.
    check_get_set_raises(OverflowError, o.GetSetInt, 0x80000000)
    check_get_set_raises(OverflowError, o.GetSetLong, 0x80000000)

    # Generated knows this should be an int, so raises ValueError
    check_get_set_raises(ValueError, o.GetSetInt, "foo")
    check_get_set_raises(ValueError, o.GetSetLong, "foo")

    # Pass some non-sequence objects to our array decoder, and watch it fail.
    try:
        o.SetVariantSafeArray("foo")
        raise error("Expected a type error")
    except TypeError:
        pass
    try:
        o.SetVariantSafeArray(666)
        raise error("Expected a type error")
    except TypeError:
        pass

    o.GetSimpleSafeArray(None)
    TestApplyResult(o.GetSimpleSafeArray, (None,), tuple(range(10)))
    resultCheck = tuple(range(5)), tuple(range(10)), tuple(range(20))
    TestApplyResult(o.GetSafeArrays, (None, None, None), resultCheck)

    l = []
    TestApplyResult(o.SetIntSafeArray, (l,), len(l))
    l = [1, 2, 3, 4]
    TestApplyResult(o.SetIntSafeArray, (l,), len(l))
    ll = [1, 2, 3, 0x100000000]
    TestApplyResult(o.SetLongLongSafeArray, (ll,), len(ll))
    TestApplyResult(o.SetULongLongSafeArray, (ll,), len(ll))

    # Tell the server to do what it does!
    TestApplyResult(o.Test2, (constants.Attr2,), constants.Attr2)
    TestApplyResult(o.Test3, (constants.Attr2,), constants.Attr2)
    TestApplyResult(o.Test4, (constants.Attr2,), constants.Attr2)
    TestApplyResult(o.Test5, (constants.Attr2,), constants.Attr2)

    TestApplyResult(o.Test6, (constants.WideAttr1,), constants.WideAttr1)
    TestApplyResult(o.Test6, (constants.WideAttr2,), constants.WideAttr2)
    TestApplyResult(o.Test6, (constants.WideAttr3,), constants.WideAttr3)
    TestApplyResult(o.Test6, (constants.WideAttr4,), constants.WideAttr4)
    TestApplyResult(o.Test6, (constants.WideAttr5,), constants.WideAttr5)

    TestApplyResult(o.TestInOut, (2.0, True, 4), (4.0, False, 8))

    o.SetParamProp(0, 1)
    if o.ParamProp(0) != 1:
        raise RuntimeError(o.paramProp(0))

    # Make sure CastTo works - even though it is only casting it to itself!
    o2 = CastTo(o, "IPyCOMTest")
    if o != o2:
        raise error("CastTo should have returned the same object")

    # Do the connection point thing...
    # Create a connection object.
    progress("Testing connection points")
    o2 = win32com.client.DispatchWithEvents(o, RandomEventHandler)
    TestEvents(o2, o2)
    o2 = win32com.client.DispatchWithEvents(o, NewStyleRandomEventHandler)
    TestEvents(o2, o2)
    # and a plain "WithEvents".
    handler = win32com.client.WithEvents(o, RandomEventHandler)
    TestEvents(o, handler)
    handler = win32com.client.WithEvents(o, NewStyleRandomEventHandler)
    TestEvents(o, handler)
    progress("Finished generated .py test.")


def TestEvents(o, handler):
    sessions = []
    handler._Init()
    try:
        for i in range(3):
            session = o.Start()
            sessions.append(session)
        time.sleep(0.5)
    finally:
        # Stop the servers
        for session in sessions:
            o.Stop(session)
        handler._DumpFireds()
        handler.close()


def _TestPyVariant(o, is_generated, val, checker=None):
    if is_generated:
        vt, got = o.GetVariantAndType(val)
    else:
        # Gotta supply all 3 args with the last 2 being explicit variants to
        # get the byref behaviour.
        var_vt = VARIANT(pythoncom.VT_UI2 | pythoncom.VT_BYREF, 0)
        var_result = VARIANT(pythoncom.VT_VARIANT | pythoncom.VT_BYREF, 0)
        o.GetVariantAndType(val, var_vt, var_result)
        vt = var_vt.value
        got = var_result.value
    if checker is not None:
        checker(got)
        return
    # default checking.
    assert vt == val.varianttype, (vt, val.varianttype)
    # Handle our safe-array test - if the passed value is a list of variants,
    # compare against the actual values.
    if type(val.value) in (tuple, list):
        check = [v.value if isinstance(v, VARIANT) else v for v in val.value]
        # pythoncom always returns arrays as tuples.
        got = list(got)
    else:
        check = val.value
    assert type(check) == type(got), (type(check), type(got))
    assert check == got, (check, got)


def _TestPyVariantFails(o, is_generated, val, exc):
    try:
        _TestPyVariant(o, is_generated, val)
        raise error("Setting %r didn't raise %s" % (val, exc))
    except exc:
        pass


def TestPyVariant(o, is_generated):
    _TestPyVariant(o, is_generated, VARIANT(pythoncom.VT_UI1, 1))
    _TestPyVariant(
        o, is_generated, VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_UI4, [1, 2, 3])
    )
    _TestPyVariant(o, is_generated, VARIANT(pythoncom.VT_BSTR, "hello"))
    _TestPyVariant(
        o,
        is_generated,
        VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_BSTR, ["hello", "there"]),
    )

    def check_dispatch(got):
        assert isinstance(got._oleobj_, pythoncom.TypeIIDs[pythoncom.IID_IDispatch])

    _TestPyVariant(o, is_generated, VARIANT(pythoncom.VT_DISPATCH, o), check_dispatch)
    _TestPyVariant(
        o, is_generated, VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_DISPATCH, [o])
    )
    # an array of variants each with a specific type.
    v = VARIANT(
        pythoncom.VT_ARRAY | pythoncom.VT_VARIANT,
        [
            VARIANT(pythoncom.VT_UI4, 1),
            VARIANT(pythoncom.VT_UI4, 2),
            VARIANT(pythoncom.VT_UI4, 3),
        ],
    )
    _TestPyVariant(o, is_generated, v)

    # and failures
    _TestPyVariantFails(o, is_generated, VARIANT(pythoncom.VT_UI1, "foo"), ValueError)


def TestCounter(counter, bIsGenerated):
    # Test random access into container
    progress("Testing counter", repr(counter))
    import random

    for i in range(50):
        num = int(random.random() * len(counter))
        try:
            # XXX - this appears broken by commit 08a14d4deb374eaa06378509cf44078ad467b9dc -
            # We shouldn't need to do generated differently than dynamic.
            if bIsGenerated:
                ret = counter.Item(num + 1)
            else:
                ret = counter[num]
            if ret != num + 1:
                raise error(
                    "Random access into element %d failed - return was %s"
                    % (num, repr(ret))
                )
        except IndexError:
            raise error("** IndexError accessing collection element %d" % num)

    num = 0
    if bIsGenerated:
        counter.SetTestProperty(1)
        counter.TestProperty = 1  # Note this has a second, default arg.
        counter.SetTestProperty(1, 2)
        if counter.TestPropertyWithDef != 0:
            raise error("Unexpected property set value!")
        if counter.TestPropertyNoDef(1) != 1:
            raise error("Unexpected property set value!")
    else:
        pass
        # counter.TestProperty = 1

    counter.LBound = 1
    counter.UBound = 10
    if counter.LBound != 1 or counter.UBound != 10:
        print("** Error - counter did not keep its properties")

    if bIsGenerated:
        bounds = counter.GetBounds()
        if bounds[0] != 1 or bounds[1] != 10:
            raise error("** Error - counter did not give the same properties back")
        counter.SetBounds(bounds[0], bounds[1])

    for item in counter:
        num = num + 1
    if num != len(counter):
        raise error("*** Length of counter and loop iterations dont match ***")
    if num != 10:
        raise error("*** Unexpected number of loop iterations ***")

    try:
        counter = iter(counter)._iter_.Clone()  # Test Clone() and enum directly
    except AttributeError:
        # *sob* - sometimes this is a real iterator and sometimes not :/
        progress("Finished testing counter (but skipped the iterator stuff")
        return
    counter.Reset()
    num = 0
    for item in counter:
        num = num + 1
    if num != 10:
        raise error("*** Unexpected number of loop iterations - got %d ***" % num)
    progress("Finished testing counter")


def TestLocalVTable(ob):
    # Python doesn't fully implement this interface.
    if ob.DoubleString("foo") != "foofoo":
        raise error("couldn't foofoo")


###############################
##
## Some vtable tests of the interface
##
def TestVTable(clsctx=pythoncom.CLSCTX_ALL):
    # Any vtable interfaces marked as dual *should* be able to be
    # correctly implemented as IDispatch.
    ob = win32com.client.Dispatch("Python.Test.PyCOMTest")
    TestLocalVTable(ob)
    # Now test it via vtable - use some C++ code to help here as Python can't do it directly yet.
    tester = win32com.client.Dispatch("PyCOMTest.PyCOMTest")
    testee = pythoncom.CoCreateInstance(
        "Python.Test.PyCOMTest", None, clsctx, pythoncom.IID_IUnknown
    )
    # check we fail gracefully with None passed.
    try:
        tester.TestMyInterface(None)
    except pythoncom.com_error as details:
        pass
    # and a real object.
    tester.TestMyInterface(testee)


def TestVTable2():
    # We once crashed creating our object with the native interface as
    # the first IID specified.  We must do it _after_ the tests, so that
    # Python has already had the gateway registered from last run.
    ob = win32com.client.Dispatch("Python.Test.PyCOMTest")
    iid = pythoncom.InterfaceNames["IPyCOMTest"]
    clsid = "Python.Test.PyCOMTest"
    clsctx = pythoncom.CLSCTX_SERVER
    try:
        testee = pythoncom.CoCreateInstance(clsid, None, clsctx, iid)
    except TypeError:
        # Python can't actually _use_ this interface yet, so this is
        # "expected".  Any COM error is not.
        pass


def TestVTableMI():
    clsctx = pythoncom.CLSCTX_SERVER
    ob = pythoncom.CoCreateInstance(
        "Python.Test.PyCOMTestMI", None, clsctx, pythoncom.IID_IUnknown
    )
    # This inherits from IStream.
    ob.QueryInterface(pythoncom.IID_IStream)
    # This implements IStorage, specifying the IID as a string
    ob.QueryInterface(pythoncom.IID_IStorage)
    # IDispatch should always work
    ob.QueryInterface(pythoncom.IID_IDispatch)

    iid = pythoncom.InterfaceNames["IPyCOMTest"]
    try:
        ob.QueryInterface(iid)
    except TypeError:
        # Python can't actually _use_ this interface yet, so this is
        # "expected".  Any COM error is not.
        pass


def TestQueryInterface(long_lived_server=0, iterations=5):
    tester = win32com.client.Dispatch("PyCOMTest.PyCOMTest")
    if long_lived_server:
        # Create a local server
        t0 = win32com.client.Dispatch(
            "Python.Test.PyCOMTest", clsctx=pythoncom.CLSCTX_LOCAL_SERVER
        )
    # Request custom interfaces a number of times
    prompt = [
        "Testing QueryInterface without long-lived local-server #%d of %d...",
        "Testing QueryInterface with long-lived local-server #%d of %d...",
    ]

    for i in range(iterations):
        progress(prompt[long_lived_server != 0] % (i + 1, iterations))
        tester.TestQueryInterface()


class Tester(win32com.test.util.TestCase):
    def testVTableInProc(self):
        # We used to crash running this the second time - do it a few times
        for i in range(3):
            progress("Testing VTables in-process #%d..." % (i + 1))
            TestVTable(pythoncom.CLSCTX_INPROC_SERVER)

    def testVTableLocalServer(self):
        for i in range(3):
            progress("Testing VTables out-of-process #%d..." % (i + 1))
            TestVTable(pythoncom.CLSCTX_LOCAL_SERVER)

    def testVTable2(self):
        for i in range(3):
            TestVTable2()

    def testVTableMI(self):
        for i in range(3):
            TestVTableMI()

    def testMultiQueryInterface(self):
        TestQueryInterface(0, 6)
        # When we use the custom interface in the presence of a long-lived
        # local server, i.e. a local server that is already running when
        # we request an instance of our COM object, and remains afterwards,
        # then after repeated requests to create an instance of our object
        # the custom interface disappears -- i.e. QueryInterface fails with
        # E_NOINTERFACE. Set the upper range of the following test to 2 to
        # pass this test, i.e. TestQueryInterface(1,2)
        TestQueryInterface(1, 6)

    def testDynamic(self):
        TestDynamic()

    def testGenerated(self):
        TestGenerated()


if __name__ == "__main__":
    # XXX - todo - Complete hack to crank threading support.
    # Should NOT be necessary
    def NullThreadFunc():
        pass

    import _thread

    _thread.start_new(NullThreadFunc, ())

    if "-v" in sys.argv:
        verbose = 1

    win32com.test.util.testmain()
