import sys
import unittest

import pythoncom
from win32com.client import Dispatch
from win32com.client.gencache import EnsureDispatch


class PippoTester(unittest.TestCase):
    def setUp(self):
        from win32com.test import pippo_server
        from win32com.test.util import RegisterPythonServer

        RegisterPythonServer(pippo_server.__file__, "Python.Test.Pippo")
        # create it.
        self.object = Dispatch("Python.Test.Pippo")

    def testLeaks(self):
        try:
            gtrc = sys.gettotalrefcount
        except AttributeError:
            print("Please run this with python_d for leak tests")
            gtrc = lambda: 0
        # note creating self.object() should have consumed our "one time" leaks
        self.object.Method1()
        start = gtrc()
        for i in range(1000):
            object = Dispatch("Python.Test.Pippo")
            object.Method1()
        object = None
        end = gtrc()
        if end - start > 5:
            self.fail("We lost %d references!" % (end - start,))

    def testResults(self):
        rc, out1 = self.object.Method2(123, 111)
        self.assertEqual(rc, 123)
        self.assertEqual(out1, 222)

    def testPythonArrays(self):
        self._testArray([-3, -2, -1, 0, 1, 2, 3])
        self._testArray([-3.14, -2, -0.1, 0.0, 1.1, 2.5, 3])

    def testNumpyArrays(self):
        try:
            import numpy
        except:
            print("Numpy test not possible because numpy module failed to import")
            return
        self._testArray(numpy.array([-3, -2, -1, 0, 1, 2, 3]))
        self._testArray(numpy.array([-3.14, -2, -0.1, 0.0, 1.1, 2.5, 3]))

    def testByteArrays(self):
        if "bytes" in dir(__builtins__):
            # Use eval to avoid compilation error in Python 2.
            self._testArray(eval("b'abcdef'"))
            self._testArray(eval("bytearray(b'abcdef')"))

    def _testArray(self, inArray):
        outArray = self.object.Method3(inArray)
        self.assertEqual(list(outArray), list(inArray))

    def testLeaksGencache(self):
        try:
            gtrc = sys.gettotalrefcount
        except AttributeError:
            print("Please run this with python_d for leak tests")
            gtrc = lambda: 0
        # note creating self.object() should have consumed our "one time" leaks
        object = EnsureDispatch("Python.Test.Pippo")
        start = gtrc()
        for i in range(1000):
            object = EnsureDispatch("Python.Test.Pippo")
            object.Method1()
        object = None
        end = gtrc()
        if end - start > 10:
            self.fail("We lost %d references!" % (end - start,))


if __name__ == "__main__":
    unittest.main()
