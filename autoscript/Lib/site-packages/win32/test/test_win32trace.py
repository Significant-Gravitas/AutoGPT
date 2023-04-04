import os
import sys
import threading
import time
import unittest

import win32trace
from pywin32_testutil import TestSkipped

if __name__ == "__main__":
    this_file = sys.argv[0]
else:
    this_file = __file__


def SkipIfCI():
    # This test often fails in CI, probably when it is being run multiple times
    # (ie, for different Python versions)
    # Github actions always have a `CI` variable.
    if "CI" in os.environ:
        raise TestSkipped("We skip this test on CI")


def CheckNoOtherReaders():
    win32trace.write("Hi")
    time.sleep(0.05)
    if win32trace.read() != "Hi":
        # Reset everything so following tests still fail with this error!
        win32trace.TermRead()
        win32trace.TermWrite()
        raise RuntimeError(
            "An existing win32trace reader appears to be "
            "running - please stop this process and try again"
        )


class TestInitOps(unittest.TestCase):
    def setUp(self):
        SkipIfCI()
        # clear old data
        win32trace.InitRead()
        win32trace.read()
        win32trace.TermRead()

    def tearDown(self):
        try:
            win32trace.TermRead()
        except win32trace.error:
            pass
        try:
            win32trace.TermWrite()
        except win32trace.error:
            pass

    def testInitTermRead(self):
        self.assertRaises(win32trace.error, win32trace.read)
        win32trace.InitRead()
        result = win32trace.read()
        self.assertEqual(result, "")
        win32trace.TermRead()
        self.assertRaises(win32trace.error, win32trace.read)

        win32trace.InitRead()
        self.assertRaises(win32trace.error, win32trace.InitRead)
        win32trace.InitWrite()
        self.assertRaises(win32trace.error, win32trace.InitWrite)
        win32trace.TermWrite()
        win32trace.TermRead()

    def testInitTermWrite(self):
        self.assertRaises(win32trace.error, win32trace.write, "Hei")
        win32trace.InitWrite()
        win32trace.write("Johan Galtung")
        win32trace.TermWrite()
        self.assertRaises(win32trace.error, win32trace.write, "Hei")

    def testTermSematics(self):
        win32trace.InitWrite()
        win32trace.write("Ta da")

        # if we both Write and Read are terminated at the same time,
        # we lose the data as the win32 object is closed.  Note that
        # if another writer is running, we do *not* lose the data - so
        # test for either the correct data or an empty string
        win32trace.TermWrite()
        win32trace.InitRead()
        self.assertTrue(win32trace.read() in ("Ta da", ""))
        win32trace.TermRead()

        # we keep the data because we init read before terminating write
        win32trace.InitWrite()
        win32trace.write("Ta da")
        win32trace.InitRead()
        win32trace.TermWrite()
        self.assertEqual("Ta da", win32trace.read())
        win32trace.TermRead()


class BasicSetupTearDown(unittest.TestCase):
    def setUp(self):
        SkipIfCI()
        win32trace.InitRead()
        # If any other writers are running (even if not actively writing),
        # terminating the module will *not* close the handle, meaning old data
        # will remain. This can cause other tests to fail.
        win32trace.read()
        win32trace.InitWrite()

    def tearDown(self):
        win32trace.TermWrite()
        win32trace.TermRead()


class TestModuleOps(BasicSetupTearDown):
    def testRoundTrip(self):
        win32trace.write("Syver Enstad")
        syverEnstad = win32trace.read()
        self.assertEqual("Syver Enstad", syverEnstad)

    def testRoundTripUnicode(self):
        win32trace.write("\xa9opyright Syver Enstad")
        syverEnstad = win32trace.read()
        # str objects are always returned in py2k (latin-1 encoding was used
        # on unicode objects)
        self.assertEqual("\xa9opyright Syver Enstad", syverEnstad)

    def testBlockingRead(self):
        win32trace.write("Syver Enstad")
        self.assertEqual("Syver Enstad", win32trace.blockingread())

    def testBlockingReadUnicode(self):
        win32trace.write("\xa9opyright Syver Enstad")
        # str objects are always returned in py2k (latin-1 encoding was used
        # on unicode objects)
        self.assertEqual("\xa9opyright Syver Enstad", win32trace.blockingread())

    def testFlush(self):
        win32trace.flush()


class TestTraceObjectOps(BasicSetupTearDown):
    def testInit(self):
        win32trace.TermRead()
        win32trace.TermWrite()
        traceObject = win32trace.GetTracer()
        self.assertRaises(win32trace.error, traceObject.read)
        self.assertRaises(win32trace.error, traceObject.write, "")
        win32trace.InitRead()
        win32trace.InitWrite()
        self.assertEqual("", traceObject.read())
        traceObject.write("Syver")

    def testFlush(self):
        traceObject = win32trace.GetTracer()
        traceObject.flush()

    def testIsatty(self):
        tracer = win32trace.GetTracer()
        assert tracer.isatty() == False

    def testRoundTrip(self):
        traceObject = win32trace.GetTracer()
        traceObject.write("Syver Enstad")
        self.assertEqual("Syver Enstad", traceObject.read())


class WriterThread(threading.Thread):
    def run(self):
        self.writeCount = 0
        for each in range(self.BucketCount):
            win32trace.write(str(each))
        self.writeCount = self.BucketCount

    def verifyWritten(self):
        return self.writeCount == self.BucketCount


class TestMultipleThreadsWriting(unittest.TestCase):
    # FullBucket is the thread count
    FullBucket = 50
    BucketCount = 9  # buckets must be a single digit number (ie. less than 10)

    def setUp(self):
        SkipIfCI()
        WriterThread.BucketCount = self.BucketCount
        win32trace.InitRead()
        win32trace.read()  # clear any old data.
        win32trace.InitWrite()
        CheckNoOtherReaders()
        self.threads = [WriterThread() for each in range(self.FullBucket)]
        self.buckets = list(range(self.BucketCount))
        for each in self.buckets:
            self.buckets[each] = 0

    def tearDown(self):
        win32trace.TermRead()
        win32trace.TermWrite()

    def areBucketsFull(self):
        bucketsAreFull = True
        for each in self.buckets:
            assert each <= self.FullBucket, each
            if each != self.FullBucket:
                bucketsAreFull = False
                break
        return bucketsAreFull

    def read(self):
        while 1:
            readString = win32trace.blockingread()
            for ch in readString:
                integer = int(ch)
                count = self.buckets[integer]
                assert count != -1
                self.buckets[integer] = count + 1
                if self.buckets[integer] == self.FullBucket:
                    if self.areBucketsFull():
                        return

    def testThreads(self):
        for each in self.threads:
            each.start()
        self.read()
        for each in self.threads:
            each.join()
        for each in self.threads:
            assert each.verifyWritten()
        assert self.areBucketsFull()


class TestHugeChunks(unittest.TestCase):
    # BiggestChunk is the size where we stop stressing the writer
    BiggestChunk = 2**16  # 256k should do it.

    def setUp(self):
        SkipIfCI()
        win32trace.InitRead()
        win32trace.read()  # clear any old data
        win32trace.InitWrite()

    def testHugeChunks(self):
        data = "*" * 1023 + "\n"
        while len(data) <= self.BiggestChunk:
            win32trace.write(data)
            data = data + data
        # If we made it here, we passed.

    def tearDown(self):
        win32trace.TermRead()
        win32trace.TermWrite()


import win32event
import win32process


class TraceWriteProcess:
    def __init__(self, threadCount):
        self.exitCode = -1
        self.threadCount = threadCount

    def start(self):
        procHandle, threadHandle, procId, threadId = win32process.CreateProcess(
            None,  # appName
            'python.exe "%s" /run_test_process %s %s'
            % (this_file, self.BucketCount, self.threadCount),
            None,  # process security
            None,  # thread security
            0,  # inherit handles
            win32process.NORMAL_PRIORITY_CLASS,
            None,  # new environment
            None,  # Current directory
            win32process.STARTUPINFO(),  # startup info
        )
        self.processHandle = procHandle

    def join(self):
        win32event.WaitForSingleObject(self.processHandle, win32event.INFINITE)
        self.exitCode = win32process.GetExitCodeProcess(self.processHandle)

    def verifyWritten(self):
        return self.exitCode == 0


class TestOutofProcess(unittest.TestCase):
    BucketCount = 9
    FullBucket = 50

    def setUp(self):
        SkipIfCI()
        win32trace.InitRead()
        TraceWriteProcess.BucketCount = self.BucketCount
        self.setUpWriters()
        self.buckets = list(range(self.BucketCount))
        for each in self.buckets:
            self.buckets[each] = 0

    def tearDown(self):
        win32trace.TermRead()

    def setUpWriters(self):
        self.processes = []
        # 5 processes, quot threads in each process
        quot, remainder = divmod(self.FullBucket, 5)
        for each in range(5):
            self.processes.append(TraceWriteProcess(quot))
        if remainder:
            self.processes.append(TraceWriteProcess(remainder))

    def areBucketsFull(self):
        bucketsAreFull = True
        for each in self.buckets:
            assert each <= self.FullBucket, each
            if each != self.FullBucket:
                bucketsAreFull = False
                break
        return bucketsAreFull

    def read(self):
        while 1:
            readString = win32trace.blockingread()
            for ch in readString:
                integer = int(ch)
                count = self.buckets[integer]
                assert count != -1
                self.buckets[integer] = count + 1
                if self.buckets[integer] == self.FullBucket:
                    if self.areBucketsFull():
                        return

    def testProcesses(self):
        for each in self.processes:
            each.start()
        self.read()
        for each in self.processes:
            each.join()
        for each in self.processes:
            assert each.verifyWritten()
        assert self.areBucketsFull()


def _RunAsTestProcess():
    # Run as an external process by the main tests.
    WriterThread.BucketCount = int(sys.argv[2])
    threadCount = int(sys.argv[3])
    threads = [WriterThread() for each in range(threadCount)]
    win32trace.InitWrite()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for t in threads:
        if not t.verifyWritten():
            sys.exit(-1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["/run_test_process"]:
        _RunAsTestProcess()
        sys.exit(0)
    # If some other win32traceutil reader is running, these tests fail
    # badly (as the other reader sometimes sees the output!)
    win32trace.InitRead()
    win32trace.InitWrite()
    CheckNoOtherReaders()
    # reset state so test env is back to normal
    win32trace.TermRead()
    win32trace.TermWrite()
    unittest.main()
