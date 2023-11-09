# errorSemantics.py

# Test the Python error handling semantics.  Specifically:
#
# * When a Python COM object is called via IDispatch, the nominated
#   scode is placed in the exception tuple, and the HRESULT is
#   DISP_E_EXCEPTION
# * When the same interface is called via IWhatever, the
#   nominated  scode is returned directly (with the scode also
#   reflected in the exception tuple)
# * In all cases, the description etc end up in the exception tuple
# * "Normal" Python exceptions resolve to an E_FAIL "internal error"

import pythoncom
import winerror
from win32com.client import Dispatch
from win32com.server.exception import COMException
from win32com.server.util import wrap
from win32com.test.util import CaptureWriter


class error(Exception):
    def __init__(self, msg, com_exception=None):
        Exception.__init__(self, msg, str(com_exception))


# Our COM server.
class TestServer:
    _public_methods_ = ["Clone", "Commit", "LockRegion", "Read"]
    _com_interfaces_ = [pythoncom.IID_IStream]

    def Clone(self):
        raise COMException("Not today", scode=winerror.E_UNEXPECTED)

    def Commit(self, flags):
        # Testing unicode: 1F600   '😀'; GRINNING FACE
        # Use the 'name' just for fun!
        if flags == 0:
            # A non com-specific exception.
            raise Exception("\N{GRINNING FACE}")
        # An explicit com_error, which is a bit of an edge-case, but might happen if
        # a COM server itself calls another COM object and it fails.
        excepinfo = (
            winerror.E_UNEXPECTED,
            "source",
            "\N{GRINNING FACE}",
            "helpfile",
            1,
            winerror.E_FAIL,
        )
        raise pythoncom.com_error(winerror.E_UNEXPECTED, "desc", excepinfo, None)


def test():
    # Call via a native interface.
    com_server = wrap(TestServer(), pythoncom.IID_IStream)
    try:
        com_server.Clone()
        raise error("Expecting this call to fail!")
    except pythoncom.com_error as com_exc:
        if com_exc.hresult != winerror.E_UNEXPECTED:
            raise error(
                "Calling the object natively did not yield the correct scode", com_exc
            )
        exc = com_exc.excepinfo
        if not exc or exc[-1] != winerror.E_UNEXPECTED:
            raise error(
                "The scode element of the exception tuple did not yield the correct scode",
                com_exc,
            )
        if exc[2] != "Not today":
            raise error(
                "The description in the exception tuple did not yield the correct string",
                com_exc,
            )
    cap = CaptureWriter()
    try:
        cap.capture()
        try:
            com_server.Commit(0)
        finally:
            cap.release()
        raise error("Expecting this call to fail!")
    except pythoncom.com_error as com_exc:
        if com_exc.hresult != winerror.E_FAIL:
            raise error("The hresult was not E_FAIL for an internal error", com_exc)
        if com_exc.excepinfo[1] != "Python COM Server Internal Error":
            raise error(
                "The description in the exception tuple did not yield the correct string",
                com_exc,
            )
    # Check we saw a traceback in stderr
    if cap.get_captured().find("Traceback") < 0:
        raise error("Could not find a traceback in stderr: %r" % (cap.get_captured(),))

    # Now do it all again, but using IDispatch
    com_server = Dispatch(wrap(TestServer()))
    try:
        com_server.Clone()
        raise error("Expecting this call to fail!")
    except pythoncom.com_error as com_exc:
        if com_exc.hresult != winerror.DISP_E_EXCEPTION:
            raise error(
                "Calling the object via IDispatch did not yield the correct scode",
                com_exc,
            )
        exc = com_exc.excepinfo
        if not exc or exc[-1] != winerror.E_UNEXPECTED:
            raise error(
                "The scode element of the exception tuple did not yield the correct scode",
                com_exc,
            )
        if exc[2] != "Not today":
            raise error(
                "The description in the exception tuple did not yield the correct string",
                com_exc,
            )

    cap.clear()
    try:
        cap.capture()
        try:
            com_server.Commit(0)
        finally:
            cap.release()
        raise error("Expecting this call to fail!")
    except pythoncom.com_error as com_exc:
        if com_exc.hresult != winerror.DISP_E_EXCEPTION:
            raise error(
                "Calling the object via IDispatch did not yield the correct scode",
                com_exc,
            )
        exc = com_exc.excepinfo
        if not exc or exc[-1] != winerror.E_FAIL:
            raise error(
                "The scode element of the exception tuple did not yield the correct scode",
                com_exc,
            )
        if exc[1] != "Python COM Server Internal Error":
            raise error(
                "The description in the exception tuple did not yield the correct string",
                com_exc,
            )
    # Check we saw a traceback in stderr
    if cap.get_captured().find("Traceback") < 0:
        raise error("Could not find a traceback in stderr: %r" % (cap.get_captured(),))

    # And an explicit com_error
    cap.clear()
    try:
        cap.capture()
        try:
            com_server.Commit(1)
        finally:
            cap.release()
        raise error("Expecting this call to fail!")
    except pythoncom.com_error as com_exc:
        if com_exc.hresult != winerror.DISP_E_EXCEPTION:
            raise error(
                "Calling the object via IDispatch did not yield the correct scode",
                com_exc,
            )
        exc = com_exc.excepinfo
        if not exc or exc[-1] != winerror.E_FAIL:
            raise error(
                "The scode element of the exception tuple did not yield the correct scode",
                com_exc,
            )
        if exc[1] != "source":
            raise error(
                "The source in the exception tuple did not yield the correct string",
                com_exc,
            )
        if exc[2] != "\U0001F600":
            raise error(
                "The description in the exception tuple did not yield the correct string",
                com_exc,
            )
        if exc[3] != "helpfile":
            raise error(
                "The helpfile in the exception tuple did not yield the correct string",
                com_exc,
            )
        if exc[4] != 1:
            raise error(
                "The help context in the exception tuple did not yield the correct string",
                com_exc,
            )


try:
    import logging
except ImportError:
    logging = None
if logging is not None:
    import win32com

    class TestLogHandler(logging.Handler):
        def __init__(self):
            self.reset()
            logging.Handler.__init__(self)

        def reset(self):
            self.num_emits = 0
            self.last_record = None

        def emit(self, record):
            self.num_emits += 1
            self.last_record = self.format(record)
            return
            print("--- record start")
            print(self.last_record)
            print("--- record end")

    def testLogger():
        assert not hasattr(win32com, "logger")
        handler = TestLogHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        log = logging.getLogger("win32com_test")
        log.addHandler(handler)
        win32com.logger = log
        # Now throw some exceptions!
        # Native interfaces
        com_server = wrap(TestServer(), pythoncom.IID_IStream)
        try:
            com_server.Commit(0)
            raise RuntimeError("should have failed")
        except pythoncom.error as exc:
            # `excepinfo` is a tuple with elt 2 being the traceback we captured.
            message = exc.excepinfo[2]
            assert message.endswith("Exception: \U0001F600\n")
        assert handler.num_emits == 1, handler.num_emits
        assert handler.last_record.startswith(
            "pythoncom error: Unexpected exception in gateway method 'Commit'"
        )
        handler.reset()

        # IDispatch
        com_server = Dispatch(wrap(TestServer()))
        try:
            com_server.Commit(0)
            raise RuntimeError("should have failed")
        except pythoncom.error as exc:
            # `excepinfo` is a tuple with elt 2 being the traceback we captured.
            message = exc.excepinfo[2]
            assert message.endswith("Exception: \U0001F600\n")
        assert handler.num_emits == 1, handler.num_emits
        handler.reset()


if __name__ == "__main__":
    test()
    if logging is not None:
        testLogger()
    from win32com.test.util import CheckClean

    CheckClean()
    print("error semantic tests worked")
