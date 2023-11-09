# This is an ISAPI extension purely for testing purposes.  It is NOT
# a 'demo' (even though it may be useful!)
#
# Install this extension, then point your browser to:
# "http://localhost/pyisapi_test/test1"
# This will execute the method 'test1' below.  See below for the list of
# test methods that are acceptable.

import urllib.error
import urllib.parse
import urllib.request

# If we have no console (eg, am running from inside IIS), redirect output
# somewhere useful - in this case, the standard win32 trace collector.
import win32api
import winerror

from isapi import ExtensionError, isapicon, threaded_extension
from isapi.simple import SimpleFilter

try:
    win32api.GetConsoleTitle()
except win32api.error:
    # No console - redirect
    import win32traceutil


# The ISAPI extension - handles requests in our virtual dir, and sends the
# response to the client.
class Extension(threaded_extension.ThreadPoolExtension):
    "Python ISAPI Tester"

    def Dispatch(self, ecb):
        print('Tester dispatching "%s"' % (ecb.GetServerVariable("URL"),))
        url = ecb.GetServerVariable("URL")
        test_name = url.split("/")[-1]
        meth = getattr(self, test_name, None)
        if meth is None:
            raise AttributeError("No test named '%s'" % (test_name,))
        result = meth(ecb)
        if result is None:
            # This means the test finalized everything
            return
        ecb.SendResponseHeaders("200 OK", "Content-type: text/html\r\n\r\n", False)
        print("<HTML><BODY>Finished running test <i>", test_name, "</i>", file=ecb)
        print("<pre>", file=ecb)
        print(result, file=ecb)
        print("</pre>", file=ecb)
        print("</BODY></HTML>", file=ecb)
        ecb.DoneWithSession()

    def test1(self, ecb):
        try:
            ecb.GetServerVariable("foo bar")
            raise RuntimeError("should have failed!")
        except ExtensionError as err:
            assert err.errno == winerror.ERROR_INVALID_INDEX, err
        return "worked!"

    def test_long_vars(self, ecb):
        qs = ecb.GetServerVariable("QUERY_STRING")
        # Our implementation has a default buffer size of 8k - so we test
        # the code that handles an overflow by ensuring there are more
        # than 8k worth of chars in the URL.
        expected_query = "x" * 8500
        if len(qs) == 0:
            # Just the URL with no query part - redirect to myself, but with
            # a huge query portion.
            me = ecb.GetServerVariable("URL")
            headers = "Location: " + me + "?" + expected_query + "\r\n\r\n"
            ecb.SendResponseHeaders("301 Moved", headers)
            ecb.DoneWithSession()
            return None
        if qs == expected_query:
            return "Total length of variable is %d - test worked!" % (len(qs),)
        else:
            return "Unexpected query portion!  Got %d chars, expected %d" % (
                len(qs),
                len(expected_query),
            )

    def test_unicode_vars(self, ecb):
        # We need to check that we are running IIS6!  This seems the only
        # effective way from an extension.
        ver = float(ecb.GetServerVariable("SERVER_SOFTWARE").split("/")[1])
        if ver < 6.0:
            return "This is IIS version %g - unicode only works in IIS6 and later" % ver

        us = ecb.GetServerVariable("UNICODE_SERVER_NAME")
        if not isinstance(us, str):
            raise RuntimeError("unexpected type!")
        if us != str(ecb.GetServerVariable("SERVER_NAME")):
            raise RuntimeError("Unicode and non-unicode values were not the same")
        return "worked!"


# The entry points for the ISAPI extension.
def __ExtensionFactory__():
    return Extension()


if __name__ == "__main__":
    # If run from the command-line, install ourselves.
    from isapi.install import *

    params = ISAPIParameters()
    # Setup the virtual directories - this is a list of directories our
    # extension uses - in this case only 1.
    # Each extension has a "script map" - this is the mapping of ISAPI
    # extensions.
    sm = [ScriptMapParams(Extension="*", Flags=0)]
    vd = VirtualDirParameters(
        Name="pyisapi_test",
        Description=Extension.__doc__,
        ScriptMaps=sm,
        ScriptMapUpdate="replace",
    )
    params.VirtualDirs = [vd]
    HandleCommandLine(params)
