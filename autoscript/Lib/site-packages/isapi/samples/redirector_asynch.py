# This is a sample ISAPI extension written in Python.

# This is like the other 'redirector' samples, but uses asnch IO when writing
# back to the client (it does *not* use asynch io talking to the remote
# server!)

import sys
import urllib.error
import urllib.parse
import urllib.request

from isapi import isapicon, threaded_extension

# sys.isapidllhandle will exist when we are loaded by the IIS framework.
# In this case we redirect our output to the win32traceutil collector.
if hasattr(sys, "isapidllhandle"):
    import win32traceutil

# The site we are proxying.
proxy = "http://www.python.org"

# We synchronously read chunks of this size then asynchronously write them.
CHUNK_SIZE = 8192


# The callback made when IIS completes the asynch write.
def io_callback(ecb, fp, cbIO, errcode):
    print("IO callback", ecb, fp, cbIO, errcode)
    chunk = fp.read(CHUNK_SIZE)
    if chunk:
        ecb.WriteClient(chunk, isapicon.HSE_IO_ASYNC)
        # and wait for the next callback to say this chunk is done.
    else:
        # eof - say we are complete.
        fp.close()
        ecb.DoneWithSession()


# The ISAPI extension - handles all requests in the site.
class Extension(threaded_extension.ThreadPoolExtension):
    "Python sample proxy server - asynch version."

    def Dispatch(self, ecb):
        print('IIS dispatching "%s"' % (ecb.GetServerVariable("URL"),))
        url = ecb.GetServerVariable("URL")

        new_url = proxy + url
        print("Opening %s" % new_url)
        fp = urllib.request.urlopen(new_url)
        headers = fp.info()
        ecb.SendResponseHeaders("200 OK", str(headers) + "\r\n", False)
        # now send the first chunk asynchronously
        ecb.ReqIOCompletion(io_callback, fp)
        chunk = fp.read(CHUNK_SIZE)
        if chunk:
            ecb.WriteClient(chunk, isapicon.HSE_IO_ASYNC)
            return isapicon.HSE_STATUS_PENDING
        # no data - just close things now.
        ecb.DoneWithSession()
        return isapicon.HSE_STATUS_SUCCESS


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
        Name="/",
        Description=Extension.__doc__,
        ScriptMaps=sm,
        ScriptMapUpdate="replace",
    )
    params.VirtualDirs = [vd]
    HandleCommandLine(params)
