# This is a sample configuration file for an ISAPI filter and extension
# written in Python.
#
# Please see README.txt in this directory, and specifically the
# information about the "loader" DLL - installing this sample will create
# "_redirector_with_filter.dll" in the current directory.  The readme explains
# this.

# Executing this script (or any server config script) will install the extension
# into your web server. As the server executes, the PyISAPI framework will load
# this module and create your Extension and Filter objects.

# This sample provides sample redirector:
# It is implemented by a filter and an extension, so that some requests can
# be ignored.  Compare with 'redirector_simple' which avoids the filter, but
# is unable to selectively ignore certain requests.
# The process is sample uses is:
# * The filter is installed globally, as all filters are.
# * A Virtual Directory named "python" is setup.  This dir has our ISAPI
#   extension as the only application, mapped to file-extension '*'.  Thus, our
#   extension handles *all* requests in this directory.
# The basic process is that the filter does URL rewriting, redirecting every
# URL to our Virtual Directory.  Our extension then handles this request,
# forwarding the data from the proxied site.
# For example:
# * URL of "index.html" comes in.
# * Filter rewrites this to "/python/index.html"
# * Our extension sees the full "/python/index.html", removes the leading
#   portion, and opens and forwards the remote URL.


# This sample is very small - it avoid most error handling, etc.  It is for
# demonstration purposes only.

import sys
import urllib.error
import urllib.parse
import urllib.request

from isapi import isapicon, threaded_extension
from isapi.simple import SimpleFilter

# sys.isapidllhandle will exist when we are loaded by the IIS framework.
# In this case we redirect our output to the win32traceutil collector.
if hasattr(sys, "isapidllhandle"):
    import win32traceutil

# The site we are proxying.
proxy = "http://www.python.org"
# The name of the virtual directory we install in, and redirect from.
virtualdir = "/python"

# The key feature of this redirector over the simple redirector is that it
# can choose to ignore certain responses by having the filter not rewrite them
# to our virtual dir. For this sample, we just exclude the IIS help directory.


# The ISAPI extension - handles requests in our virtual dir, and sends the
# response to the client.
class Extension(threaded_extension.ThreadPoolExtension):
    "Python sample Extension"

    def Dispatch(self, ecb):
        # Note that our ThreadPoolExtension base class will catch exceptions
        # in our Dispatch method, and write the traceback to the client.
        # That is perfect for this sample, so we don't catch our own.
        # print 'IIS dispatching "%s"' % (ecb.GetServerVariable("URL"),)
        url = ecb.GetServerVariable("URL")
        if url.startswith(virtualdir):
            new_url = proxy + url[len(virtualdir) :]
            print("Opening", new_url)
            fp = urllib.request.urlopen(new_url)
            headers = fp.info()
            ecb.SendResponseHeaders("200 OK", str(headers) + "\r\n", False)
            ecb.WriteClient(fp.read())
            ecb.DoneWithSession()
            print("Returned data from '%s'!" % (new_url,))
        else:
            # this should never happen - we should only see requests that
            # start with our virtual directory name.
            print("Not proxying '%s'" % (url,))


# The ISAPI filter.
class Filter(SimpleFilter):
    "Sample Python Redirector"
    filter_flags = isapicon.SF_NOTIFY_PREPROC_HEADERS | isapicon.SF_NOTIFY_ORDER_DEFAULT

    def HttpFilterProc(self, fc):
        # print "Filter Dispatch"
        nt = fc.NotificationType
        if nt != isapicon.SF_NOTIFY_PREPROC_HEADERS:
            return isapicon.SF_STATUS_REQ_NEXT_NOTIFICATION

        pp = fc.GetData()
        url = pp.GetHeader("url")
        # print "URL is '%s'" % (url,)
        prefix = virtualdir
        if not url.startswith(prefix):
            new_url = prefix + url
            print("New proxied URL is '%s'" % (new_url,))
            pp.SetHeader("url", new_url)
            # For the sake of demonstration, show how the FilterContext
            # attribute is used.  It always starts out life as None, and
            # any assignments made are automatically decref'd by the
            # framework during a SF_NOTIFY_END_OF_NET_SESSION notification.
            if fc.FilterContext is None:
                fc.FilterContext = 0
            fc.FilterContext += 1
            print("This is request number %d on this connection" % fc.FilterContext)
            return isapicon.SF_STATUS_REQ_HANDLED_NOTIFICATION
        else:
            print("Filter ignoring URL '%s'" % (url,))

            # Some older code that handled SF_NOTIFY_URL_MAP.
            # ~ print "Have URL_MAP notify"
            # ~ urlmap = fc.GetData()
            # ~ print "URI is", urlmap.URL
            # ~ print "Path is", urlmap.PhysicalPath
            # ~ if urlmap.URL.startswith("/UC/"):
            # ~ # Find the /UC/ in the physical path, and nuke it (except
            # ~ # as the path is physical, it is \)
            # ~ p = urlmap.PhysicalPath
            # ~ pos = p.index("\\UC\\")
            # ~ p = p[:pos] + p[pos+3:]
            # ~ p = r"E:\src\pyisapi\webroot\PyTest\formTest.htm"
            # ~ print "New path is", p
            # ~ urlmap.PhysicalPath = p


# The entry points for the ISAPI extension.
def __FilterFactory__():
    return Filter()


def __ExtensionFactory__():
    return Extension()


if __name__ == "__main__":
    # If run from the command-line, install ourselves.
    from isapi.install import *

    params = ISAPIParameters()
    # Setup all filters - these are global to the site.
    params.Filters = [
        FilterParameters(Name="PythonRedirector", Description=Filter.__doc__),
    ]
    # Setup the virtual directories - this is a list of directories our
    # extension uses - in this case only 1.
    # Each extension has a "script map" - this is the mapping of ISAPI
    # extensions.
    sm = [ScriptMapParams(Extension="*", Flags=0)]
    vd = VirtualDirParameters(
        Name=virtualdir[1:],
        Description=Extension.__doc__,
        ScriptMaps=sm,
        ScriptMapUpdate="replace",
    )
    params.VirtualDirs = [vd]
    HandleCommandLine(params)
