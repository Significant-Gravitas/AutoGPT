"""
Fetches a URL from a web-server supporting NTLM authentication
eg, IIS.

If no arguments are specified, a default of http://localhost/localstart.asp
is used.  This script does follow simple 302 redirections, so pointing at the
root of an IIS server is should work.
"""

import http.client  # sorry, this demo needs 2.3+
import optparse
import urllib.error
import urllib.parse
import urllib.request
from base64 import decodestring, encodestring

from sspi import ClientAuth

options = None  # set to optparse options object


def open_url(host, url):
    h = http.client.HTTPConnection(host)
    #    h.set_debuglevel(9)
    h.putrequest("GET", url)
    h.endheaders()
    resp = h.getresponse()
    print("Initial response is", resp.status, resp.reason)
    body = resp.read()
    if resp.status == 302:  # object moved
        url = "/" + resp.msg["location"]
        resp.close()
        h.putrequest("GET", url)
        h.endheaders()
        resp = h.getresponse()
        print("After redirect response is", resp.status, resp.reason)
    if options.show_headers:
        print("Initial response headers:")
        for name, val in list(resp.msg.items()):
            print(" %s: %s" % (name, val))
    if options.show_body:
        print(body)
    if resp.status == 401:
        # 401: Unauthorized - here is where the real work starts
        auth_info = None
        if options.user or options.domain or options.password:
            auth_info = options.user, options.domain, options.password
        ca = ClientAuth("NTLM", auth_info=auth_info)
        auth_scheme = ca.pkg_info["Name"]
        data = None
        while 1:
            err, out_buf = ca.authorize(data)
            data = out_buf[0].Buffer
            # Encode it as base64 as required by HTTP
            auth = encodestring(data).replace("\012", "")
            h.putrequest("GET", url)
            h.putheader("Authorization", auth_scheme + " " + auth)
            h.putheader("Content-Length", "0")
            h.endheaders()
            resp = h.getresponse()
            if options.show_headers:
                print("Token dance headers:")
                for name, val in list(resp.msg.items()):
                    print(" %s: %s" % (name, val))

            if err == 0:
                break
            else:
                if resp.status != 401:
                    print("Eeek - got response", resp.status)
                    cl = resp.msg.get("content-length")
                    if cl:
                        print(repr(resp.read(int(cl))))
                    else:
                        print("no content!")

                assert resp.status == 401, resp.status

            assert not resp.will_close, "NTLM is per-connection - must not close"
            schemes = [
                s.strip() for s in resp.msg.get("WWW-Authenticate", "").split(",")
            ]
            for scheme in schemes:
                if scheme.startswith(auth_scheme):
                    data = decodestring(scheme[len(auth_scheme) + 1 :])
                    break
            else:
                print(
                    "Could not find scheme '%s' in schemes %r" % (auth_scheme, schemes)
                )
                break

            resp.read()
    print("Final response status is", resp.status, resp.reason)
    if resp.status == 200:
        # Worked!
        # Check we can read it again without re-authenticating.
        if resp.will_close:
            print(
                "EEEK - response will close, but NTLM is per connection - it must stay open"
            )
        body = resp.read()
        if options.show_body:
            print("Final response body:")
            print(body)
        h.putrequest("GET", url)
        h.endheaders()
        resp = h.getresponse()
        print("Second fetch response is", resp.status, resp.reason)
        if options.show_headers:
            print("Second response headers:")
            for name, val in list(resp.msg.items()):
                print(" %s: %s" % (name, val))

        resp.read(int(resp.msg.get("content-length", 0)))
    elif resp.status == 500:
        print("Error text")
        print(resp.read())
    else:
        if options.show_body:
            cl = resp.msg.get("content-length")
            print(resp.read(int(cl)))


if __name__ == "__main__":
    parser = optparse.OptionParser(description=__doc__)

    parser.add_option(
        "",
        "--show-body",
        action="store_true",
        help="print the body of each response as it is received",
    )

    parser.add_option(
        "",
        "--show-headers",
        action="store_true",
        help="print the headers of each response as it is received",
    )

    parser.add_option("", "--user", action="store", help="The username to login with")

    parser.add_option(
        "", "--password", action="store", help="The password to login with"
    )

    parser.add_option("", "--domain", action="store", help="The domain to login to")

    options, args = parser.parse_args()
    if not args:
        print("Run with --help for usage details")
        args = ["http://localhost/localstart.asp"]
    for url in args:
        scheme, netloc, path, params, query, fragment = urllib.parse.urlparse(url)
        if (scheme != "http") or params or query or fragment:
            parser.error("Scheme must be http, URL must be simple")

        print("Opening '%s' from '%s'" % (path, netloc))
        r = open_url(netloc, path)
