"""A sample socket server and client using SSPI authentication and encryption.

You must run with either 'client' or 'server' as arguments.  A server must be
running before a client can connect.

To use with Kerberos you should include in the client options
--target-spn=username, where 'username' is the user under which the server is
being run.

Running either the client or server as a different user can be informative.
A command-line such as the following may be useful:
`runas /user:{user} {fqp}\python.exe {fqp}\socket_server.py --wait client|server`

{fqp} should specify the relevant fully-qualified path names.

To use 'runas' with Kerberos, the client program will need to
specify --target-spn with the username under which the *server* is running.

See the SSPI documentation for more details.
"""


import http.client  # sorry, this demo needs 2.3+
import optparse
import socketserver
import struct
import traceback

import sspi
import win32api
import win32security

options = None  # set to optparse object.


def GetUserName():
    try:
        return win32api.GetUserName()
    except win32api.error as details:
        # Seeing 'access denied' errors here for non-local users (presumably
        # without permission to login locally).  Get the fully-qualified
        # username, although a side-effect of these permission-denied errors
        # is a lack of Python codecs - so printing the Unicode value fails.
        # So just return the repr(), and avoid codecs completely.
        return repr(win32api.GetUserNameEx(win32api.NameSamCompatible))


# Send a simple "message" over a socket - send the number of bytes first,
# then the string.  Ditto for receive.
def _send_msg(s, m):
    s.send(struct.pack("i", len(m)))
    s.send(m)


def _get_msg(s):
    size_data = s.recv(struct.calcsize("i"))
    if not size_data:
        return None
    cb = struct.unpack("i", size_data)[0]
    return s.recv(cb)


class SSPISocketServer(socketserver.TCPServer):
    def __init__(self, *args, **kw):
        socketserver.TCPServer.__init__(self, *args, **kw)
        self.sa = sspi.ServerAuth(options.package)

    def verify_request(self, sock, ca):
        # Do the sspi auth dance
        self.sa.reset()
        while 1:
            data = _get_msg(sock)
            if data is None:
                return False
            try:
                err, sec_buffer = self.sa.authorize(data)
            except sspi.error as details:
                print("FAILED to authorize client:", details)
                return False

            if err == 0:
                break
            _send_msg(sock, sec_buffer[0].Buffer)
        return True

    def process_request(self, request, client_address):
        # An example using the connection once it is established.
        print("The server is running as user", GetUserName())
        self.sa.ctxt.ImpersonateSecurityContext()
        try:
            print("Having conversation with client as user", GetUserName())
            while 1:
                # we need to grab 2 bits of data - the encrypted data, and the
                # 'key'
                data = _get_msg(request)
                key = _get_msg(request)
                if data is None or key is None:
                    break
                data = self.sa.decrypt(data, key)
                print("Client sent:", repr(data))
        finally:
            self.sa.ctxt.RevertSecurityContext()
        self.close_request(request)
        print("The server is back to user", GetUserName())


def serve():
    s = SSPISocketServer(("localhost", options.port), None)
    print("Running test server...")
    s.serve_forever()


def sspi_client():
    c = http.client.HTTPConnection("localhost", options.port)
    c.connect()
    # Do the auth dance.
    ca = sspi.ClientAuth(options.package, targetspn=options.target_spn)
    data = None
    while 1:
        err, out_buf = ca.authorize(data)
        _send_msg(c.sock, out_buf[0].Buffer)
        if err == 0:
            break
        data = _get_msg(c.sock)
    print("Auth dance complete - sending a few encryted messages")
    # Assume out data is sensitive - encrypt the message.
    for data in "Hello from the client".split():
        blob, key = ca.encrypt(data)
        _send_msg(c.sock, blob)
        _send_msg(c.sock, key)
    c.sock.close()
    print("Client completed.")


if __name__ == "__main__":
    parser = optparse.OptionParser("%prog [options] client|server", description=__doc__)

    parser.add_option(
        "",
        "--package",
        action="store",
        default="NTLM",
        help="The SSPI package to use (eg, Kerberos) - default is NTLM",
    )

    parser.add_option(
        "",
        "--target-spn",
        action="store",
        help="""The target security provider name to use. The
                      string contents are security-package specific.  For
                      example, 'Kerberos' or 'Negotiate' require the server
                      principal name (SPN) (ie, the username) of the remote
                      process.  For NTLM this must be blank.""",
    )

    parser.add_option(
        "",
        "--port",
        action="store",
        default="8181",
        help="The port number to use (default=8181)",
    )

    parser.add_option(
        "",
        "--wait",
        action="store_true",
        help="""Cause the program to wait for input just before
                              terminating. Useful when using via runas to see
                              any error messages before termination.
                           """,
    )

    options, args = parser.parse_args()
    try:
        options.port = int(options.port)
    except (ValueError, TypeError):
        parser.error("--port must be an integer")

    try:
        try:
            if not args:
                args = [""]
            if args[0] == "client":
                sspi_client()
            elif args[0] == "server":
                serve()
            else:
                parser.error(
                    "You must supply 'client' or 'server' - " "use --help for details"
                )
        except KeyboardInterrupt:
            pass
        except SystemExit:
            pass
        except:
            traceback.print_exc()
    finally:
        if options.wait:
            input("Press enter to continue")
