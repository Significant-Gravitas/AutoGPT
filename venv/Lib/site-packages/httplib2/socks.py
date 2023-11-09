"""SocksiPy - Python SOCKS module.

Version 1.00

Copyright 2006 Dan-Haim. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of Dan Haim nor the names of his contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY DAN HAIM "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL DAN HAIM OR HIS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMANGE.

This module provides a standard socket-like interface for Python
for tunneling connections through SOCKS proxies.

Minor modifications made by Christopher Gilbert (http://motomastyle.com/) for
use in PyLoris (http://pyloris.sourceforge.net/).

Minor modifications made by Mario Vilas (http://breakingcode.wordpress.com/)
mainly to merge bug fixes found in Sourceforge.
"""

import base64
import socket
import struct
import sys

if getattr(socket, "socket", None) is None:
    raise ImportError("socket.socket missing, proxy support unusable")

PROXY_TYPE_SOCKS4 = 1
PROXY_TYPE_SOCKS5 = 2
PROXY_TYPE_HTTP = 3
PROXY_TYPE_HTTP_NO_TUNNEL = 4

_defaultproxy = None
_orgsocket = socket.socket


class ProxyError(Exception):
    pass


class GeneralProxyError(ProxyError):
    pass


class Socks5AuthError(ProxyError):
    pass


class Socks5Error(ProxyError):
    pass


class Socks4Error(ProxyError):
    pass


class HTTPError(ProxyError):
    pass


_generalerrors = (
    "success",
    "invalid data",
    "not connected",
    "not available",
    "bad proxy type",
    "bad input",
)

_socks5errors = (
    "succeeded",
    "general SOCKS server failure",
    "connection not allowed by ruleset",
    "Network unreachable",
    "Host unreachable",
    "Connection refused",
    "TTL expired",
    "Command not supported",
    "Address type not supported",
    "Unknown error",
)

_socks5autherrors = (
    "succeeded",
    "authentication is required",
    "all offered authentication methods were rejected",
    "unknown username or invalid password",
    "unknown error",
)

_socks4errors = (
    "request granted",
    "request rejected or failed",
    "request rejected because SOCKS server cannot connect to identd on the client",
    "request rejected because the client program and identd report different "
    "user-ids",
    "unknown error",
)


def setdefaultproxy(
    proxytype=None, addr=None, port=None, rdns=True, username=None, password=None
):
    """setdefaultproxy(proxytype, addr[, port[, rdns[, username[, password]]]])
    Sets a default proxy which all further socksocket objects will use,
    unless explicitly changed.
    """
    global _defaultproxy
    _defaultproxy = (proxytype, addr, port, rdns, username, password)


def wrapmodule(module):
    """wrapmodule(module)

    Attempts to replace a module's socket library with a SOCKS socket. Must set
    a default proxy using setdefaultproxy(...) first.
    This will only work on modules that import socket directly into the
    namespace;
    most of the Python Standard Library falls into this category.
    """
    if _defaultproxy != None:
        module.socket.socket = socksocket
    else:
        raise GeneralProxyError((4, "no proxy specified"))


class socksocket(socket.socket):
    """socksocket([family[, type[, proto]]]) -> socket object
    Open a SOCKS enabled socket. The parameters are the same as
    those of the standard socket init. In order for SOCKS to work,
    you must specify family=AF_INET, type=SOCK_STREAM and proto=0.
    """

    def __init__(
        self, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, _sock=None
    ):
        _orgsocket.__init__(self, family, type, proto, _sock)
        if _defaultproxy != None:
            self.__proxy = _defaultproxy
        else:
            self.__proxy = (None, None, None, None, None, None)
        self.__proxysockname = None
        self.__proxypeername = None
        self.__httptunnel = True

    def __recvall(self, count):
        """__recvall(count) -> data
        Receive EXACTLY the number of bytes requested from the socket.
        Blocks until the required number of bytes have been received.
        """
        data = self.recv(count)
        while len(data) < count:
            d = self.recv(count - len(data))
            if not d:
                raise GeneralProxyError((0, "connection closed unexpectedly"))
            data = data + d
        return data

    def sendall(self, content, *args):
        """ override socket.socket.sendall method to rewrite the header
        for non-tunneling proxies if needed
        """
        if not self.__httptunnel:
            content = self.__rewriteproxy(content)
        return super(socksocket, self).sendall(content, *args)

    def __rewriteproxy(self, header):
        """ rewrite HTTP request headers to support non-tunneling proxies
        (i.e. those which do not support the CONNECT method).
        This only works for HTTP (not HTTPS) since HTTPS requires tunneling.
        """
        host, endpt = None, None
        hdrs = header.split("\r\n")
        for hdr in hdrs:
            if hdr.lower().startswith("host:"):
                host = hdr
            elif hdr.lower().startswith("get") or hdr.lower().startswith("post"):
                endpt = hdr
        if host and endpt:
            hdrs.remove(host)
            hdrs.remove(endpt)
            host = host.split(" ")[1]
            endpt = endpt.split(" ")
            if self.__proxy[4] != None and self.__proxy[5] != None:
                hdrs.insert(0, self.__getauthheader())
            hdrs.insert(0, "Host: %s" % host)
            hdrs.insert(0, "%s http://%s%s %s" % (endpt[0], host, endpt[1], endpt[2]))
        return "\r\n".join(hdrs)

    def __getauthheader(self):
        auth = self.__proxy[4] + b":" + self.__proxy[5]
        return "Proxy-Authorization: Basic " + base64.b64encode(auth).decode()

    def setproxy(
        self,
        proxytype=None,
        addr=None,
        port=None,
        rdns=True,
        username=None,
        password=None,
        headers=None,
    ):
        """setproxy(proxytype, addr[, port[, rdns[, username[, password]]]])

        Sets the proxy to be used.
        proxytype -    The type of the proxy to be used. Three types
                are supported: PROXY_TYPE_SOCKS4 (including socks4a),
                PROXY_TYPE_SOCKS5 and PROXY_TYPE_HTTP
        addr -        The address of the server (IP or DNS).
        port -        The port of the server. Defaults to 1080 for SOCKS
                servers and 8080 for HTTP proxy servers.
        rdns -        Should DNS queries be preformed on the remote side
                (rather than the local side). The default is True.
                Note: This has no effect with SOCKS4 servers.
        username -    Username to authenticate with to the server.
                The default is no authentication.
        password -    Password to authenticate with to the server.
                Only relevant when username is also provided.
        headers -     Additional or modified headers for the proxy connect
        request.
        """
        self.__proxy = (
            proxytype,
            addr,
            port,
            rdns,
            username.encode() if username else None,
            password.encode() if password else None,
            headers,
        )

    def __negotiatesocks5(self, destaddr, destport):
        """__negotiatesocks5(self,destaddr,destport)
        Negotiates a connection through a SOCKS5 server.
        """
        # First we'll send the authentication packages we support.
        if (self.__proxy[4] != None) and (self.__proxy[5] != None):
            # The username/password details were supplied to the
            # setproxy method so we support the USERNAME/PASSWORD
            # authentication (in addition to the standard none).
            self.sendall(struct.pack("BBBB", 0x05, 0x02, 0x00, 0x02))
        else:
            # No username/password were entered, therefore we
            # only support connections with no authentication.
            self.sendall(struct.pack("BBB", 0x05, 0x01, 0x00))
        # We'll receive the server's response to determine which
        # method was selected
        chosenauth = self.__recvall(2)
        if chosenauth[0:1] != chr(0x05).encode():
            self.close()
            raise GeneralProxyError((1, _generalerrors[1]))
        # Check the chosen authentication method
        if chosenauth[1:2] == chr(0x00).encode():
            # No authentication is required
            pass
        elif chosenauth[1:2] == chr(0x02).encode():
            # Okay, we need to perform a basic username/password
            # authentication.
            packet = bytearray()
            packet.append(0x01)
            packet.append(len(self.__proxy[4]))
            packet.extend(self.__proxy[4])
            packet.append(len(self.__proxy[5]))
            packet.extend(self.__proxy[5])
            self.sendall(packet)
            authstat = self.__recvall(2)
            if authstat[0:1] != chr(0x01).encode():
                # Bad response
                self.close()
                raise GeneralProxyError((1, _generalerrors[1]))
            if authstat[1:2] != chr(0x00).encode():
                # Authentication failed
                self.close()
                raise Socks5AuthError((3, _socks5autherrors[3]))
            # Authentication succeeded
        else:
            # Reaching here is always bad
            self.close()
            if chosenauth[1] == chr(0xFF).encode():
                raise Socks5AuthError((2, _socks5autherrors[2]))
            else:
                raise GeneralProxyError((1, _generalerrors[1]))
        # Now we can request the actual connection
        req = struct.pack("BBB", 0x05, 0x01, 0x00)
        # If the given destination address is an IP address, we'll
        # use the IPv4 address request even if remote resolving was specified.
        try:
            ipaddr = socket.inet_aton(destaddr)
            req = req + chr(0x01).encode() + ipaddr
        except socket.error:
            # Well it's not an IP number,  so it's probably a DNS name.
            if self.__proxy[3]:
                # Resolve remotely
                ipaddr = None
                req = (
                    req
                    + chr(0x03).encode()
                    + chr(len(destaddr)).encode()
                    + destaddr.encode()
                )
            else:
                # Resolve locally
                ipaddr = socket.inet_aton(socket.gethostbyname(destaddr))
                req = req + chr(0x01).encode() + ipaddr
        req = req + struct.pack(">H", destport)
        self.sendall(req)
        # Get the response
        resp = self.__recvall(4)
        if resp[0:1] != chr(0x05).encode():
            self.close()
            raise GeneralProxyError((1, _generalerrors[1]))
        elif resp[1:2] != chr(0x00).encode():
            # Connection failed
            self.close()
            if ord(resp[1:2]) <= 8:
                raise Socks5Error((ord(resp[1:2]), _socks5errors[ord(resp[1:2])]))
            else:
                raise Socks5Error((9, _socks5errors[9]))
        # Get the bound address/port
        elif resp[3:4] == chr(0x01).encode():
            boundaddr = self.__recvall(4)
        elif resp[3:4] == chr(0x03).encode():
            resp = resp + self.recv(1)
            boundaddr = self.__recvall(ord(resp[4:5]))
        else:
            self.close()
            raise GeneralProxyError((1, _generalerrors[1]))
        boundport = struct.unpack(">H", self.__recvall(2))[0]
        self.__proxysockname = (boundaddr, boundport)
        if ipaddr != None:
            self.__proxypeername = (socket.inet_ntoa(ipaddr), destport)
        else:
            self.__proxypeername = (destaddr, destport)

    def getproxysockname(self):
        """getsockname() -> address info
        Returns the bound IP address and port number at the proxy.
        """
        return self.__proxysockname

    def getproxypeername(self):
        """getproxypeername() -> address info
        Returns the IP and port number of the proxy.
        """
        return _orgsocket.getpeername(self)

    def getpeername(self):
        """getpeername() -> address info
        Returns the IP address and port number of the destination
        machine (note: getproxypeername returns the proxy)
        """
        return self.__proxypeername

    def __negotiatesocks4(self, destaddr, destport):
        """__negotiatesocks4(self,destaddr,destport)
        Negotiates a connection through a SOCKS4 server.
        """
        # Check if the destination address provided is an IP address
        rmtrslv = False
        try:
            ipaddr = socket.inet_aton(destaddr)
        except socket.error:
            # It's a DNS name. Check where it should be resolved.
            if self.__proxy[3]:
                ipaddr = struct.pack("BBBB", 0x00, 0x00, 0x00, 0x01)
                rmtrslv = True
            else:
                ipaddr = socket.inet_aton(socket.gethostbyname(destaddr))
        # Construct the request packet
        req = struct.pack(">BBH", 0x04, 0x01, destport) + ipaddr
        # The username parameter is considered userid for SOCKS4
        if self.__proxy[4] != None:
            req = req + self.__proxy[4]
        req = req + chr(0x00).encode()
        # DNS name if remote resolving is required
        # NOTE: This is actually an extension to the SOCKS4 protocol
        # called SOCKS4A and may not be supported in all cases.
        if rmtrslv:
            req = req + destaddr + chr(0x00).encode()
        self.sendall(req)
        # Get the response from the server
        resp = self.__recvall(8)
        if resp[0:1] != chr(0x00).encode():
            # Bad data
            self.close()
            raise GeneralProxyError((1, _generalerrors[1]))
        if resp[1:2] != chr(0x5A).encode():
            # Server returned an error
            self.close()
            if ord(resp[1:2]) in (91, 92, 93):
                self.close()
                raise Socks4Error((ord(resp[1:2]), _socks4errors[ord(resp[1:2]) - 90]))
            else:
                raise Socks4Error((94, _socks4errors[4]))
        # Get the bound address/port
        self.__proxysockname = (
            socket.inet_ntoa(resp[4:]),
            struct.unpack(">H", resp[2:4])[0],
        )
        if rmtrslv != None:
            self.__proxypeername = (socket.inet_ntoa(ipaddr), destport)
        else:
            self.__proxypeername = (destaddr, destport)

    def __negotiatehttp(self, destaddr, destport):
        """__negotiatehttp(self,destaddr,destport)
        Negotiates a connection through an HTTP server.
        """
        # If we need to resolve locally, we do this now
        if not self.__proxy[3]:
            addr = socket.gethostbyname(destaddr)
        else:
            addr = destaddr
        headers = ["CONNECT ", addr, ":", str(destport), " HTTP/1.1\r\n"]
        wrote_host_header = False
        wrote_auth_header = False
        if self.__proxy[6] != None:
            for key, val in self.__proxy[6].iteritems():
                headers += [key, ": ", val, "\r\n"]
                wrote_host_header = key.lower() == "host"
                wrote_auth_header = key.lower() == "proxy-authorization"
        if not wrote_host_header:
            headers += ["Host: ", destaddr, "\r\n"]
        if not wrote_auth_header:
            if self.__proxy[4] != None and self.__proxy[5] != None:
                headers += [self.__getauthheader(), "\r\n"]
        headers.append("\r\n")
        self.sendall("".join(headers).encode())
        # We read the response until we get the string "\r\n\r\n"
        resp = self.recv(1)
        while resp.find("\r\n\r\n".encode()) == -1:
            resp = resp + self.recv(1)
        # We just need the first line to check if the connection
        # was successful
        statusline = resp.splitlines()[0].split(" ".encode(), 2)
        if statusline[0] not in ("HTTP/1.0".encode(), "HTTP/1.1".encode()):
            self.close()
            raise GeneralProxyError((1, _generalerrors[1]))
        try:
            statuscode = int(statusline[1])
        except ValueError:
            self.close()
            raise GeneralProxyError((1, _generalerrors[1]))
        if statuscode != 200:
            self.close()
            raise HTTPError((statuscode, statusline[2]))
        self.__proxysockname = ("0.0.0.0", 0)
        self.__proxypeername = (addr, destport)

    def connect(self, destpair):
        """connect(self, despair)
        Connects to the specified destination through a proxy.
        destpar - A tuple of the IP/DNS address and the port number.
        (identical to socket's connect).
        To select the proxy server use setproxy().
        """
        # Do a minimal input check first
        if (
            (not type(destpair) in (list, tuple))
            or (len(destpair) < 2)
            or (not isinstance(destpair[0], (str, bytes)))
            or (type(destpair[1]) != int)
        ):
            raise GeneralProxyError((5, _generalerrors[5]))
        if self.__proxy[0] == PROXY_TYPE_SOCKS5:
            if self.__proxy[2] != None:
                portnum = self.__proxy[2]
            else:
                portnum = 1080
            _orgsocket.connect(self, (self.__proxy[1], portnum))
            self.__negotiatesocks5(destpair[0], destpair[1])
        elif self.__proxy[0] == PROXY_TYPE_SOCKS4:
            if self.__proxy[2] != None:
                portnum = self.__proxy[2]
            else:
                portnum = 1080
            _orgsocket.connect(self, (self.__proxy[1], portnum))
            self.__negotiatesocks4(destpair[0], destpair[1])
        elif self.__proxy[0] == PROXY_TYPE_HTTP:
            if self.__proxy[2] != None:
                portnum = self.__proxy[2]
            else:
                portnum = 8080
            _orgsocket.connect(self, (self.__proxy[1], portnum))
            self.__negotiatehttp(destpair[0], destpair[1])
        elif self.__proxy[0] == PROXY_TYPE_HTTP_NO_TUNNEL:
            if self.__proxy[2] != None:
                portnum = self.__proxy[2]
            else:
                portnum = 8080
            _orgsocket.connect(self, (self.__proxy[1], portnum))
            if destpair[1] == 443:
                self.__negotiatehttp(destpair[0], destpair[1])
            else:
                self.__httptunnel = False
        elif self.__proxy[0] == None:
            _orgsocket.connect(self, (destpair[0], destpair[1]))
        else:
            raise GeneralProxyError((4, _generalerrors[4]))
