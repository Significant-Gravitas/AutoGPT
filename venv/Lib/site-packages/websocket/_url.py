import os
import socket
import struct

from urllib.parse import unquote, urlparse

"""
_url.py
websocket - WebSocket client library for Python

Copyright 2022 engn33r

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__all__ = ["parse_url", "get_proxy_info"]


def parse_url(url):
    """
    parse url and the result is tuple of
    (hostname, port, resource path and the flag of secure mode)

    Parameters
    ----------
    url: str
        url string.
    """
    if ":" not in url:
        raise ValueError("url is invalid")

    scheme, url = url.split(":", 1)

    parsed = urlparse(url, scheme="http")
    if parsed.hostname:
        hostname = parsed.hostname
    else:
        raise ValueError("hostname is invalid")
    port = 0
    if parsed.port:
        port = parsed.port

    is_secure = False
    if scheme == "ws":
        if not port:
            port = 80
    elif scheme == "wss":
        is_secure = True
        if not port:
            port = 443
    else:
        raise ValueError("scheme %s is invalid" % scheme)

    if parsed.path:
        resource = parsed.path
    else:
        resource = "/"

    if parsed.query:
        resource += "?" + parsed.query

    return hostname, port, resource, is_secure


DEFAULT_NO_PROXY_HOST = ["localhost", "127.0.0.1"]


def _is_ip_address(addr):
    try:
        socket.inet_aton(addr)
    except socket.error:
        return False
    else:
        return True


def _is_subnet_address(hostname):
    try:
        addr, netmask = hostname.split("/")
        return _is_ip_address(addr) and 0 <= int(netmask) < 32
    except ValueError:
        return False


def _is_address_in_network(ip, net):
    ipaddr = struct.unpack('!I', socket.inet_aton(ip))[0]
    netaddr, netmask = net.split('/')
    netaddr = struct.unpack('!I', socket.inet_aton(netaddr))[0]

    netmask = (0xFFFFFFFF << (32 - int(netmask))) & 0xFFFFFFFF
    return ipaddr & netmask == netaddr


def _is_no_proxy_host(hostname, no_proxy):
    if not no_proxy:
        v = os.environ.get("no_proxy", os.environ.get("NO_PROXY", "")).replace(" ", "")
        if v:
            no_proxy = v.split(",")
    if not no_proxy:
        no_proxy = DEFAULT_NO_PROXY_HOST

    if '*' in no_proxy:
        return True
    if hostname in no_proxy:
        return True
    if _is_ip_address(hostname):
        return any([_is_address_in_network(hostname, subnet) for subnet in no_proxy if _is_subnet_address(subnet)])
    for domain in [domain for domain in no_proxy if domain.startswith('.')]:
        if hostname.endswith(domain):
            return True
    return False


def get_proxy_info(
        hostname, is_secure, proxy_host=None, proxy_port=0, proxy_auth=None,
        no_proxy=None, proxy_type='http'):
    """
    Try to retrieve proxy host and port from environment
    if not provided in options.
    Result is (proxy_host, proxy_port, proxy_auth).
    proxy_auth is tuple of username and password
    of proxy authentication information.

    Parameters
    ----------
    hostname: str
        Websocket server name.
    is_secure: bool
        Is the connection secure? (wss) looks for "https_proxy" in env
        before falling back to "http_proxy"
    proxy_host: str
        http proxy host name.
    http_proxy_port: str or int
        http proxy port.
    http_no_proxy: list
        Whitelisted host names that don't use the proxy.
    http_proxy_auth: tuple
        HTTP proxy auth information. Tuple of username and password. Default is None.
    proxy_type: str
        Specify the proxy protocol (http, socks4, socks4a, socks5, socks5h). Default is "http".
        Use socks4a or socks5h if you want to send DNS requests through the proxy.
    """
    if _is_no_proxy_host(hostname, no_proxy):
        return None, 0, None

    if proxy_host:
        port = proxy_port
        auth = proxy_auth
        return proxy_host, port, auth

    env_keys = ["http_proxy"]
    if is_secure:
        env_keys.insert(0, "https_proxy")

    for key in env_keys:
        value = os.environ.get(key, os.environ.get(key.upper(), "")).replace(" ", "")
        if value:
            proxy = urlparse(value)
            auth = (unquote(proxy.username), unquote(proxy.password)) if proxy.username else None
            return proxy.hostname, proxy.port, auth

    return None, 0, None
