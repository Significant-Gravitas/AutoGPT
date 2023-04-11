import requests.adapters
import socket
import http.client as httplib

from docker.transport.basehttpadapter import BaseHTTPAdapter
from .. import constants

try:
    import requests.packages.urllib3 as urllib3
except ImportError:
    import urllib3


RecentlyUsedContainer = urllib3._collections.RecentlyUsedContainer


class UnixHTTPConnection(httplib.HTTPConnection):

    def __init__(self, base_url, unix_socket, timeout=60):
        super().__init__(
            'localhost', timeout=timeout
        )
        self.base_url = base_url
        self.unix_socket = unix_socket
        self.timeout = timeout

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.unix_socket)
        self.sock = sock

    def putheader(self, header, *values):
        super().putheader(header, *values)

    def response_class(self, sock, *args, **kwargs):
        return httplib.HTTPResponse(sock, *args, **kwargs)


class UnixHTTPConnectionPool(urllib3.connectionpool.HTTPConnectionPool):
    def __init__(self, base_url, socket_path, timeout=60, maxsize=10):
        super().__init__(
            'localhost', timeout=timeout, maxsize=maxsize
        )
        self.base_url = base_url
        self.socket_path = socket_path
        self.timeout = timeout

    def _new_conn(self):
        return UnixHTTPConnection(
            self.base_url, self.socket_path, self.timeout
        )


class UnixHTTPAdapter(BaseHTTPAdapter):

    __attrs__ = requests.adapters.HTTPAdapter.__attrs__ + ['pools',
                                                           'socket_path',
                                                           'timeout',
                                                           'max_pool_size']

    def __init__(self, socket_url, timeout=60,
                 pool_connections=constants.DEFAULT_NUM_POOLS,
                 max_pool_size=constants.DEFAULT_MAX_POOL_SIZE):
        socket_path = socket_url.replace('http+unix://', '')
        if not socket_path.startswith('/'):
            socket_path = '/' + socket_path
        self.socket_path = socket_path
        self.timeout = timeout
        self.max_pool_size = max_pool_size
        self.pools = RecentlyUsedContainer(
            pool_connections, dispose_func=lambda p: p.close()
        )
        super().__init__()

    def get_connection(self, url, proxies=None):
        with self.pools.lock:
            pool = self.pools.get(url)
            if pool:
                return pool

            pool = UnixHTTPConnectionPool(
                url, self.socket_path, self.timeout,
                maxsize=self.max_pool_size
            )
            self.pools[url] = pool

        return pool

    def request_url(self, request, proxies):
        # The select_proxy utility in requests errors out when the provided URL
        # doesn't have a hostname, like is the case when using a UNIX socket.
        # Since proxies are an irrelevant notion in the case of UNIX sockets
        # anyway, we simply return the path URL directly.
        # See also: https://github.com/docker/docker-py/issues/811
        return request.path_url
