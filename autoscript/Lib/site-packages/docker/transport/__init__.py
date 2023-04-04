# flake8: noqa
from .unixconn import UnixHTTPAdapter
from .ssladapter import SSLHTTPAdapter
try:
    from .npipeconn import NpipeHTTPAdapter
    from .npipesocket import NpipeSocket
except ImportError:
    pass

try:
    from .sshconn import SSHHTTPAdapter
except ImportError:
    pass
