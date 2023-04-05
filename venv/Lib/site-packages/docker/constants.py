import sys
from .version import __version__

DEFAULT_DOCKER_API_VERSION = '1.41'
MINIMUM_DOCKER_API_VERSION = '1.21'
DEFAULT_TIMEOUT_SECONDS = 60
STREAM_HEADER_SIZE_BYTES = 8
CONTAINER_LIMITS_KEYS = [
    'memory', 'memswap', 'cpushares', 'cpusetcpus'
]

DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_UNIX_SOCKET = "http+unix:///var/run/docker.sock"
DEFAULT_NPIPE = 'npipe:////./pipe/docker_engine'

BYTE_UNITS = {
    'b': 1,
    'k': 1024,
    'm': 1024 * 1024,
    'g': 1024 * 1024 * 1024
}


INSECURE_REGISTRY_DEPRECATION_WARNING = \
    'The `insecure_registry` argument to {} ' \
    'is deprecated and non-functional. Please remove it.'

IS_WINDOWS_PLATFORM = (sys.platform == 'win32')
WINDOWS_LONGPATH_PREFIX = '\\\\?\\'

DEFAULT_USER_AGENT = f"docker-sdk-python/{__version__}"
DEFAULT_NUM_POOLS = 25

# The OpenSSH server default value for MaxSessions is 10 which means we can
# use up to 9, leaving the final session for the underlying SSH connection.
# For more details see: https://github.com/docker/docker-py/issues/2246
DEFAULT_NUM_POOLS_SSH = 9

DEFAULT_MAX_POOL_SIZE = 10

DEFAULT_DATA_CHUNK_SIZE = 1024 * 2048

DEFAULT_SWARM_ADDR_POOL = ['10.0.0.0/8']
DEFAULT_SWARM_SUBNET_SIZE = 24
