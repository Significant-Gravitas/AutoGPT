import base64
import collections
import json
import os
import os.path
import shlex
import string
from datetime import datetime
from packaging.version import Version

from .. import errors
from ..constants import DEFAULT_HTTP_HOST
from ..constants import DEFAULT_UNIX_SOCKET
from ..constants import DEFAULT_NPIPE
from ..constants import BYTE_UNITS
from ..tls import TLSConfig

from urllib.parse import urlparse, urlunparse


URLComponents = collections.namedtuple(
    'URLComponents',
    'scheme netloc url params query fragment',
)


def create_ipam_pool(*args, **kwargs):
    raise errors.DeprecatedMethod(
        'utils.create_ipam_pool has been removed. Please use a '
        'docker.types.IPAMPool object instead.'
    )


def create_ipam_config(*args, **kwargs):
    raise errors.DeprecatedMethod(
        'utils.create_ipam_config has been removed. Please use a '
        'docker.types.IPAMConfig object instead.'
    )


def decode_json_header(header):
    data = base64.b64decode(header)
    data = data.decode('utf-8')
    return json.loads(data)


def compare_version(v1, v2):
    """Compare docker versions

    >>> v1 = '1.9'
    >>> v2 = '1.10'
    >>> compare_version(v1, v2)
    1
    >>> compare_version(v2, v1)
    -1
    >>> compare_version(v2, v2)
    0
    """
    s1 = Version(v1)
    s2 = Version(v2)
    if s1 == s2:
        return 0
    elif s1 > s2:
        return -1
    else:
        return 1


def version_lt(v1, v2):
    return compare_version(v1, v2) > 0


def version_gte(v1, v2):
    return not version_lt(v1, v2)


def _convert_port_binding(binding):
    result = {'HostIp': '', 'HostPort': ''}
    if isinstance(binding, tuple):
        if len(binding) == 2:
            result['HostPort'] = binding[1]
            result['HostIp'] = binding[0]
        elif isinstance(binding[0], str):
            result['HostIp'] = binding[0]
        else:
            result['HostPort'] = binding[0]
    elif isinstance(binding, dict):
        if 'HostPort' in binding:
            result['HostPort'] = binding['HostPort']
            if 'HostIp' in binding:
                result['HostIp'] = binding['HostIp']
        else:
            raise ValueError(binding)
    else:
        result['HostPort'] = binding

    if result['HostPort'] is None:
        result['HostPort'] = ''
    else:
        result['HostPort'] = str(result['HostPort'])

    return result


def convert_port_bindings(port_bindings):
    result = {}
    for k, v in iter(port_bindings.items()):
        key = str(k)
        if '/' not in key:
            key += '/tcp'
        if isinstance(v, list):
            result[key] = [_convert_port_binding(binding) for binding in v]
        else:
            result[key] = [_convert_port_binding(v)]
    return result


def convert_volume_binds(binds):
    if isinstance(binds, list):
        return binds

    result = []
    for k, v in binds.items():
        if isinstance(k, bytes):
            k = k.decode('utf-8')

        if isinstance(v, dict):
            if 'ro' in v and 'mode' in v:
                raise ValueError(
                    'Binding cannot contain both "ro" and "mode": {}'
                    .format(repr(v))
                )

            bind = v['bind']
            if isinstance(bind, bytes):
                bind = bind.decode('utf-8')

            if 'ro' in v:
                mode = 'ro' if v['ro'] else 'rw'
            elif 'mode' in v:
                mode = v['mode']
            else:
                mode = 'rw'

            result.append(
                f'{k}:{bind}:{mode}'
            )
        else:
            if isinstance(v, bytes):
                v = v.decode('utf-8')
            result.append(
                f'{k}:{v}:rw'
            )
    return result


def convert_tmpfs_mounts(tmpfs):
    if isinstance(tmpfs, dict):
        return tmpfs

    if not isinstance(tmpfs, list):
        raise ValueError(
            'Expected tmpfs value to be either a list or a dict, found: {}'
            .format(type(tmpfs).__name__)
        )

    result = {}
    for mount in tmpfs:
        if isinstance(mount, str):
            if ":" in mount:
                name, options = mount.split(":", 1)
            else:
                name = mount
                options = ""

        else:
            raise ValueError(
                "Expected item in tmpfs list to be a string, found: {}"
                .format(type(mount).__name__)
            )

        result[name] = options
    return result


def convert_service_networks(networks):
    if not networks:
        return networks
    if not isinstance(networks, list):
        raise TypeError('networks parameter must be a list.')

    result = []
    for n in networks:
        if isinstance(n, str):
            n = {'Target': n}
        result.append(n)
    return result


def parse_repository_tag(repo_name):
    parts = repo_name.rsplit('@', 1)
    if len(parts) == 2:
        return tuple(parts)
    parts = repo_name.rsplit(':', 1)
    if len(parts) == 2 and '/' not in parts[1]:
        return tuple(parts)
    return repo_name, None


def parse_host(addr, is_win32=False, tls=False):
    # Sensible defaults
    if not addr and is_win32:
        return DEFAULT_NPIPE
    if not addr or addr.strip() == 'unix://':
        return DEFAULT_UNIX_SOCKET

    addr = addr.strip()

    parsed_url = urlparse(addr)
    proto = parsed_url.scheme
    if not proto or any([x not in string.ascii_letters + '+' for x in proto]):
        # https://bugs.python.org/issue754016
        parsed_url = urlparse('//' + addr, 'tcp')
        proto = 'tcp'

    if proto == 'fd':
        raise errors.DockerException('fd protocol is not implemented')

    # These protos are valid aliases for our library but not for the
    # official spec
    if proto == 'http' or proto == 'https':
        tls = proto == 'https'
        proto = 'tcp'
    elif proto == 'http+unix':
        proto = 'unix'

    if proto not in ('tcp', 'unix', 'npipe', 'ssh'):
        raise errors.DockerException(
            f"Invalid bind address protocol: {addr}"
        )

    if proto == 'tcp' and not parsed_url.netloc:
        # "tcp://" is exceptionally disallowed by convention;
        # omitting a hostname for other protocols is fine
        raise errors.DockerException(
            f'Invalid bind address format: {addr}'
        )

    if any([
        parsed_url.params, parsed_url.query, parsed_url.fragment,
        parsed_url.password
    ]):
        raise errors.DockerException(
            f'Invalid bind address format: {addr}'
        )

    if parsed_url.path and proto == 'ssh':
        raise errors.DockerException(
            'Invalid bind address format: no path allowed for this protocol:'
            ' {}'.format(addr)
        )
    else:
        path = parsed_url.path
        if proto == 'unix' and parsed_url.hostname is not None:
            # For legacy reasons, we consider unix://path
            # to be valid and equivalent to unix:///path
            path = '/'.join((parsed_url.hostname, path))

    netloc = parsed_url.netloc
    if proto in ('tcp', 'ssh'):
        port = parsed_url.port or 0
        if port <= 0:
            if proto != 'ssh':
                raise errors.DockerException(
                    'Invalid bind address format: port is required:'
                    ' {}'.format(addr)
                )
            port = 22
            netloc = f'{parsed_url.netloc}:{port}'

        if not parsed_url.hostname:
            netloc = f'{DEFAULT_HTTP_HOST}:{port}'

    # Rewrite schemes to fit library internals (requests adapters)
    if proto == 'tcp':
        proto = 'http{}'.format('s' if tls else '')
    elif proto == 'unix':
        proto = 'http+unix'

    if proto in ('http+unix', 'npipe'):
        return f"{proto}://{path}".rstrip('/')

    return urlunparse(URLComponents(
        scheme=proto,
        netloc=netloc,
        url=path,
        params='',
        query='',
        fragment='',
    )).rstrip('/')


def parse_devices(devices):
    device_list = []
    for device in devices:
        if isinstance(device, dict):
            device_list.append(device)
            continue
        if not isinstance(device, str):
            raise errors.DockerException(
                f'Invalid device type {type(device)}'
            )
        device_mapping = device.split(':')
        if device_mapping:
            path_on_host = device_mapping[0]
            if len(device_mapping) > 1:
                path_in_container = device_mapping[1]
            else:
                path_in_container = path_on_host
            if len(device_mapping) > 2:
                permissions = device_mapping[2]
            else:
                permissions = 'rwm'
            device_list.append({
                'PathOnHost': path_on_host,
                'PathInContainer': path_in_container,
                'CgroupPermissions': permissions
            })
    return device_list


def kwargs_from_env(ssl_version=None, assert_hostname=None, environment=None):
    if not environment:
        environment = os.environ
    host = environment.get('DOCKER_HOST')

    # empty string for cert path is the same as unset.
    cert_path = environment.get('DOCKER_CERT_PATH') or None

    # empty string for tls verify counts as "false".
    # Any value or 'unset' counts as true.
    tls_verify = environment.get('DOCKER_TLS_VERIFY')
    if tls_verify == '':
        tls_verify = False
    else:
        tls_verify = tls_verify is not None
    enable_tls = cert_path or tls_verify

    params = {}

    if host:
        params['base_url'] = host

    if not enable_tls:
        return params

    if not cert_path:
        cert_path = os.path.join(os.path.expanduser('~'), '.docker')

    if not tls_verify and assert_hostname is None:
        # assert_hostname is a subset of TLS verification,
        # so if it's not set already then set it to false.
        assert_hostname = False

    params['tls'] = TLSConfig(
        client_cert=(os.path.join(cert_path, 'cert.pem'),
                     os.path.join(cert_path, 'key.pem')),
        ca_cert=os.path.join(cert_path, 'ca.pem'),
        verify=tls_verify,
        ssl_version=ssl_version,
        assert_hostname=assert_hostname,
    )

    return params


def convert_filters(filters):
    result = {}
    for k, v in iter(filters.items()):
        if isinstance(v, bool):
            v = 'true' if v else 'false'
        if not isinstance(v, list):
            v = [v, ]
        result[k] = [
            str(item) if not isinstance(item, str) else item
            for item in v
        ]
    return json.dumps(result)


def datetime_to_timestamp(dt):
    """Convert a UTC datetime to a Unix timestamp"""
    delta = dt - datetime.utcfromtimestamp(0)
    return delta.seconds + delta.days * 24 * 3600


def parse_bytes(s):
    if isinstance(s, (int, float,)):
        return s
    if len(s) == 0:
        return 0

    if s[-2:-1].isalpha() and s[-1].isalpha():
        if s[-1] == "b" or s[-1] == "B":
            s = s[:-1]
    units = BYTE_UNITS
    suffix = s[-1].lower()

    # Check if the variable is a string representation of an int
    # without a units part. Assuming that the units are bytes.
    if suffix.isdigit():
        digits_part = s
        suffix = 'b'
    else:
        digits_part = s[:-1]

    if suffix in units.keys() or suffix.isdigit():
        try:
            digits = float(digits_part)
        except ValueError:
            raise errors.DockerException(
                'Failed converting the string value for memory ({}) to'
                ' an integer.'.format(digits_part)
            )

        # Reconvert to long for the final result
        s = int(digits * units[suffix])
    else:
        raise errors.DockerException(
            'The specified value for memory ({}) should specify the'
            ' units. The postfix should be one of the `b` `k` `m` `g`'
            ' characters'.format(s)
        )

    return s


def normalize_links(links):
    if isinstance(links, dict):
        links = iter(links.items())

    return [f'{k}:{v}' if v else k for k, v in sorted(links)]


def parse_env_file(env_file):
    """
    Reads a line-separated environment file.
    The format of each line should be "key=value".
    """
    environment = {}

    with open(env_file) as f:
        for line in f:

            if line[0] == '#':
                continue

            line = line.strip()
            if not line:
                continue

            parse_line = line.split('=', 1)
            if len(parse_line) == 2:
                k, v = parse_line
                environment[k] = v
            else:
                raise errors.DockerException(
                    'Invalid line in environment file {}:\n{}'.format(
                        env_file, line))

    return environment


def split_command(command):
    return shlex.split(command)


def format_environment(environment):
    def format_env(key, value):
        if value is None:
            return key
        if isinstance(value, bytes):
            value = value.decode('utf-8')

        return f'{key}={value}'
    return [format_env(*var) for var in iter(environment.items())]


def format_extra_hosts(extra_hosts, task=False):
    # Use format dictated by Swarm API if container is part of a task
    if task:
        return [
            f'{v} {k}' for k, v in sorted(iter(extra_hosts.items()))
        ]

    return [
        f'{k}:{v}' for k, v in sorted(iter(extra_hosts.items()))
    ]


def create_host_config(self, *args, **kwargs):
    raise errors.DeprecatedMethod(
        'utils.create_host_config has been removed. Please use a '
        'docker.types.HostConfig object instead.'
    )
