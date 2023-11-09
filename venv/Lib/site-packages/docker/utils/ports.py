import re

PORT_SPEC = re.compile(
    "^"  # Match full string
    "("  # External part
    r"(\[?(?P<host>[a-fA-F\d.:]+)\]?:)?"  # Address
    r"(?P<ext>[\d]*)(-(?P<ext_end>[\d]+))?:"  # External range
    ")?"
    r"(?P<int>[\d]+)(-(?P<int_end>[\d]+))?"  # Internal range
    "(?P<proto>/(udp|tcp|sctp))?"  # Protocol
    "$"  # Match full string
)


def add_port_mapping(port_bindings, internal_port, external):
    if internal_port in port_bindings:
        port_bindings[internal_port].append(external)
    else:
        port_bindings[internal_port] = [external]


def add_port(port_bindings, internal_port_range, external_range):
    if external_range is None:
        for internal_port in internal_port_range:
            add_port_mapping(port_bindings, internal_port, None)
    else:
        ports = zip(internal_port_range, external_range)
        for internal_port, external_port in ports:
            add_port_mapping(port_bindings, internal_port, external_port)


def build_port_bindings(ports):
    port_bindings = {}
    for port in ports:
        internal_port_range, external_range = split_port(port)
        add_port(port_bindings, internal_port_range, external_range)
    return port_bindings


def _raise_invalid_port(port):
    raise ValueError('Invalid port "%s", should be '
                     '[[remote_ip:]remote_port[-remote_port]:]'
                     'port[/protocol]' % port)


def port_range(start, end, proto, randomly_available_port=False):
    if not start:
        return start
    if not end:
        return [start + proto]
    if randomly_available_port:
        return [f'{start}-{end}' + proto]
    return [str(port) + proto for port in range(int(start), int(end) + 1)]


def split_port(port):
    if hasattr(port, 'legacy_repr'):
        # This is the worst hack, but it prevents a bug in Compose 1.14.0
        # https://github.com/docker/docker-py/issues/1668
        # TODO: remove once fixed in Compose stable
        port = port.legacy_repr()
    port = str(port)
    match = PORT_SPEC.match(port)
    if match is None:
        _raise_invalid_port(port)
    parts = match.groupdict()

    host = parts['host']
    proto = parts['proto'] or ''
    internal = port_range(parts['int'], parts['int_end'], proto)
    external = port_range(
        parts['ext'], parts['ext_end'], '', len(internal) == 1)

    if host is None:
        if external is not None and len(internal) != len(external):
            raise ValueError('Port ranges don\'t match in length')
        return internal, external
    else:
        if not external:
            external = [None] * len(internal)
        elif len(internal) != len(external):
            raise ValueError('Port ranges don\'t match in length')
        return internal, [(host, ext_port) for ext_port in external]
