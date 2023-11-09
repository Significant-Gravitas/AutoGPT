from .. import errors
from ..utils import normalize_links, version_lt


class EndpointConfig(dict):
    def __init__(self, version, aliases=None, links=None, ipv4_address=None,
                 ipv6_address=None, link_local_ips=None, driver_opt=None,
                 mac_address=None):
        if version_lt(version, '1.22'):
            raise errors.InvalidVersion(
                'Endpoint config is not supported for API version < 1.22'
            )

        if aliases:
            self["Aliases"] = aliases

        if links:
            self["Links"] = normalize_links(links)

        ipam_config = {}
        if ipv4_address:
            ipam_config['IPv4Address'] = ipv4_address

        if ipv6_address:
            ipam_config['IPv6Address'] = ipv6_address

        if mac_address:
            if version_lt(version, '1.25'):
                raise errors.InvalidVersion(
                    'mac_address is not supported for API version < 1.25'
                )
            self['MacAddress'] = mac_address

        if link_local_ips is not None:
            if version_lt(version, '1.24'):
                raise errors.InvalidVersion(
                    'link_local_ips is not supported for API version < 1.24'
                )
            ipam_config['LinkLocalIPs'] = link_local_ips

        if ipam_config:
            self['IPAMConfig'] = ipam_config

        if driver_opt:
            if version_lt(version, '1.32'):
                raise errors.InvalidVersion(
                    'DriverOpts is not supported for API version < 1.32'
                )
            if not isinstance(driver_opt, dict):
                raise TypeError('driver_opt must be a dictionary')
            self['DriverOpts'] = driver_opt


class NetworkingConfig(dict):
    def __init__(self, endpoints_config=None):
        if endpoints_config:
            self["EndpointsConfig"] = endpoints_config


class IPAMConfig(dict):
    """
    Create an IPAM (IP Address Management) config dictionary to be used with
    :py:meth:`~docker.api.network.NetworkApiMixin.create_network`.

    Args:

        driver (str): The IPAM driver to use. Defaults to ``default``.
        pool_configs (:py:class:`list`): A list of pool configurations
          (:py:class:`~docker.types.IPAMPool`). Defaults to empty list.
        options (dict): Driver options as a key-value dictionary.
          Defaults to `None`.

    Example:

        >>> ipam_config = docker.types.IPAMConfig(driver='default')
        >>> network = client.create_network('network1', ipam=ipam_config)

    """
    def __init__(self, driver='default', pool_configs=None, options=None):
        self.update({
            'Driver': driver,
            'Config': pool_configs or []
        })

        if options:
            if not isinstance(options, dict):
                raise TypeError('IPAMConfig options must be a dictionary')
            self['Options'] = options


class IPAMPool(dict):
    """
    Create an IPAM pool config dictionary to be added to the
    ``pool_configs`` parameter of
    :py:class:`~docker.types.IPAMConfig`.

    Args:

        subnet (str): Custom subnet for this IPAM pool using the CIDR
            notation. Defaults to ``None``.
        iprange (str): Custom IP range for endpoints in this IPAM pool using
            the CIDR notation. Defaults to ``None``.
        gateway (str): Custom IP address for the pool's gateway.
        aux_addresses (dict): A dictionary of ``key -> ip_address``
            relationships specifying auxiliary addresses that need to be
            allocated by the IPAM driver.

    Example:

        >>> ipam_pool = docker.types.IPAMPool(
            subnet='124.42.0.0/16',
            iprange='124.42.0.0/24',
            gateway='124.42.0.254',
            aux_addresses={
                'reserved1': '124.42.1.1'
            }
        )
        >>> ipam_config = docker.types.IPAMConfig(
                pool_configs=[ipam_pool])
    """
    def __init__(self, subnet=None, iprange=None, gateway=None,
                 aux_addresses=None):
        self.update({
            'Subnet': subnet,
            'IPRange': iprange,
            'Gateway': gateway,
            'AuxiliaryAddresses': aux_addresses
        })
