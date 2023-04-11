from ..errors import InvalidVersion
from ..utils import check_resource, minimum_version
from ..utils import version_lt
from .. import utils


class NetworkApiMixin:
    def networks(self, names=None, ids=None, filters=None):
        """
        List networks. Similar to the ``docker network ls`` command.

        Args:
            names (:py:class:`list`): List of names to filter by
            ids (:py:class:`list`): List of ids to filter by
            filters (dict): Filters to be processed on the network list.
                Available filters:
                - ``driver=[<driver-name>]`` Matches a network's driver.
                - ``label=[<key>]``, ``label=[<key>=<value>]`` or a list of
                    such.
                - ``type=["custom"|"builtin"]`` Filters networks by type.

        Returns:
            (dict): List of network objects.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """

        if filters is None:
            filters = {}
        if names:
            filters['name'] = names
        if ids:
            filters['id'] = ids
        params = {'filters': utils.convert_filters(filters)}
        url = self._url("/networks")
        res = self._get(url, params=params)
        return self._result(res, json=True)

    def create_network(self, name, driver=None, options=None, ipam=None,
                       check_duplicate=None, internal=False, labels=None,
                       enable_ipv6=False, attachable=None, scope=None,
                       ingress=None):
        """
        Create a network. Similar to the ``docker network create``.

        Args:
            name (str): Name of the network
            driver (str): Name of the driver used to create the network
            options (dict): Driver options as a key-value dictionary
            ipam (IPAMConfig): Optional custom IP scheme for the network.
            check_duplicate (bool): Request daemon to check for networks with
                same name. Default: ``None``.
            internal (bool): Restrict external access to the network. Default
                ``False``.
            labels (dict): Map of labels to set on the network. Default
                ``None``.
            enable_ipv6 (bool): Enable IPv6 on the network. Default ``False``.
            attachable (bool): If enabled, and the network is in the global
                scope,  non-service containers on worker nodes will be able to
                connect to the network.
            scope (str): Specify the network's scope (``local``, ``global`` or
                ``swarm``)
            ingress (bool): If set, create an ingress network which provides
                the routing-mesh in swarm mode.

        Returns:
            (dict): The created network reference object

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Example:
            A network using the bridge driver:

                >>> client.api.create_network("network1", driver="bridge")

            You can also create more advanced networks with custom IPAM
            configurations. For example, setting the subnet to
            ``192.168.52.0/24`` and gateway address to ``192.168.52.254``.

            .. code-block:: python

                >>> ipam_pool = docker.types.IPAMPool(
                    subnet='192.168.52.0/24',
                    gateway='192.168.52.254'
                )
                >>> ipam_config = docker.types.IPAMConfig(
                    pool_configs=[ipam_pool]
                )
                >>> client.api.create_network("network1", driver="bridge",
                                                 ipam=ipam_config)
        """
        if options is not None and not isinstance(options, dict):
            raise TypeError('options must be a dictionary')

        data = {
            'Name': name,
            'Driver': driver,
            'Options': options,
            'IPAM': ipam,
            'CheckDuplicate': check_duplicate,
        }

        if labels is not None:
            if version_lt(self._version, '1.23'):
                raise InvalidVersion(
                    'network labels were introduced in API 1.23'
                )
            if not isinstance(labels, dict):
                raise TypeError('labels must be a dictionary')
            data["Labels"] = labels

        if enable_ipv6:
            if version_lt(self._version, '1.23'):
                raise InvalidVersion(
                    'enable_ipv6 was introduced in API 1.23'
                )
            data['EnableIPv6'] = True

        if internal:
            if version_lt(self._version, '1.22'):
                raise InvalidVersion('Internal networks are not '
                                     'supported in API version < 1.22')
            data['Internal'] = True

        if attachable is not None:
            if version_lt(self._version, '1.24'):
                raise InvalidVersion(
                    'attachable is not supported in API version < 1.24'
                )
            data['Attachable'] = attachable

        if ingress is not None:
            if version_lt(self._version, '1.29'):
                raise InvalidVersion(
                    'ingress is not supported in API version < 1.29'
                )

            data['Ingress'] = ingress

        if scope is not None:
            if version_lt(self._version, '1.30'):
                raise InvalidVersion(
                    'scope is not supported in API version < 1.30'
                )
            data['Scope'] = scope

        url = self._url("/networks/create")
        res = self._post_json(url, data=data)
        return self._result(res, json=True)

    @minimum_version('1.25')
    def prune_networks(self, filters=None):
        """
        Delete unused networks

        Args:
            filters (dict): Filters to process on the prune list.

        Returns:
            (dict): A dict containing a list of deleted network names and
                the amount of disk space reclaimed in bytes.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        params = {}
        if filters:
            params['filters'] = utils.convert_filters(filters)
        url = self._url('/networks/prune')
        return self._result(self._post(url, params=params), True)

    @check_resource('net_id')
    def remove_network(self, net_id):
        """
        Remove a network. Similar to the ``docker network rm`` command.

        Args:
            net_id (str): The network's id
        """
        url = self._url("/networks/{0}", net_id)
        res = self._delete(url)
        self._raise_for_status(res)

    @check_resource('net_id')
    def inspect_network(self, net_id, verbose=None, scope=None):
        """
        Get detailed information about a network.

        Args:
            net_id (str): ID of network
            verbose (bool): Show the service details across the cluster in
                swarm mode.
            scope (str): Filter the network by scope (``swarm``, ``global``
                or ``local``).
        """
        params = {}
        if verbose is not None:
            if version_lt(self._version, '1.28'):
                raise InvalidVersion('verbose was introduced in API 1.28')
            params['verbose'] = verbose
        if scope is not None:
            if version_lt(self._version, '1.31'):
                raise InvalidVersion('scope was introduced in API 1.31')
            params['scope'] = scope

        url = self._url("/networks/{0}", net_id)
        res = self._get(url, params=params)
        return self._result(res, json=True)

    @check_resource('container')
    def connect_container_to_network(self, container, net_id,
                                     ipv4_address=None, ipv6_address=None,
                                     aliases=None, links=None,
                                     link_local_ips=None, driver_opt=None,
                                     mac_address=None):
        """
        Connect a container to a network.

        Args:
            container (str): container-id/name to be connected to the network
            net_id (str): network id
            aliases (:py:class:`list`): A list of aliases for this endpoint.
                Names in that list can be used within the network to reach the
                container. Defaults to ``None``.
            links (:py:class:`list`): A list of links for this endpoint.
                Containers declared in this list will be linked to this
                container. Defaults to ``None``.
            ipv4_address (str): The IP address of this container on the
                network, using the IPv4 protocol. Defaults to ``None``.
            ipv6_address (str): The IP address of this container on the
                network, using the IPv6 protocol. Defaults to ``None``.
            link_local_ips (:py:class:`list`): A list of link-local
                (IPv4/IPv6) addresses.
            mac_address (str): The MAC address of this container on the
                network. Defaults to ``None``.
        """
        data = {
            "Container": container,
            "EndpointConfig": self.create_endpoint_config(
                aliases=aliases, links=links, ipv4_address=ipv4_address,
                ipv6_address=ipv6_address, link_local_ips=link_local_ips,
                driver_opt=driver_opt,
                mac_address=mac_address
            ),
        }

        url = self._url("/networks/{0}/connect", net_id)
        res = self._post_json(url, data=data)
        self._raise_for_status(res)

    @check_resource('container')
    def disconnect_container_from_network(self, container, net_id,
                                          force=False):
        """
        Disconnect a container from a network.

        Args:
            container (str): container ID or name to be disconnected from the
                network
            net_id (str): network ID
            force (bool): Force the container to disconnect from a network.
                Default: ``False``
        """
        data = {"Container": container}
        if force:
            if version_lt(self._version, '1.22'):
                raise InvalidVersion(
                    'Forced disconnect was introduced in API 1.22'
                )
            data['Force'] = force
        url = self._url("/networks/{0}/disconnect", net_id)
        res = self._post_json(url, data=data)
        self._raise_for_status(res)
