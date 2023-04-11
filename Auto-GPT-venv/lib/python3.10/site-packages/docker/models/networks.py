from ..api import APIClient
from ..utils import version_gte
from .containers import Container
from .resource import Model, Collection


class Network(Model):
    """
    A Docker network.
    """
    @property
    def name(self):
        """
        The name of the network.
        """
        return self.attrs.get('Name')

    @property
    def containers(self):
        """
        The containers that are connected to the network, as a list of
        :py:class:`~docker.models.containers.Container` objects.
        """
        return [
            self.client.containers.get(cid) for cid in
            (self.attrs.get('Containers') or {}).keys()
        ]

    def connect(self, container, *args, **kwargs):
        """
        Connect a container to this network.

        Args:
            container (str): Container to connect to this network, as either
                an ID, name, or :py:class:`~docker.models.containers.Container`
                object.
            aliases (:py:class:`list`): A list of aliases for this endpoint.
                Names in that list can be used within the network to reach the
                container. Defaults to ``None``.
            links (:py:class:`list`): A list of links for this endpoint.
                Containers declared in this list will be linkedto this
                container. Defaults to ``None``.
            ipv4_address (str): The IP address of this container on the
                network, using the IPv4 protocol. Defaults to ``None``.
            ipv6_address (str): The IP address of this container on the
                network, using the IPv6 protocol. Defaults to ``None``.
            link_local_ips (:py:class:`list`): A list of link-local (IPv4/IPv6)
                addresses.
            driver_opt (dict): A dictionary of options to provide to the
                network driver. Defaults to ``None``.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        if isinstance(container, Container):
            container = container.id
        return self.client.api.connect_container_to_network(
            container, self.id, *args, **kwargs
        )

    def disconnect(self, container, *args, **kwargs):
        """
        Disconnect a container from this network.

        Args:
            container (str): Container to disconnect from this network, as
                either an ID, name, or
                :py:class:`~docker.models.containers.Container` object.
            force (bool): Force the container to disconnect from a network.
                Default: ``False``

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        if isinstance(container, Container):
            container = container.id
        return self.client.api.disconnect_container_from_network(
            container, self.id, *args, **kwargs
        )

    def remove(self):
        """
        Remove this network.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.remove_network(self.id)


class NetworkCollection(Collection):
    """
    Networks on the Docker server.
    """
    model = Network

    def create(self, name, *args, **kwargs):
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
            (:py:class:`Network`): The network that was created.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Example:
            A network using the bridge driver:

                >>> client.networks.create("network1", driver="bridge")

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
                >>> client.networks.create(
                    "network1",
                    driver="bridge",
                    ipam=ipam_config
                )

        """
        resp = self.client.api.create_network(name, *args, **kwargs)
        return self.get(resp['Id'])

    def get(self, network_id, *args, **kwargs):
        """
        Get a network by its ID.

        Args:
            network_id (str): The ID of the network.
            verbose (bool): Retrieve the service details across the cluster in
                swarm mode.
            scope (str): Filter the network by scope (``swarm``, ``global``
                or ``local``).

        Returns:
            (:py:class:`Network`) The network.

        Raises:
            :py:class:`docker.errors.NotFound`
                If the network does not exist.

            :py:class:`docker.errors.APIError`
                If the server returns an error.

        """
        return self.prepare_model(
            self.client.api.inspect_network(network_id, *args, **kwargs)
        )

    def list(self, *args, **kwargs):
        """
        List networks. Similar to the ``docker networks ls`` command.

        Args:
            names (:py:class:`list`): List of names to filter by.
            ids (:py:class:`list`): List of ids to filter by.
            filters (dict): Filters to be processed on the network list.
                Available filters:
                - ``driver=[<driver-name>]`` Matches a network's driver.
                - `label` (str|list): format either ``"key"``, ``"key=value"``
                    or a list of such.
                - ``type=["custom"|"builtin"]`` Filters networks by type.
            greedy (bool): Fetch more details for each network individually.
                You might want this to get the containers attached to them.

        Returns:
            (list of :py:class:`Network`) The networks on the server.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        greedy = kwargs.pop('greedy', False)
        resp = self.client.api.networks(*args, **kwargs)
        networks = [self.prepare_model(item) for item in resp]
        if greedy and version_gte(self.client.api._version, '1.28'):
            for net in networks:
                net.reload()
        return networks

    def prune(self, filters=None):
        return self.client.api.prune_networks(filters=filters)
    prune.__doc__ = APIClient.prune_networks.__doc__
