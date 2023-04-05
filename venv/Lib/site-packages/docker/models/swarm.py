from docker.api import APIClient
from docker.errors import APIError
from .resource import Model


class Swarm(Model):
    """
    The server's Swarm state. This a singleton that must be reloaded to get
    the current state of the Swarm.
    """
    id_attribute = 'ID'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.client:
            try:
                self.reload()
            except APIError as e:
                # FIXME: https://github.com/docker/docker/issues/29192
                if e.response.status_code not in (406, 503):
                    raise

    @property
    def version(self):
        """
        The version number of the swarm. If this is not the same as the
        server, the :py:meth:`update` function will not work and you will
        need to call :py:meth:`reload` before calling it again.
        """
        return self.attrs.get('Version').get('Index')

    def get_unlock_key(self):
        return self.client.api.get_unlock_key()
    get_unlock_key.__doc__ = APIClient.get_unlock_key.__doc__

    def init(self, advertise_addr=None, listen_addr='0.0.0.0:2377',
             force_new_cluster=False, default_addr_pool=None,
             subnet_size=None, data_path_addr=None, data_path_port=None,
             **kwargs):
        """
        Initialize a new swarm on this Engine.

        Args:
            advertise_addr (str): Externally reachable address advertised to
                other nodes. This can either be an address/port combination in
                the form ``192.168.1.1:4567``, or an interface followed by a
                port number, like ``eth0:4567``. If the port number is omitted,
                the port number from the listen address is used.

                If not specified, it will be automatically detected when
                possible.
            listen_addr (str): Listen address used for inter-manager
                communication, as well as determining the networking interface
                used for the VXLAN Tunnel Endpoint (VTEP). This can either be
                an address/port combination in the form ``192.168.1.1:4567``,
                or an interface followed by a port number, like ``eth0:4567``.
                If the port number is omitted, the default swarm listening port
                is used. Default: ``0.0.0.0:2377``
            force_new_cluster (bool): Force creating a new Swarm, even if
                already part of one. Default: False
            default_addr_pool (list of str): Default Address Pool specifies
                default subnet pools for global scope networks. Each pool
                should be specified as a CIDR block, like '10.0.0.0/8'.
                Default: None
            subnet_size (int): SubnetSize specifies the subnet size of the
                networks created from the default subnet pool. Default: None
            data_path_addr (string): Address or interface to use for data path
                traffic. For example, 192.168.1.1, or an interface, like eth0.
            data_path_port (int): Port number to use for data path traffic.
                Acceptable port range is 1024 to 49151. If set to ``None`` or
                0, the default port 4789 will be used. Default: None
            task_history_retention_limit (int): Maximum number of tasks
                history stored.
            snapshot_interval (int): Number of logs entries between snapshot.
            keep_old_snapshots (int): Number of snapshots to keep beyond the
                current snapshot.
            log_entries_for_slow_followers (int): Number of log entries to
                keep around to sync up slow followers after a snapshot is
                created.
            heartbeat_tick (int): Amount of ticks (in seconds) between each
                heartbeat.
            election_tick (int): Amount of ticks (in seconds) needed without a
                leader to trigger a new election.
            dispatcher_heartbeat_period (int):  The delay for an agent to send
                a heartbeat to the dispatcher.
            node_cert_expiry (int): Automatic expiry for nodes certificates.
            external_ca (dict): Configuration for forwarding signing requests
                to an external certificate authority. Use
                ``docker.types.SwarmExternalCA``.
            name (string): Swarm's name
            labels (dict): User-defined key/value metadata.
            signing_ca_cert (str): The desired signing CA certificate for all
                swarm node TLS leaf certificates, in PEM format.
            signing_ca_key (str): The desired signing CA key for all swarm
                node TLS leaf certificates, in PEM format.
            ca_force_rotate (int): An integer whose purpose is to force swarm
                to generate a new signing CA certificate and key, if none have
                been specified.
            autolock_managers (boolean): If set, generate a key and use it to
                lock data stored on the managers.
            log_driver (DriverConfig): The default log driver to use for tasks
                created in the orchestrator.

        Returns:
            (str): The ID of the created node.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Example:

            >>> client.swarm.init(
                advertise_addr='eth0', listen_addr='0.0.0.0:5000',
                force_new_cluster=False, default_addr_pool=['10.20.0.0/16],
                subnet_size=24, snapshot_interval=5000,
                log_entries_for_slow_followers=1200
            )

        """
        init_kwargs = {
            'advertise_addr': advertise_addr,
            'listen_addr': listen_addr,
            'force_new_cluster': force_new_cluster,
            'default_addr_pool': default_addr_pool,
            'subnet_size': subnet_size,
            'data_path_addr': data_path_addr,
            'data_path_port': data_path_port,
        }
        init_kwargs['swarm_spec'] = self.client.api.create_swarm_spec(**kwargs)
        node_id = self.client.api.init_swarm(**init_kwargs)
        self.reload()
        return node_id

    def join(self, *args, **kwargs):
        return self.client.api.join_swarm(*args, **kwargs)
    join.__doc__ = APIClient.join_swarm.__doc__

    def leave(self, *args, **kwargs):
        return self.client.api.leave_swarm(*args, **kwargs)
    leave.__doc__ = APIClient.leave_swarm.__doc__

    def reload(self):
        """
        Inspect the swarm on the server and store the response in
        :py:attr:`attrs`.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        self.attrs = self.client.api.inspect_swarm()

    def unlock(self, key):
        return self.client.api.unlock_swarm(key)
    unlock.__doc__ = APIClient.unlock_swarm.__doc__

    def update(self, rotate_worker_token=False, rotate_manager_token=False,
               rotate_manager_unlock_key=False, **kwargs):
        """
        Update the swarm's configuration.

        It takes the same arguments as :py:meth:`init`, except
        ``advertise_addr``, ``listen_addr``, and ``force_new_cluster``. In
        addition, it takes these arguments:

        Args:
            rotate_worker_token (bool): Rotate the worker join token. Default:
                ``False``.
            rotate_manager_token (bool): Rotate the manager join token.
                Default: ``False``.
            rotate_manager_unlock_key (bool): Rotate the manager unlock key.
                Default: ``False``.
        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        """
        # this seems to have to be set
        if kwargs.get('node_cert_expiry') is None:
            kwargs['node_cert_expiry'] = 7776000000000000

        return self.client.api.update_swarm(
            version=self.version,
            swarm_spec=self.client.api.create_swarm_spec(**kwargs),
            rotate_worker_token=rotate_worker_token,
            rotate_manager_token=rotate_manager_token,
            rotate_manager_unlock_key=rotate_manager_unlock_key
        )
