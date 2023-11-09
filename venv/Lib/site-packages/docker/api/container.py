from datetime import datetime

from .. import errors
from .. import utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
from ..types import CancellableStream
from ..types import ContainerConfig
from ..types import EndpointConfig
from ..types import HostConfig
from ..types import NetworkingConfig


class ContainerApiMixin:
    @utils.check_resource('container')
    def attach(self, container, stdout=True, stderr=True,
               stream=False, logs=False, demux=False):
        """
        Attach to a container.

        The ``.logs()`` function is a wrapper around this method, which you can
        use instead if you want to fetch/stream container output without first
        retrieving the entire backlog.

        Args:
            container (str): The container to attach to.
            stdout (bool): Include stdout.
            stderr (bool): Include stderr.
            stream (bool): Return container output progressively as an iterator
                of strings, rather than a single string.
            logs (bool): Include the container's previous output.
            demux (bool): Keep stdout and stderr separate.

        Returns:
            By default, the container's output as a single string (two if
            ``demux=True``: one for stdout and one for stderr).

            If ``stream=True``, an iterator of output strings. If
            ``demux=True``, two iterators are returned: one for stdout and one
            for stderr.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        params = {
            'logs': logs and 1 or 0,
            'stdout': stdout and 1 or 0,
            'stderr': stderr and 1 or 0,
            'stream': stream and 1 or 0
        }

        headers = {
            'Connection': 'Upgrade',
            'Upgrade': 'tcp'
        }

        u = self._url("/containers/{0}/attach", container)
        response = self._post(u, headers=headers, params=params, stream=True)

        output = self._read_from_socket(
            response, stream, self._check_is_tty(container), demux=demux)

        if stream:
            return CancellableStream(output, response)
        else:
            return output

    @utils.check_resource('container')
    def attach_socket(self, container, params=None, ws=False):
        """
        Like ``attach``, but returns the underlying socket-like object for the
        HTTP request.

        Args:
            container (str): The container to attach to.
            params (dict): Dictionary of request parameters (e.g. ``stdout``,
                ``stderr``, ``stream``).
                For ``detachKeys``, ~/.docker/config.json is used by default.
            ws (bool): Use websockets instead of raw HTTP.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        if params is None:
            params = {
                'stdout': 1,
                'stderr': 1,
                'stream': 1
            }

        if 'detachKeys' not in params \
                and 'detachKeys' in self._general_configs:

            params['detachKeys'] = self._general_configs['detachKeys']

        if ws:
            return self._attach_websocket(container, params)

        headers = {
            'Connection': 'Upgrade',
            'Upgrade': 'tcp'
        }

        u = self._url("/containers/{0}/attach", container)
        return self._get_raw_response_socket(
            self.post(
                u, None, params=self._attach_params(params), stream=True,
                headers=headers
            )
        )

    @utils.check_resource('container')
    def commit(self, container, repository=None, tag=None, message=None,
               author=None, changes=None, conf=None):
        """
        Commit a container to an image. Similar to the ``docker commit``
        command.

        Args:
            container (str): The image hash of the container
            repository (str): The repository to push the image to
            tag (str): The tag to push
            message (str): A commit message
            author (str): The name of the author
            changes (str): Dockerfile instructions to apply while committing
            conf (dict): The configuration for the container. See the
                `Engine API documentation
                <https://docs.docker.com/reference/api/docker_remote_api/>`_
                for full details.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        params = {
            'container': container,
            'repo': repository,
            'tag': tag,
            'comment': message,
            'author': author,
            'changes': changes
        }
        u = self._url("/commit")
        return self._result(
            self._post_json(u, data=conf, params=params), json=True
        )

    def containers(self, quiet=False, all=False, trunc=False, latest=False,
                   since=None, before=None, limit=-1, size=False,
                   filters=None):
        """
        List containers. Similar to the ``docker ps`` command.

        Args:
            quiet (bool): Only display numeric Ids
            all (bool): Show all containers. Only running containers are shown
                by default
            trunc (bool): Truncate output
            latest (bool): Show only the latest created container, include
                non-running ones.
            since (str): Show only containers created since Id or Name, include
                non-running ones
            before (str): Show only container created before Id or Name,
                include non-running ones
            limit (int): Show `limit` last created containers, include
                non-running ones
            size (bool): Display sizes
            filters (dict): Filters to be processed on the image list.
                Available filters:

                - `exited` (int): Only containers with specified exit code
                - `status` (str): One of ``restarting``, ``running``,
                    ``paused``, ``exited``
                - `label` (str|list): format either ``"key"``, ``"key=value"``
                    or a list of such.
                - `id` (str): The id of the container.
                - `name` (str): The name of the container.
                - `ancestor` (str): Filter by container ancestor. Format of
                    ``<image-name>[:tag]``, ``<image-id>``, or
                    ``<image@digest>``.
                - `before` (str): Only containers created before a particular
                    container. Give the container name or id.
                - `since` (str): Only containers created after a particular
                    container. Give container name or id.

                A comprehensive list can be found in the documentation for
                `docker ps
                <https://docs.docker.com/engine/reference/commandline/ps>`_.

        Returns:
            A list of dicts, one per container

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        params = {
            'limit': 1 if latest else limit,
            'all': 1 if all else 0,
            'size': 1 if size else 0,
            'trunc_cmd': 1 if trunc else 0,
            'since': since,
            'before': before
        }
        if filters:
            params['filters'] = utils.convert_filters(filters)
        u = self._url("/containers/json")
        res = self._result(self._get(u, params=params), True)

        if quiet:
            return [{'Id': x['Id']} for x in res]
        if trunc:
            for x in res:
                x['Id'] = x['Id'][:12]
        return res

    def create_container(self, image, command=None, hostname=None, user=None,
                         detach=False, stdin_open=False, tty=False, ports=None,
                         environment=None, volumes=None,
                         network_disabled=False, name=None, entrypoint=None,
                         working_dir=None, domainname=None, host_config=None,
                         mac_address=None, labels=None, stop_signal=None,
                         networking_config=None, healthcheck=None,
                         stop_timeout=None, runtime=None,
                         use_config_proxy=True, platform=None):
        """
        Creates a container. Parameters are similar to those for the ``docker
        run`` command except it doesn't support the attach options (``-a``).

        The arguments that are passed directly to this function are
        host-independent configuration options. Host-specific configuration
        is passed with the `host_config` argument. You'll normally want to
        use this method in combination with the :py:meth:`create_host_config`
        method to generate ``host_config``.

        **Port bindings**

        Port binding is done in two parts: first, provide a list of ports to
        open inside the container with the ``ports`` parameter, then declare
        bindings with the ``host_config`` parameter. For example:

        .. code-block:: python

            container_id = client.api.create_container(
                'busybox', 'ls', ports=[1111, 2222],
                host_config=client.api.create_host_config(port_bindings={
                    1111: 4567,
                    2222: None
                })
            )


        You can limit the host address on which the port will be exposed like
        such:

        .. code-block:: python

            client.api.create_host_config(
                port_bindings={1111: ('127.0.0.1', 4567)}
            )

        Or without host port assignment:

        .. code-block:: python

            client.api.create_host_config(port_bindings={1111: ('127.0.0.1',)})

        If you wish to use UDP instead of TCP (default), you need to declare
        ports as such in both the config and host config:

        .. code-block:: python

            container_id = client.api.create_container(
                'busybox', 'ls', ports=[(1111, 'udp'), 2222],
                host_config=client.api.create_host_config(port_bindings={
                    '1111/udp': 4567, 2222: None
                })
            )

        To bind multiple host ports to a single container port, use the
        following syntax:

        .. code-block:: python

            client.api.create_host_config(port_bindings={
                1111: [1234, 4567]
            })

        You can also bind multiple IPs to a single container port:

        .. code-block:: python

            client.api.create_host_config(port_bindings={
                1111: [
                    ('192.168.0.100', 1234),
                    ('192.168.0.101', 1234)
                ]
            })

        **Using volumes**

        Volume declaration is done in two parts. Provide a list of
        paths to use as mountpoints inside the container with the
        ``volumes`` parameter, and declare mappings from paths on the host
        in the ``host_config`` section.

        .. code-block:: python

            container_id = client.api.create_container(
                'busybox', 'ls', volumes=['/mnt/vol1', '/mnt/vol2'],
                host_config=client.api.create_host_config(binds={
                    '/home/user1/': {
                        'bind': '/mnt/vol2',
                        'mode': 'rw',
                    },
                    '/var/www': {
                        'bind': '/mnt/vol1',
                        'mode': 'ro',
                    }
                })
            )

        You can alternatively specify binds as a list. This code is equivalent
        to the example above:

        .. code-block:: python

            container_id = client.api.create_container(
                'busybox', 'ls', volumes=['/mnt/vol1', '/mnt/vol2'],
                host_config=client.api.create_host_config(binds=[
                    '/home/user1/:/mnt/vol2',
                    '/var/www:/mnt/vol1:ro',
                ])
            )

        **Networking**

        You can specify networks to connect the container to by using the
        ``networking_config`` parameter. At the time of creation, you can
        only connect a container to a single networking, but you
        can create more connections by using
        :py:meth:`~connect_container_to_network`.

        For example:

        .. code-block:: python

            networking_config = client.api.create_networking_config({
                'network1': client.api.create_endpoint_config(
                    ipv4_address='172.28.0.124',
                    aliases=['foo', 'bar'],
                    links=['container2']
                )
            })

            ctnr = client.api.create_container(
                img, command, networking_config=networking_config
            )

        Args:
            image (str): The image to run
            command (str or list): The command to be run in the container
            hostname (str): Optional hostname for the container
            user (str or int): Username or UID
            detach (bool): Detached mode: run container in the background and
                return container ID
            stdin_open (bool): Keep STDIN open even if not attached
            tty (bool): Allocate a pseudo-TTY
            ports (list of ints): A list of port numbers
            environment (dict or list): A dictionary or a list of strings in
                the following format ``["PASSWORD=xxx"]`` or
                ``{"PASSWORD": "xxx"}``.
            volumes (str or list): List of paths inside the container to use
                as volumes.
            network_disabled (bool): Disable networking
            name (str): A name for the container
            entrypoint (str or list): An entrypoint
            working_dir (str): Path to the working directory
            domainname (str): The domain name to use for the container
            host_config (dict): A dictionary created with
                :py:meth:`create_host_config`.
            mac_address (str): The Mac Address to assign the container
            labels (dict or list): A dictionary of name-value labels (e.g.
                ``{"label1": "value1", "label2": "value2"}``) or a list of
                names of labels to set with empty values (e.g.
                ``["label1", "label2"]``)
            stop_signal (str): The stop signal to use to stop the container
                (e.g. ``SIGINT``).
            stop_timeout (int): Timeout to stop the container, in seconds.
                Default: 10
            networking_config (dict): A networking configuration generated
                by :py:meth:`create_networking_config`.
            runtime (str): Runtime to use with this container.
            healthcheck (dict): Specify a test to perform to check that the
                container is healthy.
            use_config_proxy (bool): If ``True``, and if the docker client
                configuration file (``~/.docker/config.json`` by default)
                contains a proxy configuration, the corresponding environment
                variables will be set in the container being created.
            platform (str): Platform in the format ``os[/arch[/variant]]``.

        Returns:
            A dictionary with an image 'Id' key and a 'Warnings' key.

        Raises:
            :py:class:`docker.errors.ImageNotFound`
                If the specified image does not exist.
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        if isinstance(volumes, str):
            volumes = [volumes, ]

        if isinstance(environment, dict):
            environment = utils.utils.format_environment(environment)

        if use_config_proxy:
            environment = self._proxy_configs.inject_proxy_environment(
                environment
            ) or None

        config = self.create_container_config(
            image, command, hostname, user, detach, stdin_open, tty,
            ports, environment, volumes,
            network_disabled, entrypoint, working_dir, domainname,
            host_config, mac_address, labels,
            stop_signal, networking_config, healthcheck,
            stop_timeout, runtime
        )
        return self.create_container_from_config(config, name, platform)

    def create_container_config(self, *args, **kwargs):
        return ContainerConfig(self._version, *args, **kwargs)

    def create_container_from_config(self, config, name=None, platform=None):
        u = self._url("/containers/create")
        params = {
            'name': name
        }
        if platform:
            if utils.version_lt(self._version, '1.41'):
                raise errors.InvalidVersion(
                    'platform is not supported for API version < 1.41'
                )
            params['platform'] = platform
        res = self._post_json(u, data=config, params=params)
        return self._result(res, True)

    def create_host_config(self, *args, **kwargs):
        """
        Create a dictionary for the ``host_config`` argument to
        :py:meth:`create_container`.

        Args:
            auto_remove (bool): enable auto-removal of the container on daemon
                side when the container's process exits.
            binds (dict): Volumes to bind. See :py:meth:`create_container`
                    for more information.
            blkio_weight_device: Block IO weight (relative device weight) in
                the form of: ``[{"Path": "device_path", "Weight": weight}]``.
            blkio_weight: Block IO weight (relative weight), accepts a weight
                value between 10 and 1000.
            cap_add (list of str): Add kernel capabilities. For example,
                ``["SYS_ADMIN", "MKNOD"]``.
            cap_drop (list of str): Drop kernel capabilities.
            cpu_period (int): The length of a CPU period in microseconds.
            cpu_quota (int): Microseconds of CPU time that the container can
                get in a CPU period.
            cpu_shares (int): CPU shares (relative weight).
            cpuset_cpus (str): CPUs in which to allow execution (``0-3``,
                ``0,1``).
            cpuset_mems (str): Memory nodes (MEMs) in which to allow execution
                (``0-3``, ``0,1``). Only effective on NUMA systems.
            device_cgroup_rules (:py:class:`list`): A list of cgroup rules to
                apply to the container.
            device_read_bps: Limit read rate (bytes per second) from a device
                in the form of: `[{"Path": "device_path", "Rate": rate}]`
            device_read_iops: Limit read rate (IO per second) from a device.
            device_write_bps: Limit write rate (bytes per second) from a
                device.
            device_write_iops: Limit write rate (IO per second) from a device.
            devices (:py:class:`list`): Expose host devices to the container,
                as a list of strings in the form
                ``<path_on_host>:<path_in_container>:<cgroup_permissions>``.

                For example, ``/dev/sda:/dev/xvda:rwm`` allows the container
                to have read-write access to the host's ``/dev/sda`` via a
                node named ``/dev/xvda`` inside the container.
            device_requests (:py:class:`list`): Expose host resources such as
                GPUs to the container, as a list of
                :py:class:`docker.types.DeviceRequest` instances.
            dns (:py:class:`list`): Set custom DNS servers.
            dns_opt (:py:class:`list`): Additional options to be added to the
                container's ``resolv.conf`` file
            dns_search (:py:class:`list`): DNS search domains.
            extra_hosts (dict): Additional hostnames to resolve inside the
                container, as a mapping of hostname to IP address.
            group_add (:py:class:`list`): List of additional group names and/or
                IDs that the container process will run as.
            init (bool): Run an init inside the container that forwards
                signals and reaps processes
            ipc_mode (str): Set the IPC mode for the container.
            isolation (str): Isolation technology to use. Default: ``None``.
            links (dict): Mapping of links using the
                ``{'container': 'alias'}`` format. The alias is optional.
                Containers declared in this dict will be linked to the new
                container using the provided alias. Default: ``None``.
            log_config (LogConfig): Logging configuration
            lxc_conf (dict): LXC config.
            mem_limit (float or str): Memory limit. Accepts float values
                (which represent the memory limit of the created container in
                bytes) or a string with a units identification char
                (``100000b``, ``1000k``, ``128m``, ``1g``). If a string is
                specified without a units character, bytes are assumed as an
            mem_reservation (float or str): Memory soft limit.
            mem_swappiness (int): Tune a container's memory swappiness
                behavior. Accepts number between 0 and 100.
            memswap_limit (str or int): Maximum amount of memory + swap a
                container is allowed to consume.
            mounts (:py:class:`list`): Specification for mounts to be added to
                the container. More powerful alternative to ``binds``. Each
                item in the list is expected to be a
                :py:class:`docker.types.Mount` object.
            network_mode (str): One of:

                - ``bridge`` Create a new network stack for the container on
                  the bridge network.
                - ``none`` No networking for this container.
                - ``container:<name|id>`` Reuse another container's network
                  stack.
                - ``host`` Use the host network stack.
                  This mode is incompatible with ``port_bindings``.

            oom_kill_disable (bool): Whether to disable OOM killer.
            oom_score_adj (int): An integer value containing the score given
                to the container in order to tune OOM killer preferences.
            pid_mode (str): If set to ``host``, use the host PID namespace
                inside the container.
            pids_limit (int): Tune a container's pids limit. Set ``-1`` for
                unlimited.
            port_bindings (dict): See :py:meth:`create_container`
                for more information.
                Imcompatible with ``host`` in ``network_mode``.
            privileged (bool): Give extended privileges to this container.
            publish_all_ports (bool): Publish all ports to the host.
            read_only (bool): Mount the container's root filesystem as read
                only.
            restart_policy (dict): Restart the container when it exits.
                Configured as a dictionary with keys:

                - ``Name`` One of ``on-failure``, or ``always``.
                - ``MaximumRetryCount`` Number of times to restart the
                  container on failure.
            security_opt (:py:class:`list`): A list of string values to
                customize labels for MLS systems, such as SELinux.
            shm_size (str or int): Size of /dev/shm (e.g. ``1G``).
            storage_opt (dict): Storage driver options per container as a
                key-value mapping.
            sysctls (dict): Kernel parameters to set in the container.
            tmpfs (dict): Temporary filesystems to mount, as a dictionary
                mapping a path inside the container to options for that path.

                For example:

                .. code-block:: python

                    {
                        '/mnt/vol2': '',
                        '/mnt/vol1': 'size=3G,uid=1000'
                    }

            ulimits (:py:class:`list`): Ulimits to set inside the container,
                as a list of :py:class:`docker.types.Ulimit` instances.
            userns_mode (str): Sets the user namespace mode for the container
                when user namespace remapping option is enabled. Supported
                values are: ``host``
            uts_mode (str): Sets the UTS namespace mode for the container.
                Supported values are: ``host``
            volumes_from (:py:class:`list`): List of container names or IDs to
                get volumes from.
            runtime (str): Runtime to use with this container.


        Returns:
            (dict) A dictionary which can be passed to the ``host_config``
            argument to :py:meth:`create_container`.

        Example:

            >>> client.api.create_host_config(
            ...     privileged=True,
            ...     cap_drop=['MKNOD'],
            ...     volumes_from=['nostalgic_newton'],
            ... )
            {'CapDrop': ['MKNOD'], 'LxcConf': None, 'Privileged': True,
            'VolumesFrom': ['nostalgic_newton'], 'PublishAllPorts': False}

"""
        if not kwargs:
            kwargs = {}
        if 'version' in kwargs:
            raise TypeError(
                "create_host_config() got an unexpected "
                "keyword argument 'version'"
            )
        kwargs['version'] = self._version
        return HostConfig(*args, **kwargs)

    def create_networking_config(self, *args, **kwargs):
        """
        Create a networking config dictionary to be used as the
        ``networking_config`` parameter in :py:meth:`create_container`.

        Args:
            endpoints_config (dict): A dictionary mapping network names to
                endpoint configurations generated by
                :py:meth:`create_endpoint_config`.

        Returns:
            (dict) A networking config.

        Example:

            >>> client.api.create_network('network1')
            >>> networking_config = client.api.create_networking_config({
                'network1': client.api.create_endpoint_config()
            })
            >>> container = client.api.create_container(
                img, command, networking_config=networking_config
            )

        """
        return NetworkingConfig(*args, **kwargs)

    def create_endpoint_config(self, *args, **kwargs):
        """
        Create an endpoint config dictionary to be used with
        :py:meth:`create_networking_config`.

        Args:
            aliases (:py:class:`list`): A list of aliases for this endpoint.
                Names in that list can be used within the network to reach the
                container. Defaults to ``None``.
            links (dict): Mapping of links for this endpoint using the
                ``{'container': 'alias'}`` format. The alias is optional.
                Containers declared in this dict will be linked to this
                container using the provided alias. Defaults to ``None``.
            ipv4_address (str): The IP address of this container on the
                network, using the IPv4 protocol. Defaults to ``None``.
            ipv6_address (str): The IP address of this container on the
                network, using the IPv6 protocol. Defaults to ``None``.
            link_local_ips (:py:class:`list`): A list of link-local (IPv4/IPv6)
                addresses.
            driver_opt (dict): A dictionary of options to provide to the
                network driver. Defaults to ``None``.

        Returns:
            (dict) An endpoint config.

        Example:

            >>> endpoint_config = client.api.create_endpoint_config(
                aliases=['web', 'app'],
                links={'app_db': 'db', 'another': None},
                ipv4_address='132.65.0.123'
            )

        """
        return EndpointConfig(self._version, *args, **kwargs)

    @utils.check_resource('container')
    def diff(self, container):
        """
        Inspect changes on a container's filesystem.

        Args:
            container (str): The container to diff

        Returns:
            (str)

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self._result(
            self._get(self._url("/containers/{0}/changes", container)), True
        )

    @utils.check_resource('container')
    def export(self, container, chunk_size=DEFAULT_DATA_CHUNK_SIZE):
        """
        Export the contents of a filesystem as a tar archive.

        Args:
            container (str): The container to export
            chunk_size (int): The number of bytes returned by each iteration
                of the generator. If ``None``, data will be streamed as it is
                received. Default: 2 MB

        Returns:
            (generator): The archived filesystem data stream

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        res = self._get(
            self._url("/containers/{0}/export", container), stream=True
        )
        return self._stream_raw_result(res, chunk_size, False)

    @utils.check_resource('container')
    def get_archive(self, container, path, chunk_size=DEFAULT_DATA_CHUNK_SIZE,
                    encode_stream=False):
        """
        Retrieve a file or folder from a container in the form of a tar
        archive.

        Args:
            container (str): The container where the file is located
            path (str): Path to the file or folder to retrieve
            chunk_size (int): The number of bytes returned by each iteration
                of the generator. If ``None``, data will be streamed as it is
                received. Default: 2 MB
            encode_stream (bool): Determines if data should be encoded
                (gzip-compressed) during transmission. Default: False

        Returns:
            (tuple): First element is a raw tar data stream. Second element is
            a dict containing ``stat`` information on the specified ``path``.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Example:

            >>> c = docker.APIClient()
            >>> f = open('./sh_bin.tar', 'wb')
            >>> bits, stat = c.api.get_archive(container, '/bin/sh')
            >>> print(stat)
            {'name': 'sh', 'size': 1075464, 'mode': 493,
             'mtime': '2018-10-01T15:37:48-07:00', 'linkTarget': ''}
            >>> for chunk in bits:
            ...    f.write(chunk)
            >>> f.close()
        """
        params = {
            'path': path
        }
        headers = {
            "Accept-Encoding": "gzip, deflate"
        } if encode_stream else {
            "Accept-Encoding": "identity"
        }
        url = self._url('/containers/{0}/archive', container)
        res = self._get(url, params=params, stream=True, headers=headers)
        self._raise_for_status(res)
        encoded_stat = res.headers.get('x-docker-container-path-stat')
        return (
            self._stream_raw_result(res, chunk_size, False),
            utils.decode_json_header(encoded_stat) if encoded_stat else None
        )

    @utils.check_resource('container')
    def inspect_container(self, container):
        """
        Identical to the `docker inspect` command, but only for containers.

        Args:
            container (str): The container to inspect

        Returns:
            (dict): Similar to the output of `docker inspect`, but as a
            single dict

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self._result(
            self._get(self._url("/containers/{0}/json", container)), True
        )

    @utils.check_resource('container')
    def kill(self, container, signal=None):
        """
        Kill a container or send a signal to a container.

        Args:
            container (str): The container to kill
            signal (str or int): The signal to send. Defaults to ``SIGKILL``

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        url = self._url("/containers/{0}/kill", container)
        params = {}
        if signal is not None:
            if not isinstance(signal, str):
                signal = int(signal)
            params['signal'] = signal
        res = self._post(url, params=params)

        self._raise_for_status(res)

    @utils.check_resource('container')
    def logs(self, container, stdout=True, stderr=True, stream=False,
             timestamps=False, tail='all', since=None, follow=None,
             until=None):
        """
        Get logs from a container. Similar to the ``docker logs`` command.

        The ``stream`` parameter makes the ``logs`` function return a blocking
        generator you can iterate over to retrieve log output as it happens.

        Args:
            container (str): The container to get logs from
            stdout (bool): Get ``STDOUT``. Default ``True``
            stderr (bool): Get ``STDERR``. Default ``True``
            stream (bool): Stream the response. Default ``False``
            timestamps (bool): Show timestamps. Default ``False``
            tail (str or int): Output specified number of lines at the end of
                logs. Either an integer of number of lines or the string
                ``all``. Default ``all``
            since (datetime, int, or float): Show logs since a given datetime,
                integer epoch (in seconds) or float (in fractional seconds)
            follow (bool): Follow log output. Default ``False``
            until (datetime, int, or float): Show logs that occurred before
                the given datetime, integer epoch (in seconds), or
                float (in fractional seconds)

        Returns:
            (generator or str)

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        if follow is None:
            follow = stream
        params = {'stderr': stderr and 1 or 0,
                  'stdout': stdout and 1 or 0,
                  'timestamps': timestamps and 1 or 0,
                  'follow': follow and 1 or 0,
                  }
        if tail != 'all' and (not isinstance(tail, int) or tail < 0):
            tail = 'all'
        params['tail'] = tail

        if since is not None:
            if isinstance(since, datetime):
                params['since'] = utils.datetime_to_timestamp(since)
            elif (isinstance(since, int) and since > 0):
                params['since'] = since
            elif (isinstance(since, float) and since > 0.0):
                params['since'] = since
            else:
                raise errors.InvalidArgument(
                    'since value should be datetime or positive int/float, '
                    'not {}'.format(type(since))
                )

        if until is not None:
            if utils.version_lt(self._version, '1.35'):
                raise errors.InvalidVersion(
                    'until is not supported for API version < 1.35'
                )
            if isinstance(until, datetime):
                params['until'] = utils.datetime_to_timestamp(until)
            elif (isinstance(until, int) and until > 0):
                params['until'] = until
            elif (isinstance(until, float) and until > 0.0):
                params['until'] = until
            else:
                raise errors.InvalidArgument(
                    'until value should be datetime or positive int/float, '
                    'not {}'.format(type(until))
                )

        url = self._url("/containers/{0}/logs", container)
        res = self._get(url, params=params, stream=stream)
        output = self._get_result(container, stream, res)

        if stream:
            return CancellableStream(output, res)
        else:
            return output

    @utils.check_resource('container')
    def pause(self, container):
        """
        Pauses all processes within a container.

        Args:
            container (str): The container to pause

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        url = self._url('/containers/{0}/pause', container)
        res = self._post(url)
        self._raise_for_status(res)

    @utils.check_resource('container')
    def port(self, container, private_port):
        """
        Lookup the public-facing port that is NAT-ed to ``private_port``.
        Identical to the ``docker port`` command.

        Args:
            container (str): The container to look up
            private_port (int): The private port to inspect

        Returns:
            (list of dict): The mapping for the host ports

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Example:
            .. code-block:: bash

                $ docker run -d -p 80:80 ubuntu:14.04 /bin/sleep 30
                7174d6347063a83f412fad6124c99cffd25ffe1a0807eb4b7f9cec76ac8cb43b

            .. code-block:: python

                >>> client.api.port('7174d6347063', 80)
                [{'HostIp': '0.0.0.0', 'HostPort': '80'}]
        """
        res = self._get(self._url("/containers/{0}/json", container))
        self._raise_for_status(res)
        json_ = res.json()
        private_port = str(private_port)
        h_ports = None

        # Port settings is None when the container is running with
        # network_mode=host.
        port_settings = json_.get('NetworkSettings', {}).get('Ports')
        if port_settings is None:
            return None

        if '/' in private_port:
            return port_settings.get(private_port)

        for protocol in ['tcp', 'udp', 'sctp']:
            h_ports = port_settings.get(private_port + '/' + protocol)
            if h_ports:
                break

        return h_ports

    @utils.check_resource('container')
    def put_archive(self, container, path, data):
        """
        Insert a file or folder in an existing container using a tar archive as
        source.

        Args:
            container (str): The container where the file(s) will be extracted
            path (str): Path inside the container where the file(s) will be
                extracted. Must exist.
            data (bytes): tar data to be extracted

        Returns:
            (bool): True if the call succeeds.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        params = {'path': path}
        url = self._url('/containers/{0}/archive', container)
        res = self._put(url, params=params, data=data)
        self._raise_for_status(res)
        return res.status_code == 200

    @utils.minimum_version('1.25')
    def prune_containers(self, filters=None):
        """
        Delete stopped containers

        Args:
            filters (dict): Filters to process on the prune list.

        Returns:
            (dict): A dict containing a list of deleted container IDs and
                the amount of disk space reclaimed in bytes.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        params = {}
        if filters:
            params['filters'] = utils.convert_filters(filters)
        url = self._url('/containers/prune')
        return self._result(self._post(url, params=params), True)

    @utils.check_resource('container')
    def remove_container(self, container, v=False, link=False, force=False):
        """
        Remove a container. Similar to the ``docker rm`` command.

        Args:
            container (str): The container to remove
            v (bool): Remove the volumes associated with the container
            link (bool): Remove the specified link and not the underlying
                container
            force (bool): Force the removal of a running container (uses
                ``SIGKILL``)

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        params = {'v': v, 'link': link, 'force': force}
        res = self._delete(
            self._url("/containers/{0}", container), params=params
        )
        self._raise_for_status(res)

    @utils.check_resource('container')
    def rename(self, container, name):
        """
        Rename a container. Similar to the ``docker rename`` command.

        Args:
            container (str): ID of the container to rename
            name (str): New name for the container

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        url = self._url("/containers/{0}/rename", container)
        params = {'name': name}
        res = self._post(url, params=params)
        self._raise_for_status(res)

    @utils.check_resource('container')
    def resize(self, container, height, width):
        """
        Resize the tty session.

        Args:
            container (str or dict): The container to resize
            height (int): Height of tty session
            width (int): Width of tty session

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        params = {'h': height, 'w': width}
        url = self._url("/containers/{0}/resize", container)
        res = self._post(url, params=params)
        self._raise_for_status(res)

    @utils.check_resource('container')
    def restart(self, container, timeout=10):
        """
        Restart a container. Similar to the ``docker restart`` command.

        Args:
            container (str or dict): The container to restart. If a dict, the
                ``Id`` key is used.
            timeout (int): Number of seconds to try to stop for before killing
                the container. Once killed it will then be restarted. Default
                is 10 seconds.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        params = {'t': timeout}
        url = self._url("/containers/{0}/restart", container)
        conn_timeout = self.timeout
        if conn_timeout is not None:
            conn_timeout += timeout
        res = self._post(url, params=params, timeout=conn_timeout)
        self._raise_for_status(res)

    @utils.check_resource('container')
    def start(self, container, *args, **kwargs):
        """
        Start a container. Similar to the ``docker start`` command, but
        doesn't support attach options.

        **Deprecation warning:** Passing configuration options in ``start`` is
        no longer supported. Users are expected to provide host config options
        in the ``host_config`` parameter of
        :py:meth:`~ContainerApiMixin.create_container`.


        Args:
            container (str): The container to start

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
            :py:class:`docker.errors.DeprecatedMethod`
                If any argument besides ``container`` are provided.

        Example:

            >>> container = client.api.create_container(
            ...     image='busybox:latest',
            ...     command='/bin/sleep 30')
            >>> client.api.start(container=container.get('Id'))
        """
        if args or kwargs:
            raise errors.DeprecatedMethod(
                'Providing configuration in the start() method is no longer '
                'supported. Use the host_config param in create_container '
                'instead.'
            )
        url = self._url("/containers/{0}/start", container)
        res = self._post(url)
        self._raise_for_status(res)

    @utils.check_resource('container')
    def stats(self, container, decode=None, stream=True):
        """
        Stream statistics for a specific container. Similar to the
        ``docker stats`` command.

        Args:
            container (str): The container to stream statistics from
            decode (bool): If set to true, stream will be decoded into dicts
                on the fly. Only applicable if ``stream`` is True.
                False by default.
            stream (bool): If set to false, only the current stats will be
                returned instead of a stream. True by default.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        """
        url = self._url("/containers/{0}/stats", container)
        if stream:
            return self._stream_helper(self._get(url, stream=True),
                                       decode=decode)
        else:
            if decode:
                raise errors.InvalidArgument(
                    "decode is only available in conjunction with stream=True"
                )
            return self._result(self._get(url, params={'stream': False}),
                                json=True)

    @utils.check_resource('container')
    def stop(self, container, timeout=None):
        """
        Stops a container. Similar to the ``docker stop`` command.

        Args:
            container (str): The container to stop
            timeout (int): Timeout in seconds to wait for the container to
                stop before sending a ``SIGKILL``. If None, then the
                StopTimeout value of the container will be used.
                Default: None

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        if timeout is None:
            params = {}
            timeout = 10
        else:
            params = {'t': timeout}
        url = self._url("/containers/{0}/stop", container)
        conn_timeout = self.timeout
        if conn_timeout is not None:
            conn_timeout += timeout
        res = self._post(url, params=params, timeout=conn_timeout)
        self._raise_for_status(res)

    @utils.check_resource('container')
    def top(self, container, ps_args=None):
        """
        Display the running processes of a container.

        Args:
            container (str): The container to inspect
            ps_args (str): An optional arguments passed to ps (e.g. ``aux``)

        Returns:
            (str): The output of the top

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        u = self._url("/containers/{0}/top", container)
        params = {}
        if ps_args is not None:
            params['ps_args'] = ps_args
        return self._result(self._get(u, params=params), True)

    @utils.check_resource('container')
    def unpause(self, container):
        """
        Unpause all processes within a container.

        Args:
            container (str): The container to unpause
        """
        url = self._url('/containers/{0}/unpause', container)
        res = self._post(url)
        self._raise_for_status(res)

    @utils.minimum_version('1.22')
    @utils.check_resource('container')
    def update_container(
        self, container, blkio_weight=None, cpu_period=None, cpu_quota=None,
        cpu_shares=None, cpuset_cpus=None, cpuset_mems=None, mem_limit=None,
        mem_reservation=None, memswap_limit=None, kernel_memory=None,
        restart_policy=None
    ):
        """
        Update resource configs of one or more containers.

        Args:
            container (str): The container to inspect
            blkio_weight (int): Block IO (relative weight), between 10 and 1000
            cpu_period (int): Limit CPU CFS (Completely Fair Scheduler) period
            cpu_quota (int): Limit CPU CFS (Completely Fair Scheduler) quota
            cpu_shares (int): CPU shares (relative weight)
            cpuset_cpus (str): CPUs in which to allow execution
            cpuset_mems (str): MEMs in which to allow execution
            mem_limit (float or str): Memory limit
            mem_reservation (float or str): Memory soft limit
            memswap_limit (int or str): Total memory (memory + swap), -1 to
                disable swap
            kernel_memory (int or str): Kernel memory limit
            restart_policy (dict): Restart policy dictionary

        Returns:
            (dict): Dictionary containing a ``Warnings`` key.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        url = self._url('/containers/{0}/update', container)
        data = {}
        if blkio_weight:
            data['BlkioWeight'] = blkio_weight
        if cpu_period:
            data['CpuPeriod'] = cpu_period
        if cpu_shares:
            data['CpuShares'] = cpu_shares
        if cpu_quota:
            data['CpuQuota'] = cpu_quota
        if cpuset_cpus:
            data['CpusetCpus'] = cpuset_cpus
        if cpuset_mems:
            data['CpusetMems'] = cpuset_mems
        if mem_limit:
            data['Memory'] = utils.parse_bytes(mem_limit)
        if mem_reservation:
            data['MemoryReservation'] = utils.parse_bytes(mem_reservation)
        if memswap_limit:
            data['MemorySwap'] = utils.parse_bytes(memswap_limit)
        if kernel_memory:
            data['KernelMemory'] = utils.parse_bytes(kernel_memory)
        if restart_policy:
            if utils.version_lt(self._version, '1.23'):
                raise errors.InvalidVersion(
                    'restart policy update is not supported '
                    'for API version < 1.23'
                )
            data['RestartPolicy'] = restart_policy

        res = self._post_json(url, data=data)
        return self._result(res, True)

    @utils.check_resource('container')
    def wait(self, container, timeout=None, condition=None):
        """
        Block until a container stops, then return its exit code. Similar to
        the ``docker wait`` command.

        Args:
            container (str or dict): The container to wait on. If a dict, the
                ``Id`` key is used.
            timeout (int): Request timeout
            condition (str): Wait until a container state reaches the given
                condition, either ``not-running`` (default), ``next-exit``,
                or ``removed``

        Returns:
            (dict): The API's response as a Python dictionary, including
                the container's exit code under the ``StatusCode`` attribute.

        Raises:
            :py:class:`requests.exceptions.ReadTimeout`
                If the timeout is exceeded.
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        url = self._url("/containers/{0}/wait", container)
        params = {}
        if condition is not None:
            if utils.version_lt(self._version, '1.30'):
                raise errors.InvalidVersion(
                    'wait condition is not supported for API version < 1.30'
                )
            params['condition'] = condition

        res = self._post(url, timeout=timeout, params=params)
        return self._result(res, True)
