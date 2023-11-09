import copy
import ntpath
from collections import namedtuple

from ..api import APIClient
from ..constants import DEFAULT_DATA_CHUNK_SIZE
from ..errors import (
    ContainerError, DockerException, ImageNotFound,
    NotFound, create_unexpected_kwargs_error
)
from ..types import HostConfig
from ..utils import version_gte
from .images import Image
from .resource import Collection, Model


class Container(Model):
    """ Local representation of a container object. Detailed configuration may
        be accessed through the :py:attr:`attrs` attribute. Note that local
        attributes are cached; users may call :py:meth:`reload` to
        query the Docker daemon for the current properties, causing
        :py:attr:`attrs` to be refreshed.
    """
    @property
    def name(self):
        """
        The name of the container.
        """
        if self.attrs.get('Name') is not None:
            return self.attrs['Name'].lstrip('/')

    @property
    def image(self):
        """
        The image of the container.
        """
        image_id = self.attrs.get('ImageID', self.attrs['Image'])
        if image_id is None:
            return None
        return self.client.images.get(image_id.split(':')[1])

    @property
    def labels(self):
        """
        The labels of a container as dictionary.
        """
        try:
            result = self.attrs['Config'].get('Labels')
            return result or {}
        except KeyError:
            raise DockerException(
                'Label data is not available for sparse objects. Call reload()'
                ' to retrieve all information'
            )

    @property
    def status(self):
        """
        The status of the container. For example, ``running``, or ``exited``.
        """
        if isinstance(self.attrs['State'], dict):
            return self.attrs['State']['Status']
        return self.attrs['State']

    @property
    def ports(self):
        """
        The ports that the container exposes as a dictionary.
        """
        return self.attrs.get('NetworkSettings', {}).get('Ports', {})

    def attach(self, **kwargs):
        """
        Attach to this container.

        :py:meth:`logs` is a wrapper around this method, which you can
        use instead if you want to fetch/stream container output without first
        retrieving the entire backlog.

        Args:
            stdout (bool): Include stdout.
            stderr (bool): Include stderr.
            stream (bool): Return container output progressively as an iterator
                of strings, rather than a single string.
            logs (bool): Include the container's previous output.

        Returns:
            By default, the container's output as a single string.

            If ``stream=True``, an iterator of output strings.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.attach(self.id, **kwargs)

    def attach_socket(self, **kwargs):
        """
        Like :py:meth:`attach`, but returns the underlying socket-like object
        for the HTTP request.

        Args:
            params (dict): Dictionary of request parameters (e.g. ``stdout``,
                ``stderr``, ``stream``).
            ws (bool): Use websockets instead of raw HTTP.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.attach_socket(self.id, **kwargs)

    def commit(self, repository=None, tag=None, **kwargs):
        """
        Commit a container to an image. Similar to the ``docker commit``
        command.

        Args:
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

        resp = self.client.api.commit(self.id, repository=repository, tag=tag,
                                      **kwargs)
        return self.client.images.get(resp['Id'])

    def diff(self):
        """
        Inspect changes on a container's filesystem.

        Returns:
            (str)

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.diff(self.id)

    def exec_run(self, cmd, stdout=True, stderr=True, stdin=False, tty=False,
                 privileged=False, user='', detach=False, stream=False,
                 socket=False, environment=None, workdir=None, demux=False):
        """
        Run a command inside this container. Similar to
        ``docker exec``.

        Args:
            cmd (str or list): Command to be executed
            stdout (bool): Attach to stdout. Default: ``True``
            stderr (bool): Attach to stderr. Default: ``True``
            stdin (bool): Attach to stdin. Default: ``False``
            tty (bool): Allocate a pseudo-TTY. Default: False
            privileged (bool): Run as privileged.
            user (str): User to execute command as. Default: root
            detach (bool): If true, detach from the exec command.
                Default: False
            stream (bool): Stream response data. Default: False
            socket (bool): Return the connection socket to allow custom
                read/write operations. Default: False
            environment (dict or list): A dictionary or a list of strings in
                the following format ``["PASSWORD=xxx"]`` or
                ``{"PASSWORD": "xxx"}``.
            workdir (str): Path to working directory for this exec session
            demux (bool): Return stdout and stderr separately

        Returns:
            (ExecResult): A tuple of (exit_code, output)
                exit_code: (int):
                    Exit code for the executed command or ``None`` if
                    either ``stream`` or ``socket`` is ``True``.
                output: (generator, bytes, or tuple):
                    If ``stream=True``, a generator yielding response chunks.
                    If ``socket=True``, a socket object for the connection.
                    If ``demux=True``, a tuple of two bytes: stdout and stderr.
                    A bytestring containing response data otherwise.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        resp = self.client.api.exec_create(
            self.id, cmd, stdout=stdout, stderr=stderr, stdin=stdin, tty=tty,
            privileged=privileged, user=user, environment=environment,
            workdir=workdir,
        )
        exec_output = self.client.api.exec_start(
            resp['Id'], detach=detach, tty=tty, stream=stream, socket=socket,
            demux=demux
        )
        if socket or stream:
            return ExecResult(None, exec_output)

        return ExecResult(
            self.client.api.exec_inspect(resp['Id'])['ExitCode'],
            exec_output
        )

    def export(self, chunk_size=DEFAULT_DATA_CHUNK_SIZE):
        """
        Export the contents of the container's filesystem as a tar archive.

        Args:
            chunk_size (int): The number of bytes returned by each iteration
                of the generator. If ``None``, data will be streamed as it is
                received. Default: 2 MB

        Returns:
            (str): The filesystem tar archive

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.export(self.id, chunk_size)

    def get_archive(self, path, chunk_size=DEFAULT_DATA_CHUNK_SIZE,
                    encode_stream=False):
        """
        Retrieve a file or folder from the container in the form of a tar
        archive.

        Args:
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

            >>> f = open('./sh_bin.tar', 'wb')
            >>> bits, stat = container.get_archive('/bin/sh')
            >>> print(stat)
            {'name': 'sh', 'size': 1075464, 'mode': 493,
             'mtime': '2018-10-01T15:37:48-07:00', 'linkTarget': ''}
            >>> for chunk in bits:
            ...    f.write(chunk)
            >>> f.close()
        """
        return self.client.api.get_archive(self.id, path,
                                           chunk_size, encode_stream)

    def kill(self, signal=None):
        """
        Kill or send a signal to the container.

        Args:
            signal (str or int): The signal to send. Defaults to ``SIGKILL``

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """

        return self.client.api.kill(self.id, signal=signal)

    def logs(self, **kwargs):
        """
        Get logs from this container. Similar to the ``docker logs`` command.

        The ``stream`` parameter makes the ``logs`` function return a blocking
        generator you can iterate over to retrieve log output as it happens.

        Args:
            stdout (bool): Get ``STDOUT``. Default ``True``
            stderr (bool): Get ``STDERR``. Default ``True``
            stream (bool): Stream the response. Default ``False``
            timestamps (bool): Show timestamps. Default ``False``
            tail (str or int): Output specified number of lines at the end of
                logs. Either an integer of number of lines or the string
                ``all``. Default ``all``
            since (datetime, int, or float): Show logs since a given datetime,
                integer epoch (in seconds) or float (in nanoseconds)
            follow (bool): Follow log output. Default ``False``
            until (datetime, int, or float): Show logs that occurred before
                the given datetime, integer epoch (in seconds), or
                float (in nanoseconds)

        Returns:
            (generator or str): Logs from the container.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.logs(self.id, **kwargs)

    def pause(self):
        """
        Pauses all processes within this container.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.pause(self.id)

    def put_archive(self, path, data):
        """
        Insert a file or folder in this container using a tar archive as
        source.

        Args:
            path (str): Path inside the container where the file(s) will be
                extracted. Must exist.
            data (bytes): tar data to be extracted

        Returns:
            (bool): True if the call succeeds.

        Raises:
            :py:class:`~docker.errors.APIError` If an error occurs.
        """
        return self.client.api.put_archive(self.id, path, data)

    def remove(self, **kwargs):
        """
        Remove this container. Similar to the ``docker rm`` command.

        Args:
            v (bool): Remove the volumes associated with the container
            link (bool): Remove the specified link and not the underlying
                container
            force (bool): Force the removal of a running container (uses
                ``SIGKILL``)

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.remove_container(self.id, **kwargs)

    def rename(self, name):
        """
        Rename this container. Similar to the ``docker rename`` command.

        Args:
            name (str): New name for the container

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.rename(self.id, name)

    def resize(self, height, width):
        """
        Resize the tty session.

        Args:
            height (int): Height of tty session
            width (int): Width of tty session

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.resize(self.id, height, width)

    def restart(self, **kwargs):
        """
        Restart this container. Similar to the ``docker restart`` command.

        Args:
            timeout (int): Number of seconds to try to stop for before killing
                the container. Once killed it will then be restarted. Default
                is 10 seconds.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.restart(self.id, **kwargs)

    def start(self, **kwargs):
        """
        Start this container. Similar to the ``docker start`` command, but
        doesn't support attach options.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.start(self.id, **kwargs)

    def stats(self, **kwargs):
        """
        Stream statistics for this container. Similar to the
        ``docker stats`` command.

        Args:
            decode (bool): If set to true, stream will be decoded into dicts
                on the fly. Only applicable if ``stream`` is True.
                False by default.
            stream (bool): If set to false, only the current stats will be
                returned instead of a stream. True by default.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.stats(self.id, **kwargs)

    def stop(self, **kwargs):
        """
        Stops a container. Similar to the ``docker stop`` command.

        Args:
            timeout (int): Timeout in seconds to wait for the container to
                stop before sending a ``SIGKILL``. Default: 10

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.stop(self.id, **kwargs)

    def top(self, **kwargs):
        """
        Display the running processes of the container.

        Args:
            ps_args (str): An optional arguments passed to ps (e.g. ``aux``)

        Returns:
            (str): The output of the top

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.top(self.id, **kwargs)

    def unpause(self):
        """
        Unpause all processes within the container.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.client.api.unpause(self.id)

    def update(self, **kwargs):
        """
        Update resource configuration of the containers.

        Args:
            blkio_weight (int): Block IO (relative weight), between 10 and 1000
            cpu_period (int): Limit CPU CFS (Completely Fair Scheduler) period
            cpu_quota (int): Limit CPU CFS (Completely Fair Scheduler) quota
            cpu_shares (int): CPU shares (relative weight)
            cpuset_cpus (str): CPUs in which to allow execution
            cpuset_mems (str): MEMs in which to allow execution
            mem_limit (int or str): Memory limit
            mem_reservation (int or str): Memory soft limit
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
        return self.client.api.update_container(self.id, **kwargs)

    def wait(self, **kwargs):
        """
        Block until the container stops, then return its exit code. Similar to
        the ``docker wait`` command.

        Args:
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
        return self.client.api.wait(self.id, **kwargs)


class ContainerCollection(Collection):
    model = Container

    def run(self, image, command=None, stdout=True, stderr=False,
            remove=False, **kwargs):
        """
        Run a container. By default, it will wait for the container to finish
        and return its logs, similar to ``docker run``.

        If the ``detach`` argument is ``True``, it will start the container
        and immediately return a :py:class:`Container` object, similar to
        ``docker run -d``.

        Example:
            Run a container and get its output:

            >>> import docker
            >>> client = docker.from_env()
            >>> client.containers.run('alpine', 'echo hello world')
            b'hello world\\n'

            Run a container and detach:

            >>> container = client.containers.run('bfirsh/reticulate-splines',
                                                  detach=True)
            >>> container.logs()
            'Reticulating spline 1...\\nReticulating spline 2...\\n'

        Args:
            image (str): The image to run.
            command (str or list): The command to run in the container.
            auto_remove (bool): enable auto-removal of the container on daemon
                side when the container's process exits.
            blkio_weight_device: Block IO weight (relative device weight) in
                the form of: ``[{"Path": "device_path", "Weight": weight}]``.
            blkio_weight: Block IO weight (relative weight), accepts a weight
                value between 10 and 1000.
            cap_add (list of str): Add kernel capabilities. For example,
                ``["SYS_ADMIN", "MKNOD"]``.
            cap_drop (list of str): Drop kernel capabilities.
            cgroup_parent (str): Override the default parent cgroup.
            cgroupns (str): Override the default cgroup namespace mode for the
                container. One of:
                - ``private`` the container runs in its own private cgroup
                  namespace.
                - ``host`` use the host system's cgroup namespace.
            cpu_count (int): Number of usable CPUs (Windows only).
            cpu_percent (int): Usable percentage of the available CPUs
                (Windows only).
            cpu_period (int): The length of a CPU period in microseconds.
            cpu_quota (int): Microseconds of CPU time that the container can
                get in a CPU period.
            cpu_rt_period (int): Limit CPU real-time period in microseconds.
            cpu_rt_runtime (int): Limit CPU real-time runtime in microseconds.
            cpu_shares (int): CPU shares (relative weight).
            cpuset_cpus (str): CPUs in which to allow execution (``0-3``,
                ``0,1``).
            cpuset_mems (str): Memory nodes (MEMs) in which to allow execution
                (``0-3``, ``0,1``). Only effective on NUMA systems.
            detach (bool): Run container in the background and return a
                :py:class:`Container` object.
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
                container's ``resolv.conf`` file.
            dns_search (:py:class:`list`): DNS search domains.
            domainname (str or list): Set custom DNS search domains.
            entrypoint (str or list): The entrypoint for the container.
            environment (dict or list): Environment variables to set inside
                the container, as a dictionary or a list of strings in the
                format ``["SOMEVARIABLE=xxx"]``.
            extra_hosts (dict): Additional hostnames to resolve inside the
                container, as a mapping of hostname to IP address.
            group_add (:py:class:`list`): List of additional group names and/or
                IDs that the container process will run as.
            healthcheck (dict): Specify a test to perform to check that the
                container is healthy. The dict takes the following keys:

                - test (:py:class:`list` or str): Test to perform to determine
                    container health. Possible values:

                    - Empty list: Inherit healthcheck from parent image
                    - ``["NONE"]``: Disable healthcheck
                    - ``["CMD", args...]``: exec arguments directly.
                    - ``["CMD-SHELL", command]``: Run command in the system's
                      default shell.

                    If a string is provided, it will be used as a ``CMD-SHELL``
                    command.
                - interval (int): The time to wait between checks in
                  nanoseconds. It should be 0 or at least 1000000 (1 ms).
                - timeout (int): The time to wait before considering the check
                  to have hung. It should be 0 or at least 1000000 (1 ms).
                - retries (int): The number of consecutive failures needed to
                    consider a container as unhealthy.
                - start_period (int): Start period for the container to
                    initialize before starting health-retries countdown in
                    nanoseconds. It should be 0 or at least 1000000 (1 ms).
            hostname (str): Optional hostname for the container.
            init (bool): Run an init inside the container that forwards
                signals and reaps processes
            init_path (str): Path to the docker-init binary
            ipc_mode (str): Set the IPC mode for the container.
            isolation (str): Isolation technology to use. Default: `None`.
            kernel_memory (int or str): Kernel memory limit
            labels (dict or list): A dictionary of name-value labels (e.g.
                ``{"label1": "value1", "label2": "value2"}``) or a list of
                names of labels to set with empty values (e.g.
                ``["label1", "label2"]``)
            links (dict): Mapping of links using the
                ``{'container': 'alias'}`` format. The alias is optional.
                Containers declared in this dict will be linked to the new
                container using the provided alias. Default: ``None``.
            log_config (LogConfig): Logging configuration.
            lxc_conf (dict): LXC config.
            mac_address (str): MAC address to assign to the container.
            mem_limit (int or str): Memory limit. Accepts float values
                (which represent the memory limit of the created container in
                bytes) or a string with a units identification char
                (``100000b``, ``1000k``, ``128m``, ``1g``). If a string is
                specified without a units character, bytes are assumed as an
                intended unit.
            mem_reservation (int or str): Memory soft limit.
            mem_swappiness (int): Tune a container's memory swappiness
                behavior. Accepts number between 0 and 100.
            memswap_limit (str or int): Maximum amount of memory + swap a
                container is allowed to consume.
            mounts (:py:class:`list`): Specification for mounts to be added to
                the container. More powerful alternative to ``volumes``. Each
                item in the list is expected to be a
                :py:class:`docker.types.Mount` object.
            name (str): The name for this container.
            nano_cpus (int):  CPU quota in units of 1e-9 CPUs.
            network (str): Name of the network this container will be connected
                to at creation time. You can connect to additional networks
                using :py:meth:`Network.connect`. Incompatible with
                ``network_mode``.
            network_disabled (bool): Disable networking.
            network_mode (str): One of:

                - ``bridge`` Create a new network stack for the container on
                  the bridge network.
                - ``none`` No networking for this container.
                - ``container:<name|id>`` Reuse another container's network
                  stack.
                - ``host`` Use the host network stack.
                  This mode is incompatible with ``ports``.

                Incompatible with ``network``.
            oom_kill_disable (bool): Whether to disable OOM killer.
            oom_score_adj (int): An integer value containing the score given
                to the container in order to tune OOM killer preferences.
            pid_mode (str): If set to ``host``, use the host PID namespace
                inside the container.
            pids_limit (int): Tune a container's pids limit. Set ``-1`` for
                unlimited.
            platform (str): Platform in the format ``os[/arch[/variant]]``.
                Only used if the method needs to pull the requested image.
            ports (dict): Ports to bind inside the container.

                The keys of the dictionary are the ports to bind inside the
                container, either as an integer or a string in the form
                ``port/protocol``, where the protocol is either ``tcp``,
                ``udp``, or ``sctp``.

                The values of the dictionary are the corresponding ports to
                open on the host, which can be either:

                - The port number, as an integer. For example,
                  ``{'2222/tcp': 3333}`` will expose port 2222 inside the
                  container as port 3333 on the host.
                - ``None``, to assign a random host port. For example,
                  ``{'2222/tcp': None}``.
                - A tuple of ``(address, port)`` if you want to specify the
                  host interface. For example,
                  ``{'1111/tcp': ('127.0.0.1', 1111)}``.
                - A list of integers, if you want to bind multiple host ports
                  to a single container port. For example,
                  ``{'1111/tcp': [1234, 4567]}``.

                Incompatible with ``host`` network mode.
            privileged (bool): Give extended privileges to this container.
            publish_all_ports (bool): Publish all ports to the host.
            read_only (bool): Mount the container's root filesystem as read
                only.
            remove (bool): Remove the container when it has finished running.
                Default: ``False``.
            restart_policy (dict): Restart the container when it exits.
                Configured as a dictionary with keys:

                - ``Name`` One of ``on-failure``, or ``always``.
                - ``MaximumRetryCount`` Number of times to restart the
                  container on failure.

                For example:
                ``{"Name": "on-failure", "MaximumRetryCount": 5}``

            runtime (str): Runtime to use with this container.
            security_opt (:py:class:`list`): A list of string values to
                customize labels for MLS systems, such as SELinux.
            shm_size (str or int): Size of /dev/shm (e.g. ``1G``).
            stdin_open (bool): Keep ``STDIN`` open even if not attached.
            stdout (bool): Return logs from ``STDOUT`` when ``detach=False``.
                Default: ``True``.
            stderr (bool): Return logs from ``STDERR`` when ``detach=False``.
                Default: ``False``.
            stop_signal (str): The stop signal to use to stop the container
                (e.g. ``SIGINT``).
            storage_opt (dict): Storage driver options per container as a
                key-value mapping.
            stream (bool): If true and ``detach`` is false, return a log
                generator instead of a string. Ignored if ``detach`` is true.
                Default: ``False``.
            sysctls (dict): Kernel parameters to set in the container.
            tmpfs (dict): Temporary filesystems to mount, as a dictionary
                mapping a path inside the container to options for that path.

                For example:

                .. code-block:: python

                    {
                        '/mnt/vol2': '',
                        '/mnt/vol1': 'size=3G,uid=1000'
                    }

            tty (bool): Allocate a pseudo-TTY.
            ulimits (:py:class:`list`): Ulimits to set inside the container,
                as a list of :py:class:`docker.types.Ulimit` instances.
            use_config_proxy (bool): If ``True``, and if the docker client
                configuration file (``~/.docker/config.json`` by default)
                contains a proxy configuration, the corresponding environment
                variables will be set in the container being built.
            user (str or int): Username or UID to run commands as inside the
                container.
            userns_mode (str): Sets the user namespace mode for the container
                when user namespace remapping option is enabled. Supported
                values are: ``host``
            uts_mode (str): Sets the UTS namespace mode for the container.
                Supported values are: ``host``
            version (str): The version of the API to use. Set to ``auto`` to
                automatically detect the server's version. Default: ``1.35``
            volume_driver (str): The name of a volume driver/plugin.
            volumes (dict or list): A dictionary to configure volumes mounted
                inside the container. The key is either the host path or a
                volume name, and the value is a dictionary with the keys:

                - ``bind`` The path to mount the volume inside the container
                - ``mode`` Either ``rw`` to mount the volume read/write, or
                  ``ro`` to mount it read-only.

                For example:

                .. code-block:: python

                    {'/home/user1/': {'bind': '/mnt/vol2', 'mode': 'rw'},
                     '/var/www': {'bind': '/mnt/vol1', 'mode': 'ro'}}

                Or a list of strings which each one of its elements specifies a
                mount volume.

                For example:

                .. code-block:: python

                    ['/home/user1/:/mnt/vol2','/var/www:/mnt/vol1']

            volumes_from (:py:class:`list`): List of container names or IDs to
                get volumes from.
            working_dir (str): Path to the working directory.

        Returns:
            The container logs, either ``STDOUT``, ``STDERR``, or both,
            depending on the value of the ``stdout`` and ``stderr`` arguments.

            ``STDOUT`` and ``STDERR`` may be read only if either ``json-file``
            or ``journald`` logging driver used. Thus, if you are using none of
            these drivers, a ``None`` object is returned instead. See the
            `Engine API documentation
            <https://docs.docker.com/engine/api/v1.30/#operation/ContainerLogs/>`_
            for full details.

            If ``detach`` is ``True``, a :py:class:`Container` object is
            returned instead.

        Raises:
            :py:class:`docker.errors.ContainerError`
                If the container exits with a non-zero exit code and
                ``detach`` is ``False``.
            :py:class:`docker.errors.ImageNotFound`
                If the specified image does not exist.
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        if isinstance(image, Image):
            image = image.id
        stream = kwargs.pop('stream', False)
        detach = kwargs.pop('detach', False)
        platform = kwargs.get('platform', None)

        if detach and remove:
            if version_gte(self.client.api._version, '1.25'):
                kwargs["auto_remove"] = True
            else:
                raise RuntimeError("The options 'detach' and 'remove' cannot "
                                   "be used together in api versions < 1.25.")

        if kwargs.get('network') and kwargs.get('network_mode'):
            raise RuntimeError(
                'The options "network" and "network_mode" can not be used '
                'together.'
            )

        try:
            container = self.create(image=image, command=command,
                                    detach=detach, **kwargs)
        except ImageNotFound:
            self.client.images.pull(image, platform=platform)
            container = self.create(image=image, command=command,
                                    detach=detach, **kwargs)

        container.start()

        if detach:
            return container

        logging_driver = container.attrs['HostConfig']['LogConfig']['Type']

        out = None
        if logging_driver == 'json-file' or logging_driver == 'journald':
            out = container.logs(
                stdout=stdout, stderr=stderr, stream=True, follow=True
            )

        exit_status = container.wait()['StatusCode']
        if exit_status != 0:
            out = None
            if not kwargs.get('auto_remove'):
                out = container.logs(stdout=False, stderr=True)

        if remove:
            container.remove()
        if exit_status != 0:
            raise ContainerError(
                container, exit_status, command, image, out
            )

        return out if stream or out is None else b''.join(
            [line for line in out]
        )

    def create(self, image, command=None, **kwargs):
        """
        Create a container without starting it. Similar to ``docker create``.

        Takes the same arguments as :py:meth:`run`, except for ``stdout``,
        ``stderr``, and ``remove``.

        Returns:
            A :py:class:`Container` object.

        Raises:
            :py:class:`docker.errors.ImageNotFound`
                If the specified image does not exist.
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        if isinstance(image, Image):
            image = image.id
        kwargs['image'] = image
        kwargs['command'] = command
        kwargs['version'] = self.client.api._version
        create_kwargs = _create_container_args(kwargs)
        resp = self.client.api.create_container(**create_kwargs)
        return self.get(resp['Id'])

    def get(self, container_id):
        """
        Get a container by name or ID.

        Args:
            container_id (str): Container name or ID.

        Returns:
            A :py:class:`Container` object.

        Raises:
            :py:class:`docker.errors.NotFound`
                If the container does not exist.
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        resp = self.client.api.inspect_container(container_id)
        return self.prepare_model(resp)

    def list(self, all=False, before=None, filters=None, limit=-1, since=None,
             sparse=False, ignore_removed=False):
        """
        List containers. Similar to the ``docker ps`` command.

        Args:
            all (bool): Show all containers. Only running containers are shown
                by default
            since (str): Show only containers created since Id or Name, include
                non-running ones
            before (str): Show only container created before Id or Name,
                include non-running ones
            limit (int): Show `limit` last created containers, include
                non-running ones
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

            sparse (bool): Do not inspect containers. Returns partial
                information, but guaranteed not to block. Use
                :py:meth:`Container.reload` on resulting objects to retrieve
                all attributes. Default: ``False``
            ignore_removed (bool): Ignore failures due to missing containers
                when attempting to inspect containers from the original list.
                Set to ``True`` if race conditions are likely. Has no effect
                if ``sparse=True``. Default: ``False``

        Returns:
            (list of :py:class:`Container`)

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        resp = self.client.api.containers(all=all, before=before,
                                          filters=filters, limit=limit,
                                          since=since)
        if sparse:
            return [self.prepare_model(r) for r in resp]
        else:
            containers = []
            for r in resp:
                try:
                    containers.append(self.get(r['Id']))
                # a container may have been removed while iterating
                except NotFound:
                    if not ignore_removed:
                        raise
            return containers

    def prune(self, filters=None):
        return self.client.api.prune_containers(filters=filters)
    prune.__doc__ = APIClient.prune_containers.__doc__


# kwargs to copy straight from run to create
RUN_CREATE_KWARGS = [
    'command',
    'detach',
    'domainname',
    'entrypoint',
    'environment',
    'healthcheck',
    'hostname',
    'image',
    'labels',
    'mac_address',
    'name',
    'network_disabled',
    'platform',
    'stdin_open',
    'stop_signal',
    'tty',
    'use_config_proxy',
    'user',
    'working_dir',
]

# kwargs to copy straight from run to host_config
RUN_HOST_CONFIG_KWARGS = [
    'auto_remove',
    'blkio_weight_device',
    'blkio_weight',
    'cap_add',
    'cap_drop',
    'cgroup_parent',
    'cgroupns',
    'cpu_count',
    'cpu_percent',
    'cpu_period',
    'cpu_quota',
    'cpu_shares',
    'cpuset_cpus',
    'cpuset_mems',
    'cpu_rt_period',
    'cpu_rt_runtime',
    'device_cgroup_rules',
    'device_read_bps',
    'device_read_iops',
    'device_write_bps',
    'device_write_iops',
    'devices',
    'device_requests',
    'dns_opt',
    'dns_search',
    'dns',
    'extra_hosts',
    'group_add',
    'init',
    'init_path',
    'ipc_mode',
    'isolation',
    'kernel_memory',
    'links',
    'log_config',
    'lxc_conf',
    'mem_limit',
    'mem_reservation',
    'mem_swappiness',
    'memswap_limit',
    'mounts',
    'nano_cpus',
    'network_mode',
    'oom_kill_disable',
    'oom_score_adj',
    'pid_mode',
    'pids_limit',
    'privileged',
    'publish_all_ports',
    'read_only',
    'restart_policy',
    'security_opt',
    'shm_size',
    'storage_opt',
    'sysctls',
    'tmpfs',
    'ulimits',
    'userns_mode',
    'uts_mode',
    'version',
    'volume_driver',
    'volumes_from',
    'runtime'
]


def _create_container_args(kwargs):
    """
    Convert arguments to create() to arguments to create_container().
    """
    # Copy over kwargs which can be copied directly
    create_kwargs = {}
    for key in copy.copy(kwargs):
        if key in RUN_CREATE_KWARGS:
            create_kwargs[key] = kwargs.pop(key)
    host_config_kwargs = {}
    for key in copy.copy(kwargs):
        if key in RUN_HOST_CONFIG_KWARGS:
            host_config_kwargs[key] = kwargs.pop(key)

    # Process kwargs which are split over both create and host_config
    ports = kwargs.pop('ports', {})
    if ports:
        host_config_kwargs['port_bindings'] = ports

    volumes = kwargs.pop('volumes', {})
    if volumes:
        host_config_kwargs['binds'] = volumes

    network = kwargs.pop('network', None)
    if network:
        create_kwargs['networking_config'] = {network: None}
        host_config_kwargs['network_mode'] = network

    # All kwargs should have been consumed by this point, so raise
    # error if any are left
    if kwargs:
        raise create_unexpected_kwargs_error('run', kwargs)

    create_kwargs['host_config'] = HostConfig(**host_config_kwargs)

    # Fill in any kwargs which need processing by create_host_config first
    port_bindings = create_kwargs['host_config'].get('PortBindings')
    if port_bindings:
        # sort to make consistent for tests
        create_kwargs['ports'] = [tuple(p.split('/', 1))
                                  for p in sorted(port_bindings.keys())]
    if volumes:
        if isinstance(volumes, dict):
            create_kwargs['volumes'] = [
                v.get('bind') for v in volumes.values()
            ]
        else:
            create_kwargs['volumes'] = [
                _host_volume_from_bind(v) for v in volumes
            ]
    return create_kwargs


def _host_volume_from_bind(bind):
    drive, rest = ntpath.splitdrive(bind)
    bits = rest.split(':', 1)
    if len(bits) == 1 or bits[1] in ('ro', 'rw'):
        return drive + bits[0]
    else:
        return bits[1].rstrip(':ro').rstrip(':rw')


ExecResult = namedtuple('ExecResult', 'exit_code,output')
""" A result of Container.exec_run with the properties ``exit_code`` and
    ``output``. """
