from .. import errors
from .. import utils


class ExecApiMixin:
    @utils.check_resource('container')
    def exec_create(self, container, cmd, stdout=True, stderr=True,
                    stdin=False, tty=False, privileged=False, user='',
                    environment=None, workdir=None, detach_keys=None):
        """
        Sets up an exec instance in a running container.

        Args:
            container (str): Target container where exec instance will be
                created
            cmd (str or list): Command to be executed
            stdout (bool): Attach to stdout. Default: ``True``
            stderr (bool): Attach to stderr. Default: ``True``
            stdin (bool): Attach to stdin. Default: ``False``
            tty (bool): Allocate a pseudo-TTY. Default: False
            privileged (bool): Run as privileged.
            user (str): User to execute command as. Default: root
            environment (dict or list): A dictionary or a list of strings in
                the following format ``["PASSWORD=xxx"]`` or
                ``{"PASSWORD": "xxx"}``.
            workdir (str): Path to working directory for this exec session
            detach_keys (str): Override the key sequence for detaching
                a container. Format is a single character `[a-Z]`
                or `ctrl-<value>` where `<value>` is one of:
                `a-z`, `@`, `^`, `[`, `,` or `_`.
                ~/.docker/config.json is used by default.

        Returns:
            (dict): A dictionary with an exec ``Id`` key.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """

        if environment is not None and utils.version_lt(self._version, '1.25'):
            raise errors.InvalidVersion(
                'Setting environment for exec is not supported in API < 1.25'
            )

        if isinstance(cmd, str):
            cmd = utils.split_command(cmd)

        if isinstance(environment, dict):
            environment = utils.utils.format_environment(environment)

        data = {
            'Container': container,
            'User': user,
            'Privileged': privileged,
            'Tty': tty,
            'AttachStdin': stdin,
            'AttachStdout': stdout,
            'AttachStderr': stderr,
            'Cmd': cmd,
            'Env': environment,
        }

        if workdir is not None:
            if utils.version_lt(self._version, '1.35'):
                raise errors.InvalidVersion(
                    'workdir is not supported for API version < 1.35'
                )
            data['WorkingDir'] = workdir

        if detach_keys:
            data['detachKeys'] = detach_keys
        elif 'detachKeys' in self._general_configs:
            data['detachKeys'] = self._general_configs['detachKeys']

        url = self._url('/containers/{0}/exec', container)
        res = self._post_json(url, data=data)
        return self._result(res, True)

    def exec_inspect(self, exec_id):
        """
        Return low-level information about an exec command.

        Args:
            exec_id (str): ID of the exec instance

        Returns:
            (dict): Dictionary of values returned by the endpoint.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        if isinstance(exec_id, dict):
            exec_id = exec_id.get('Id')
        res = self._get(self._url("/exec/{0}/json", exec_id))
        return self._result(res, True)

    def exec_resize(self, exec_id, height=None, width=None):
        """
        Resize the tty session used by the specified exec command.

        Args:
            exec_id (str): ID of the exec instance
            height (int): Height of tty session
            width (int): Width of tty session
        """

        if isinstance(exec_id, dict):
            exec_id = exec_id.get('Id')

        params = {'h': height, 'w': width}
        url = self._url("/exec/{0}/resize", exec_id)
        res = self._post(url, params=params)
        self._raise_for_status(res)

    @utils.check_resource('exec_id')
    def exec_start(self, exec_id, detach=False, tty=False, stream=False,
                   socket=False, demux=False):
        """
        Start a previously set up exec instance.

        Args:
            exec_id (str): ID of the exec instance
            detach (bool): If true, detach from the exec command.
                Default: False
            tty (bool): Allocate a pseudo-TTY. Default: False
            stream (bool): Stream response data. Default: False
            socket (bool): Return the connection socket to allow custom
                read/write operations.
            demux (bool): Return stdout and stderr separately

        Returns:

            (generator or str or tuple): If ``stream=True``, a generator
            yielding response chunks. If ``socket=True``, a socket object for
            the connection. A string containing response data otherwise. If
            ``demux=True``, a tuple with two elements of type byte: stdout and
            stderr.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        # we want opened socket if socket == True

        data = {
            'Tty': tty,
            'Detach': detach
        }

        headers = {} if detach else {
            'Connection': 'Upgrade',
            'Upgrade': 'tcp'
        }

        res = self._post_json(
            self._url('/exec/{0}/start', exec_id),
            headers=headers,
            data=data,
            stream=True
        )
        if detach:
            return self._result(res)
        if socket:
            return self._get_raw_response_socket(res)
        return self._read_from_socket(res, stream, tty=tty, demux=demux)
