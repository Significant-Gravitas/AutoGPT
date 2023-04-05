import json
import logging
import os
import random

from .. import auth
from .. import constants
from .. import errors
from .. import utils


log = logging.getLogger(__name__)


class BuildApiMixin:
    def build(self, path=None, tag=None, quiet=False, fileobj=None,
              nocache=False, rm=False, timeout=None,
              custom_context=False, encoding=None, pull=False,
              forcerm=False, dockerfile=None, container_limits=None,
              decode=False, buildargs=None, gzip=False, shmsize=None,
              labels=None, cache_from=None, target=None, network_mode=None,
              squash=None, extra_hosts=None, platform=None, isolation=None,
              use_config_proxy=True):
        """
        Similar to the ``docker build`` command. Either ``path`` or ``fileobj``
        needs to be set. ``path`` can be a local path (to a directory
        containing a Dockerfile) or a remote URL. ``fileobj`` must be a
        readable file-like object to a Dockerfile.

        If you have a tar file for the Docker build context (including a
        Dockerfile) already, pass a readable file-like object to ``fileobj``
        and also pass ``custom_context=True``. If the stream is compressed
        also, set ``encoding`` to the correct value (e.g ``gzip``).

        Example:
            >>> from io import BytesIO
            >>> from docker import APIClient
            >>> dockerfile = '''
            ... # Shared Volume
            ... FROM busybox:buildroot-2014.02
            ... VOLUME /data
            ... CMD ["/bin/sh"]
            ... '''
            >>> f = BytesIO(dockerfile.encode('utf-8'))
            >>> cli = APIClient(base_url='tcp://127.0.0.1:2375')
            >>> response = [line for line in cli.build(
            ...     fileobj=f, rm=True, tag='yourname/volume'
            ... )]
            >>> response
            ['{"stream":" ---\\u003e a9eb17255234\\n"}',
             '{"stream":"Step 1 : VOLUME /data\\n"}',
             '{"stream":" ---\\u003e Running in abdc1e6896c6\\n"}',
             '{"stream":" ---\\u003e 713bca62012e\\n"}',
             '{"stream":"Removing intermediate container abdc1e6896c6\\n"}',
             '{"stream":"Step 2 : CMD [\\"/bin/sh\\"]\\n"}',
             '{"stream":" ---\\u003e Running in dba30f2a1a7e\\n"}',
             '{"stream":" ---\\u003e 032b8b2855fc\\n"}',
             '{"stream":"Removing intermediate container dba30f2a1a7e\\n"}',
             '{"stream":"Successfully built 032b8b2855fc\\n"}']

        Args:
            path (str): Path to the directory containing the Dockerfile
            fileobj: A file object to use as the Dockerfile. (Or a file-like
                object)
            tag (str): A tag to add to the final image
            quiet (bool): Whether to return the status
            nocache (bool): Don't use the cache when set to ``True``
            rm (bool): Remove intermediate containers. The ``docker build``
                command now defaults to ``--rm=true``, but we have kept the old
                default of `False` to preserve backward compatibility
            timeout (int): HTTP timeout
            custom_context (bool): Optional if using ``fileobj``
            encoding (str): The encoding for a stream. Set to ``gzip`` for
                compressing
            pull (bool): Downloads any updates to the FROM image in Dockerfiles
            forcerm (bool): Always remove intermediate containers, even after
                unsuccessful builds
            dockerfile (str): path within the build context to the Dockerfile
            gzip (bool): If set to ``True``, gzip compression/encoding is used
            buildargs (dict): A dictionary of build arguments
            container_limits (dict): A dictionary of limits applied to each
                container created by the build process. Valid keys:

                - memory (int): set memory limit for build
                - memswap (int): Total memory (memory + swap), -1 to disable
                    swap
                - cpushares (int): CPU shares (relative weight)
                - cpusetcpus (str): CPUs in which to allow execution, e.g.,
                    ``"0-3"``, ``"0,1"``
            decode (bool): If set to ``True``, the returned stream will be
                decoded into dicts on the fly. Default ``False``
            shmsize (int): Size of `/dev/shm` in bytes. The size must be
                greater than 0. If omitted the system uses 64MB
            labels (dict): A dictionary of labels to set on the image
            cache_from (:py:class:`list`): A list of images used for build
                cache resolution
            target (str): Name of the build-stage to build in a multi-stage
                Dockerfile
            network_mode (str): networking mode for the run commands during
                build
            squash (bool): Squash the resulting images layers into a
                single layer.
            extra_hosts (dict): Extra hosts to add to /etc/hosts in building
                containers, as a mapping of hostname to IP address.
            platform (str): Platform in the format ``os[/arch[/variant]]``
            isolation (str): Isolation technology used during build.
                Default: `None`.
            use_config_proxy (bool): If ``True``, and if the docker client
                configuration file (``~/.docker/config.json`` by default)
                contains a proxy configuration, the corresponding environment
                variables will be set in the container being built.

        Returns:
            A generator for the build output.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
            ``TypeError``
                If neither ``path`` nor ``fileobj`` is specified.
        """
        remote = context = None
        headers = {}
        container_limits = container_limits or {}
        buildargs = buildargs or {}
        if path is None and fileobj is None:
            raise TypeError("Either path or fileobj needs to be provided.")
        if gzip and encoding is not None:
            raise errors.DockerException(
                'Can not use custom encoding if gzip is enabled'
            )

        for key in container_limits.keys():
            if key not in constants.CONTAINER_LIMITS_KEYS:
                raise errors.DockerException(
                    f'Invalid container_limits key {key}'
                )

        if custom_context:
            if not fileobj:
                raise TypeError("You must specify fileobj with custom_context")
            context = fileobj
        elif fileobj is not None:
            context = utils.mkbuildcontext(fileobj)
        elif path.startswith(('http://', 'https://',
                              'git://', 'github.com/', 'git@')):
            remote = path
        elif not os.path.isdir(path):
            raise TypeError("You must specify a directory to build in path")
        else:
            dockerignore = os.path.join(path, '.dockerignore')
            exclude = None
            if os.path.exists(dockerignore):
                with open(dockerignore) as f:
                    exclude = list(filter(
                        lambda x: x != '' and x[0] != '#',
                        [line.strip() for line in f.read().splitlines()]
                    ))
            dockerfile = process_dockerfile(dockerfile, path)
            context = utils.tar(
                path, exclude=exclude, dockerfile=dockerfile, gzip=gzip
            )
            encoding = 'gzip' if gzip else encoding

        u = self._url('/build')
        params = {
            't': tag,
            'remote': remote,
            'q': quiet,
            'nocache': nocache,
            'rm': rm,
            'forcerm': forcerm,
            'pull': pull,
            'dockerfile': dockerfile,
        }
        params.update(container_limits)

        if use_config_proxy:
            proxy_args = self._proxy_configs.get_environment()
            for k, v in proxy_args.items():
                buildargs.setdefault(k, v)
        if buildargs:
            params.update({'buildargs': json.dumps(buildargs)})

        if shmsize:
            if utils.version_gte(self._version, '1.22'):
                params.update({'shmsize': shmsize})
            else:
                raise errors.InvalidVersion(
                    'shmsize was only introduced in API version 1.22'
                )

        if labels:
            if utils.version_gte(self._version, '1.23'):
                params.update({'labels': json.dumps(labels)})
            else:
                raise errors.InvalidVersion(
                    'labels was only introduced in API version 1.23'
                )

        if cache_from:
            if utils.version_gte(self._version, '1.25'):
                params.update({'cachefrom': json.dumps(cache_from)})
            else:
                raise errors.InvalidVersion(
                    'cache_from was only introduced in API version 1.25'
                )

        if target:
            if utils.version_gte(self._version, '1.29'):
                params.update({'target': target})
            else:
                raise errors.InvalidVersion(
                    'target was only introduced in API version 1.29'
                )

        if network_mode:
            if utils.version_gte(self._version, '1.25'):
                params.update({'networkmode': network_mode})
            else:
                raise errors.InvalidVersion(
                    'network_mode was only introduced in API version 1.25'
                )

        if squash:
            if utils.version_gte(self._version, '1.25'):
                params.update({'squash': squash})
            else:
                raise errors.InvalidVersion(
                    'squash was only introduced in API version 1.25'
                )

        if extra_hosts is not None:
            if utils.version_lt(self._version, '1.27'):
                raise errors.InvalidVersion(
                    'extra_hosts was only introduced in API version 1.27'
                )

            if isinstance(extra_hosts, dict):
                extra_hosts = utils.format_extra_hosts(extra_hosts)
            params.update({'extrahosts': extra_hosts})

        if platform is not None:
            if utils.version_lt(self._version, '1.32'):
                raise errors.InvalidVersion(
                    'platform was only introduced in API version 1.32'
                )
            params['platform'] = platform

        if isolation is not None:
            if utils.version_lt(self._version, '1.24'):
                raise errors.InvalidVersion(
                    'isolation was only introduced in API version 1.24'
                )
            params['isolation'] = isolation

        if context is not None:
            headers = {'Content-Type': 'application/tar'}
            if encoding:
                headers['Content-Encoding'] = encoding

        self._set_auth_headers(headers)

        response = self._post(
            u,
            data=context,
            params=params,
            headers=headers,
            stream=True,
            timeout=timeout,
        )

        if context is not None and not custom_context:
            context.close()

        return self._stream_helper(response, decode=decode)

    @utils.minimum_version('1.31')
    def prune_builds(self):
        """
        Delete the builder cache

        Returns:
            (dict): A dictionary containing information about the operation's
                    result. The ``SpaceReclaimed`` key indicates the amount of
                    bytes of disk space reclaimed.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        url = self._url("/build/prune")
        return self._result(self._post(url), True)

    def _set_auth_headers(self, headers):
        log.debug('Looking for auth config')

        # If we don't have any auth data so far, try reloading the config
        # file one more time in case anything showed up in there.
        if not self._auth_configs or self._auth_configs.is_empty:
            log.debug("No auth config in memory - loading from filesystem")
            self._auth_configs = auth.load_config(
                credstore_env=self.credstore_env
            )

        # Send the full auth configuration (if any exists), since the build
        # could use any (or all) of the registries.
        if self._auth_configs:
            auth_data = self._auth_configs.get_all_credentials()

            # See https://github.com/docker/docker-py/issues/1683
            if (auth.INDEX_URL not in auth_data and
                    auth.INDEX_NAME in auth_data):
                auth_data[auth.INDEX_URL] = auth_data.get(auth.INDEX_NAME, {})

            log.debug(
                'Sending auth config ({})'.format(
                    ', '.join(repr(k) for k in auth_data.keys())
                )
            )

            if auth_data:
                headers['X-Registry-Config'] = auth.encode_header(
                    auth_data
                )
        else:
            log.debug('No auth config found')


def process_dockerfile(dockerfile, path):
    if not dockerfile:
        return (None, None)

    abs_dockerfile = dockerfile
    if not os.path.isabs(dockerfile):
        abs_dockerfile = os.path.join(path, dockerfile)
        if constants.IS_WINDOWS_PLATFORM and path.startswith(
                constants.WINDOWS_LONGPATH_PREFIX):
            abs_dockerfile = '{}{}'.format(
                constants.WINDOWS_LONGPATH_PREFIX,
                os.path.normpath(
                    abs_dockerfile[len(constants.WINDOWS_LONGPATH_PREFIX):]
                )
            )
    if (os.path.splitdrive(path)[0] != os.path.splitdrive(abs_dockerfile)[0] or
            os.path.relpath(abs_dockerfile, path).startswith('..')):
        # Dockerfile not in context - read data to insert into tar later
        with open(abs_dockerfile) as df:
            return (
                f'.dockerfile.{random.getrandbits(160):x}',
                df.read()
            )

    # Dockerfile is inside the context - return path relative to context root
    if dockerfile == abs_dockerfile:
        # Only calculate relpath if necessary to avoid errors
        # on Windows client -> Linux Docker
        # see https://github.com/docker/compose/issues/5969
        dockerfile = os.path.relpath(abs_dockerfile, path)
    return (dockerfile, None)
