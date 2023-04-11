import logging
import os

from .. import auth, errors, utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE

log = logging.getLogger(__name__)


class ImageApiMixin:

    @utils.check_resource('image')
    def get_image(self, image, chunk_size=DEFAULT_DATA_CHUNK_SIZE):
        """
        Get a tarball of an image. Similar to the ``docker save`` command.

        Args:
            image (str): Image name to get
            chunk_size (int): The number of bytes returned by each iteration
                of the generator. If ``None``, data will be streamed as it is
                received. Default: 2 MB

        Returns:
            (generator): A stream of raw archive data.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Example:

            >>> image = client.api.get_image("busybox:latest")
            >>> f = open('/tmp/busybox-latest.tar', 'wb')
            >>> for chunk in image:
            >>>   f.write(chunk)
            >>> f.close()
        """
        res = self._get(self._url("/images/{0}/get", image), stream=True)
        return self._stream_raw_result(res, chunk_size, False)

    @utils.check_resource('image')
    def history(self, image):
        """
        Show the history of an image.

        Args:
            image (str): The image to show history for

        Returns:
            (str): The history of the image

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        res = self._get(self._url("/images/{0}/history", image))
        return self._result(res, True)

    def images(self, name=None, quiet=False, all=False, filters=None):
        """
        List images. Similar to the ``docker images`` command.

        Args:
            name (str): Only show images belonging to the repository ``name``
            quiet (bool): Only return numeric IDs as a list.
            all (bool): Show intermediate image layers. By default, these are
                filtered out.
            filters (dict): Filters to be processed on the image list.
                Available filters:
                - ``dangling`` (bool)
                - `label` (str|list): format either ``"key"``, ``"key=value"``
                    or a list of such.

        Returns:
            (dict or list): A list if ``quiet=True``, otherwise a dict.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        params = {
            'only_ids': 1 if quiet else 0,
            'all': 1 if all else 0,
        }
        if name:
            if utils.version_lt(self._version, '1.25'):
                # only use "filter" on API 1.24 and under, as it is deprecated
                params['filter'] = name
            else:
                if filters:
                    filters['reference'] = name
                else:
                    filters = {'reference': name}
        if filters:
            params['filters'] = utils.convert_filters(filters)
        res = self._result(self._get(self._url("/images/json"), params=params),
                           True)
        if quiet:
            return [x['Id'] for x in res]
        return res

    def import_image(self, src=None, repository=None, tag=None, image=None,
                     changes=None, stream_src=False):
        """
        Import an image. Similar to the ``docker import`` command.

        If ``src`` is a string or unicode string, it will first be treated as a
        path to a tarball on the local system. If there is an error reading
        from that file, ``src`` will be treated as a URL instead to fetch the
        image from. You can also pass an open file handle as ``src``, in which
        case the data will be read from that file.

        If ``src`` is unset but ``image`` is set, the ``image`` parameter will
        be taken as the name of an existing image to import from.

        Args:
            src (str or file): Path to tarfile, URL, or file-like object
            repository (str): The repository to create
            tag (str): The tag to apply
            image (str): Use another image like the ``FROM`` Dockerfile
                parameter
        """
        if not (src or image):
            raise errors.DockerException(
                'Must specify src or image to import from'
            )
        u = self._url('/images/create')

        params = _import_image_params(
            repository, tag, image,
            src=(src if isinstance(src, str) else None),
            changes=changes
        )
        headers = {'Content-Type': 'application/tar'}

        if image or params.get('fromSrc') != '-':  # from image or URL
            return self._result(
                self._post(u, data=None, params=params)
            )
        elif isinstance(src, str):  # from file path
            with open(src, 'rb') as f:
                return self._result(
                    self._post(
                        u, data=f, params=params, headers=headers, timeout=None
                    )
                )
        else:  # from raw data
            if stream_src:
                headers['Transfer-Encoding'] = 'chunked'
            return self._result(
                self._post(u, data=src, params=params, headers=headers)
            )

    def import_image_from_data(self, data, repository=None, tag=None,
                               changes=None):
        """
        Like :py:meth:`~docker.api.image.ImageApiMixin.import_image`, but
        allows importing in-memory bytes data.

        Args:
            data (bytes collection): Bytes collection containing valid tar data
            repository (str): The repository to create
            tag (str): The tag to apply
        """

        u = self._url('/images/create')
        params = _import_image_params(
            repository, tag, src='-', changes=changes
        )
        headers = {'Content-Type': 'application/tar'}
        return self._result(
            self._post(
                u, data=data, params=params, headers=headers, timeout=None
            )
        )

    def import_image_from_file(self, filename, repository=None, tag=None,
                               changes=None):
        """
        Like :py:meth:`~docker.api.image.ImageApiMixin.import_image`, but only
        supports importing from a tar file on disk.

        Args:
            filename (str): Full path to a tar file.
            repository (str): The repository to create
            tag (str): The tag to apply

        Raises:
            IOError: File does not exist.
        """

        return self.import_image(
            src=filename, repository=repository, tag=tag, changes=changes
        )

    def import_image_from_stream(self, stream, repository=None, tag=None,
                                 changes=None):
        return self.import_image(
            src=stream, stream_src=True, repository=repository, tag=tag,
            changes=changes
        )

    def import_image_from_url(self, url, repository=None, tag=None,
                              changes=None):
        """
        Like :py:meth:`~docker.api.image.ImageApiMixin.import_image`, but only
        supports importing from a URL.

        Args:
            url (str): A URL pointing to a tar file.
            repository (str): The repository to create
            tag (str): The tag to apply
        """
        return self.import_image(
            src=url, repository=repository, tag=tag, changes=changes
        )

    def import_image_from_image(self, image, repository=None, tag=None,
                                changes=None):
        """
        Like :py:meth:`~docker.api.image.ImageApiMixin.import_image`, but only
        supports importing from another image, like the ``FROM`` Dockerfile
        parameter.

        Args:
            image (str): Image name to import from
            repository (str): The repository to create
            tag (str): The tag to apply
        """
        return self.import_image(
            image=image, repository=repository, tag=tag, changes=changes
        )

    @utils.check_resource('image')
    def inspect_image(self, image):
        """
        Get detailed information about an image. Similar to the ``docker
        inspect`` command, but only for images.

        Args:
            image (str): The image to inspect

        Returns:
            (dict): Similar to the output of ``docker inspect``, but as a
        single dict

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self._result(
            self._get(self._url("/images/{0}/json", image)), True
        )

    @utils.minimum_version('1.30')
    @utils.check_resource('image')
    def inspect_distribution(self, image, auth_config=None):
        """
        Get image digest and platform information by contacting the registry.

        Args:
            image (str): The image name to inspect
            auth_config (dict): Override the credentials that are found in the
                config for this request.  ``auth_config`` should contain the
                ``username`` and ``password`` keys to be valid.

        Returns:
            (dict): A dict containing distribution data

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        registry, _ = auth.resolve_repository_name(image)

        headers = {}
        if auth_config is None:
            header = auth.get_config_header(self, registry)
            if header:
                headers['X-Registry-Auth'] = header
        else:
            log.debug('Sending supplied auth config')
            headers['X-Registry-Auth'] = auth.encode_header(auth_config)

        url = self._url("/distribution/{0}/json", image)

        return self._result(
            self._get(url, headers=headers), True
        )

    def load_image(self, data, quiet=None):
        """
        Load an image that was previously saved using
        :py:meth:`~docker.api.image.ImageApiMixin.get_image` (or ``docker
        save``). Similar to ``docker load``.

        Args:
            data (binary): Image data to be loaded.
            quiet (boolean): Suppress progress details in response.

        Returns:
            (generator): Progress output as JSON objects. Only available for
                         API version >= 1.23

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        params = {}

        if quiet is not None:
            if utils.version_lt(self._version, '1.23'):
                raise errors.InvalidVersion(
                    'quiet is not supported in API version < 1.23'
                )
            params['quiet'] = quiet

        res = self._post(
            self._url("/images/load"), data=data, params=params, stream=True
        )
        if utils.version_gte(self._version, '1.23'):
            return self._stream_helper(res, decode=True)

        self._raise_for_status(res)

    @utils.minimum_version('1.25')
    def prune_images(self, filters=None):
        """
        Delete unused images

        Args:
            filters (dict): Filters to process on the prune list.
                Available filters:
                - dangling (bool):  When set to true (or 1), prune only
                unused and untagged images.

        Returns:
            (dict): A dict containing a list of deleted image IDs and
                the amount of disk space reclaimed in bytes.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        url = self._url("/images/prune")
        params = {}
        if filters is not None:
            params['filters'] = utils.convert_filters(filters)
        return self._result(self._post(url, params=params), True)

    def pull(self, repository, tag=None, stream=False, auth_config=None,
             decode=False, platform=None, all_tags=False):
        """
        Pulls an image. Similar to the ``docker pull`` command.

        Args:
            repository (str): The repository to pull
            tag (str): The tag to pull. If ``tag`` is ``None`` or empty, it
                is set to ``latest``.
            stream (bool): Stream the output as a generator. Make sure to
                consume the generator, otherwise pull might get cancelled.
            auth_config (dict): Override the credentials that are found in the
                config for this request.  ``auth_config`` should contain the
                ``username`` and ``password`` keys to be valid.
            decode (bool): Decode the JSON data from the server into dicts.
                Only applies with ``stream=True``
            platform (str): Platform in the format ``os[/arch[/variant]]``
            all_tags (bool): Pull all image tags, the ``tag`` parameter is
                ignored.

        Returns:
            (generator or str): The output

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Example:

            >>> resp = client.api.pull('busybox', stream=True, decode=True)
            ... for line in resp:
            ...     print(json.dumps(line, indent=4))
            {
                "status": "Pulling image (latest) from busybox",
                "progressDetail": {},
                "id": "e72ac664f4f0"
            }
            {
                "status": "Pulling image (latest) from busybox, endpoint: ...",
                "progressDetail": {},
                "id": "e72ac664f4f0"
            }

        """
        repository, image_tag = utils.parse_repository_tag(repository)
        tag = tag or image_tag or 'latest'

        if all_tags:
            tag = None

        registry, repo_name = auth.resolve_repository_name(repository)

        params = {
            'tag': tag,
            'fromImage': repository
        }
        headers = {}

        if auth_config is None:
            header = auth.get_config_header(self, registry)
            if header:
                headers['X-Registry-Auth'] = header
        else:
            log.debug('Sending supplied auth config')
            headers['X-Registry-Auth'] = auth.encode_header(auth_config)

        if platform is not None:
            if utils.version_lt(self._version, '1.32'):
                raise errors.InvalidVersion(
                    'platform was only introduced in API version 1.32'
                )
            params['platform'] = platform

        response = self._post(
            self._url('/images/create'), params=params, headers=headers,
            stream=stream, timeout=None
        )

        self._raise_for_status(response)

        if stream:
            return self._stream_helper(response, decode=decode)

        return self._result(response)

    def push(self, repository, tag=None, stream=False, auth_config=None,
             decode=False):
        """
        Push an image or a repository to the registry. Similar to the ``docker
        push`` command.

        Args:
            repository (str): The repository to push to
            tag (str): An optional tag to push
            stream (bool): Stream the output as a blocking generator
            auth_config (dict): Override the credentials that are found in the
                config for this request.  ``auth_config`` should contain the
                ``username`` and ``password`` keys to be valid.
            decode (bool): Decode the JSON data from the server into dicts.
                Only applies with ``stream=True``

        Returns:
            (generator or str): The output from the server.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Example:
            >>> resp = client.api.push(
            ...     'yourname/app',
            ...     stream=True,
            ...     decode=True,
            ... )
            ... for line in resp:
            ...   print(line)
            {'status': 'Pushing repository yourname/app (1 tags)'}
            {'status': 'Pushing','progressDetail': {}, 'id': '511136ea3c5a'}
            {'status': 'Image already pushed, skipping', 'progressDetail':{},
             'id': '511136ea3c5a'}
            ...

        """
        if not tag:
            repository, tag = utils.parse_repository_tag(repository)
        registry, repo_name = auth.resolve_repository_name(repository)
        u = self._url("/images/{0}/push", repository)
        params = {
            'tag': tag
        }
        headers = {}

        if auth_config is None:
            header = auth.get_config_header(self, registry)
            if header:
                headers['X-Registry-Auth'] = header
        else:
            log.debug('Sending supplied auth config')
            headers['X-Registry-Auth'] = auth.encode_header(auth_config)

        response = self._post_json(
            u, None, headers=headers, stream=stream, params=params
        )

        self._raise_for_status(response)

        if stream:
            return self._stream_helper(response, decode=decode)

        return self._result(response)

    @utils.check_resource('image')
    def remove_image(self, image, force=False, noprune=False):
        """
        Remove an image. Similar to the ``docker rmi`` command.

        Args:
            image (str): The image to remove
            force (bool): Force removal of the image
            noprune (bool): Do not delete untagged parents
        """
        params = {'force': force, 'noprune': noprune}
        res = self._delete(self._url("/images/{0}", image), params=params)
        return self._result(res, True)

    def search(self, term, limit=None):
        """
        Search for images on Docker Hub. Similar to the ``docker search``
        command.

        Args:
            term (str): A term to search for.
            limit (int): The maximum number of results to return.

        Returns:
            (list of dicts): The response of the search.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        params = {'term': term}
        if limit is not None:
            params['limit'] = limit

        return self._result(
            self._get(self._url("/images/search"), params=params),
            True
        )

    @utils.check_resource('image')
    def tag(self, image, repository, tag=None, force=False):
        """
        Tag an image into a repository. Similar to the ``docker tag`` command.

        Args:
            image (str): The image to tag
            repository (str): The repository to set for the tag
            tag (str): The tag name
            force (bool): Force

        Returns:
            (bool): ``True`` if successful

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Example:

            >>> client.api.tag('ubuntu', 'localhost:5000/ubuntu', 'latest',
                           force=True)
        """
        params = {
            'tag': tag,
            'repo': repository,
            'force': 1 if force else 0
        }
        url = self._url("/images/{0}/tag", image)
        res = self._post(url, params=params)
        self._raise_for_status(res)
        return res.status_code == 201


def is_file(src):
    try:
        return (
            isinstance(src, str) and
            os.path.isfile(src)
        )
    except TypeError:  # a data string will make isfile() raise a TypeError
        return False


def _import_image_params(repo, tag, image=None, src=None,
                         changes=None):
    params = {
        'repo': repo,
        'tag': tag,
    }
    if image:
        params['fromImage'] = image
    elif src and not is_file(src):
        params['fromSrc'] = src
    else:
        params['fromSrc'] = '-'

    if changes:
        params['changes'] = changes

    return params
