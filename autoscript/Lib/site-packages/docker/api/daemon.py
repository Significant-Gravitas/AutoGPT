import os
from datetime import datetime

from .. import auth, types, utils


class DaemonApiMixin:
    @utils.minimum_version('1.25')
    def df(self):
        """
        Get data usage information.

        Returns:
            (dict): A dictionary representing different resource categories
            and their respective data usage.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        url = self._url('/system/df')
        return self._result(self._get(url), True)

    def events(self, since=None, until=None, filters=None, decode=None):
        """
        Get real-time events from the server. Similar to the ``docker events``
        command.

        Args:
            since (UTC datetime or int): Get events from this point
            until (UTC datetime or int): Get events until this point
            filters (dict): Filter the events by event time, container or image
            decode (bool): If set to true, stream will be decoded into dicts on
                the fly. False by default.

        Returns:
            A :py:class:`docker.types.daemon.CancellableStream` generator

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Example:

            >>> for event in client.events(decode=True)
            ...   print(event)
            {u'from': u'image/with:tag',
             u'id': u'container-id',
             u'status': u'start',
             u'time': 1423339459}
            ...

            or

            >>> events = client.events()
            >>> for event in events:
            ...   print(event)
            >>> # and cancel from another thread
            >>> events.close()
        """

        if isinstance(since, datetime):
            since = utils.datetime_to_timestamp(since)

        if isinstance(until, datetime):
            until = utils.datetime_to_timestamp(until)

        if filters:
            filters = utils.convert_filters(filters)

        params = {
            'since': since,
            'until': until,
            'filters': filters
        }
        url = self._url('/events')

        response = self._get(url, params=params, stream=True, timeout=None)
        stream = self._stream_helper(response, decode=decode)

        return types.CancellableStream(stream, response)

    def info(self):
        """
        Display system-wide information. Identical to the ``docker info``
        command.

        Returns:
            (dict): The info as a dict

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self._result(self._get(self._url("/info")), True)

    def login(self, username, password=None, email=None, registry=None,
              reauth=False, dockercfg_path=None):
        """
        Authenticate with a registry. Similar to the ``docker login`` command.

        Args:
            username (str): The registry username
            password (str): The plaintext password
            email (str): The email for the registry account
            registry (str): URL to the registry.  E.g.
                ``https://index.docker.io/v1/``
            reauth (bool): Whether or not to refresh existing authentication on
                the Docker server.
            dockercfg_path (str): Use a custom path for the Docker config file
                (default ``$HOME/.docker/config.json`` if present,
                otherwise ``$HOME/.dockercfg``)

        Returns:
            (dict): The response from the login request

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """

        # If we don't have any auth data so far, try reloading the config file
        # one more time in case anything showed up in there.
        # If dockercfg_path is passed check to see if the config file exists,
        # if so load that config.
        if dockercfg_path and os.path.exists(dockercfg_path):
            self._auth_configs = auth.load_config(
                dockercfg_path, credstore_env=self.credstore_env
            )
        elif not self._auth_configs or self._auth_configs.is_empty:
            self._auth_configs = auth.load_config(
                credstore_env=self.credstore_env
            )

        authcfg = self._auth_configs.resolve_authconfig(registry)
        # If we found an existing auth config for this registry and username
        # combination, we can return it immediately unless reauth is requested.
        if authcfg and authcfg.get('username', None) == username \
                and not reauth:
            return authcfg

        req_data = {
            'username': username,
            'password': password,
            'email': email,
            'serveraddress': registry,
        }

        response = self._post_json(self._url('/auth'), data=req_data)
        if response.status_code == 200:
            self._auth_configs.add_auth(registry or auth.INDEX_NAME, req_data)
        return self._result(response, json=True)

    def ping(self):
        """
        Checks the server is responsive. An exception will be raised if it
        isn't responding.

        Returns:
            (bool) The response from the server.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self._result(self._get(self._url('/_ping'))) == 'OK'

    def version(self, api_version=True):
        """
        Returns version information from the server. Similar to the ``docker
        version`` command.

        Returns:
            (dict): The server version information

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        url = self._url("/version", versioned_api=api_version)
        return self._result(self._get(url), json=True)
