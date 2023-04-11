from .. import auth, utils


class PluginApiMixin:
    @utils.minimum_version('1.25')
    @utils.check_resource('name')
    def configure_plugin(self, name, options):
        """
            Configure a plugin.

            Args:
                name (string): The name of the plugin. The ``:latest`` tag is
                    optional, and is the default if omitted.
                options (dict): A key-value mapping of options

            Returns:
                ``True`` if successful
        """
        url = self._url('/plugins/{0}/set', name)
        data = options
        if isinstance(data, dict):
            data = [f'{k}={v}' for k, v in data.items()]
        res = self._post_json(url, data=data)
        self._raise_for_status(res)
        return True

    @utils.minimum_version('1.25')
    def create_plugin(self, name, plugin_data_dir, gzip=False):
        """
            Create a new plugin.

            Args:
                name (string): The name of the plugin. The ``:latest`` tag is
                    optional, and is the default if omitted.
                plugin_data_dir (string): Path to the plugin data directory.
                    Plugin data directory must contain the ``config.json``
                    manifest file and the ``rootfs`` directory.
                gzip (bool): Compress the context using gzip. Default: False

            Returns:
                ``True`` if successful
        """
        url = self._url('/plugins/create')

        with utils.create_archive(
            root=plugin_data_dir, gzip=gzip,
            files=set(utils.build.walk(plugin_data_dir, []))
        ) as archv:
            res = self._post(url, params={'name': name}, data=archv)
        self._raise_for_status(res)
        return True

    @utils.minimum_version('1.25')
    def disable_plugin(self, name, force=False):
        """
            Disable an installed plugin.

            Args:
                name (string): The name of the plugin. The ``:latest`` tag is
                    optional, and is the default if omitted.
                force (bool): To enable the force query parameter.

            Returns:
                ``True`` if successful
        """
        url = self._url('/plugins/{0}/disable', name)
        res = self._post(url, params={'force': force})
        self._raise_for_status(res)
        return True

    @utils.minimum_version('1.25')
    def enable_plugin(self, name, timeout=0):
        """
            Enable an installed plugin.

            Args:
                name (string): The name of the plugin. The ``:latest`` tag is
                    optional, and is the default if omitted.
                timeout (int): Operation timeout (in seconds). Default: 0

            Returns:
                ``True`` if successful
        """
        url = self._url('/plugins/{0}/enable', name)
        params = {'timeout': timeout}
        res = self._post(url, params=params)
        self._raise_for_status(res)
        return True

    @utils.minimum_version('1.25')
    def inspect_plugin(self, name):
        """
            Retrieve plugin metadata.

            Args:
                name (string): The name of the plugin. The ``:latest`` tag is
                    optional, and is the default if omitted.

            Returns:
                A dict containing plugin info
        """
        url = self._url('/plugins/{0}/json', name)
        return self._result(self._get(url), True)

    @utils.minimum_version('1.25')
    def pull_plugin(self, remote, privileges, name=None):
        """
            Pull and install a plugin. After the plugin is installed, it can be
            enabled using :py:meth:`~enable_plugin`.

            Args:
                remote (string): Remote reference for the plugin to install.
                    The ``:latest`` tag is optional, and is the default if
                    omitted.
                privileges (:py:class:`list`): A list of privileges the user
                    consents to grant to the plugin. Can be retrieved using
                    :py:meth:`~plugin_privileges`.
                name (string): Local name for the pulled plugin. The
                    ``:latest`` tag is optional, and is the default if omitted.

            Returns:
                An iterable object streaming the decoded API logs
        """
        url = self._url('/plugins/pull')
        params = {
            'remote': remote,
        }
        if name:
            params['name'] = name

        headers = {}
        registry, repo_name = auth.resolve_repository_name(remote)
        header = auth.get_config_header(self, registry)
        if header:
            headers['X-Registry-Auth'] = header
        response = self._post_json(
            url, params=params, headers=headers, data=privileges,
            stream=True
        )
        self._raise_for_status(response)
        return self._stream_helper(response, decode=True)

    @utils.minimum_version('1.25')
    def plugins(self):
        """
            Retrieve a list of installed plugins.

            Returns:
                A list of dicts, one per plugin
        """
        url = self._url('/plugins')
        return self._result(self._get(url), True)

    @utils.minimum_version('1.25')
    def plugin_privileges(self, name):
        """
            Retrieve list of privileges to be granted to a plugin.

            Args:
                name (string): Name of the remote plugin to examine. The
                    ``:latest`` tag is optional, and is the default if omitted.

            Returns:
                A list of dictionaries representing the plugin's
                permissions

        """
        params = {
            'remote': name,
        }

        headers = {}
        registry, repo_name = auth.resolve_repository_name(name)
        header = auth.get_config_header(self, registry)
        if header:
            headers['X-Registry-Auth'] = header

        url = self._url('/plugins/privileges')
        return self._result(
            self._get(url, params=params, headers=headers), True
        )

    @utils.minimum_version('1.25')
    @utils.check_resource('name')
    def push_plugin(self, name):
        """
            Push a plugin to the registry.

            Args:
                name (string): Name of the plugin to upload. The ``:latest``
                    tag is optional, and is the default if omitted.

            Returns:
                ``True`` if successful
        """
        url = self._url('/plugins/{0}/pull', name)

        headers = {}
        registry, repo_name = auth.resolve_repository_name(name)
        header = auth.get_config_header(self, registry)
        if header:
            headers['X-Registry-Auth'] = header
        res = self._post(url, headers=headers)
        self._raise_for_status(res)
        return self._stream_helper(res, decode=True)

    @utils.minimum_version('1.25')
    @utils.check_resource('name')
    def remove_plugin(self, name, force=False):
        """
            Remove an installed plugin.

            Args:
                name (string): Name of the plugin to remove. The ``:latest``
                    tag is optional, and is the default if omitted.
                force (bool): Disable the plugin before removing. This may
                    result in issues if the plugin is in use by a container.

            Returns:
                ``True`` if successful
        """
        url = self._url('/plugins/{0}', name)
        res = self._delete(url, params={'force': force})
        self._raise_for_status(res)
        return True

    @utils.minimum_version('1.26')
    @utils.check_resource('name')
    def upgrade_plugin(self, name, remote, privileges):
        """
            Upgrade an installed plugin.

            Args:
                name (string): Name of the plugin to upgrade. The ``:latest``
                    tag is optional and is the default if omitted.
                remote (string): Remote reference to upgrade to. The
                    ``:latest`` tag is optional and is the default if omitted.
                privileges (:py:class:`list`): A list of privileges the user
                    consents to grant to the plugin. Can be retrieved using
                    :py:meth:`~plugin_privileges`.

            Returns:
                An iterable object streaming the decoded API logs
        """

        url = self._url('/plugins/{0}/upgrade', name)
        params = {
            'remote': remote,
        }

        headers = {}
        registry, repo_name = auth.resolve_repository_name(remote)
        header = auth.get_config_header(self, registry)
        if header:
            headers['X-Registry-Auth'] = header
        response = self._post_json(
            url, params=params, headers=headers, data=privileges,
            stream=True
        )
        self._raise_for_status(response)
        return self._stream_helper(response, decode=True)
