from .. import errors
from .resource import Collection, Model


class Plugin(Model):
    """
    A plugin on the server.
    """
    def __repr__(self):
        return f"<{self.__class__.__name__}: '{self.name}'>"

    @property
    def name(self):
        """
        The plugin's name.
        """
        return self.attrs.get('Name')

    @property
    def enabled(self):
        """
        Whether the plugin is enabled.
        """
        return self.attrs.get('Enabled')

    @property
    def settings(self):
        """
        A dictionary representing the plugin's configuration.
        """
        return self.attrs.get('Settings')

    def configure(self, options):
        """
            Update the plugin's settings.

            Args:
                options (dict): A key-value mapping of options.

            Raises:
                :py:class:`docker.errors.APIError`
                    If the server returns an error.
        """
        self.client.api.configure_plugin(self.name, options)
        self.reload()

    def disable(self, force=False):
        """
            Disable the plugin.

            Args:
                force (bool): Force disable. Default: False

            Raises:
                :py:class:`docker.errors.APIError`
                    If the server returns an error.
        """

        self.client.api.disable_plugin(self.name, force)
        self.reload()

    def enable(self, timeout=0):
        """
            Enable the plugin.

            Args:
                timeout (int): Timeout in seconds. Default: 0

            Raises:
                :py:class:`docker.errors.APIError`
                    If the server returns an error.
        """
        self.client.api.enable_plugin(self.name, timeout)
        self.reload()

    def push(self):
        """
            Push the plugin to a remote registry.

            Returns:
                A dict iterator streaming the status of the upload.

            Raises:
                :py:class:`docker.errors.APIError`
                    If the server returns an error.
        """
        return self.client.api.push_plugin(self.name)

    def remove(self, force=False):
        """
            Remove the plugin from the server.

            Args:
                force (bool): Remove even if the plugin is enabled.
                    Default: False

            Raises:
                :py:class:`docker.errors.APIError`
                    If the server returns an error.
        """
        return self.client.api.remove_plugin(self.name, force=force)

    def upgrade(self, remote=None):
        """
            Upgrade the plugin.

            Args:
                remote (string): Remote reference to upgrade to. The
                    ``:latest`` tag is optional and is the default if omitted.
                    Default: this plugin's name.

            Returns:
                A generator streaming the decoded API logs
        """
        if self.enabled:
            raise errors.DockerError(
                'Plugin must be disabled before upgrading.'
            )

        if remote is None:
            remote = self.name
        privileges = self.client.api.plugin_privileges(remote)
        yield from self.client.api.upgrade_plugin(
            self.name,
            remote,
            privileges,
        )
        self.reload()


class PluginCollection(Collection):
    model = Plugin

    def create(self, name, plugin_data_dir, gzip=False):
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
                (:py:class:`Plugin`): The newly created plugin.
        """
        self.client.api.create_plugin(name, plugin_data_dir, gzip)
        return self.get(name)

    def get(self, name):
        """
        Gets a plugin.

        Args:
            name (str): The name of the plugin.

        Returns:
            (:py:class:`Plugin`): The plugin.

        Raises:
            :py:class:`docker.errors.NotFound` If the plugin does not
            exist.
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.prepare_model(self.client.api.inspect_plugin(name))

    def install(self, remote_name, local_name=None):
        """
            Pull and install a plugin.

            Args:
                remote_name (string): Remote reference for the plugin to
                    install. The ``:latest`` tag is optional, and is the
                    default if omitted.
                local_name (string): Local name for the pulled plugin.
                    The ``:latest`` tag is optional, and is the default if
                    omitted. Optional.

            Returns:
                (:py:class:`Plugin`): The installed plugin
            Raises:
                :py:class:`docker.errors.APIError`
                    If the server returns an error.
        """
        privileges = self.client.api.plugin_privileges(remote_name)
        it = self.client.api.pull_plugin(remote_name, privileges, local_name)
        for data in it:
            pass
        return self.get(local_name or remote_name)

    def list(self):
        """
        List plugins installed on the server.

        Returns:
            (list of :py:class:`Plugin`): The plugins.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        resp = self.client.api.plugins()
        return [self.prepare_model(r) for r in resp]
