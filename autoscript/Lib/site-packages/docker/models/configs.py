from ..api import APIClient
from .resource import Model, Collection


class Config(Model):
    """A config."""
    id_attribute = 'ID'

    def __repr__(self):
        return f"<{self.__class__.__name__}: '{self.name}'>"

    @property
    def name(self):
        return self.attrs['Spec']['Name']

    def remove(self):
        """
        Remove this config.

        Raises:
            :py:class:`docker.errors.APIError`
                If config failed to remove.
        """
        return self.client.api.remove_config(self.id)


class ConfigCollection(Collection):
    """Configs on the Docker server."""
    model = Config

    def create(self, **kwargs):
        obj = self.client.api.create_config(**kwargs)
        return self.prepare_model(obj)
    create.__doc__ = APIClient.create_config.__doc__

    def get(self, config_id):
        """
        Get a config.

        Args:
            config_id (str): Config ID.

        Returns:
            (:py:class:`Config`): The config.

        Raises:
            :py:class:`docker.errors.NotFound`
                If the config does not exist.
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.prepare_model(self.client.api.inspect_config(config_id))

    def list(self, **kwargs):
        """
        List configs. Similar to the ``docker config ls`` command.

        Args:
            filters (dict): Server-side list filtering options.

        Returns:
            (list of :py:class:`Config`): The configs.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        resp = self.client.api.configs(**kwargs)
        return [self.prepare_model(obj) for obj in resp]
