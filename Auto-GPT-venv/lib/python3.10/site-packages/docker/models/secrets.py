from ..api import APIClient
from .resource import Model, Collection


class Secret(Model):
    """A secret."""
    id_attribute = 'ID'

    def __repr__(self):
        return f"<{self.__class__.__name__}: '{self.name}'>"

    @property
    def name(self):
        return self.attrs['Spec']['Name']

    def remove(self):
        """
        Remove this secret.

        Raises:
            :py:class:`docker.errors.APIError`
                If secret failed to remove.
        """
        return self.client.api.remove_secret(self.id)


class SecretCollection(Collection):
    """Secrets on the Docker server."""
    model = Secret

    def create(self, **kwargs):
        obj = self.client.api.create_secret(**kwargs)
        obj.setdefault("Spec", {})["Name"] = kwargs.get("name")
        return self.prepare_model(obj)
    create.__doc__ = APIClient.create_secret.__doc__

    def get(self, secret_id):
        """
        Get a secret.

        Args:
            secret_id (str): Secret ID.

        Returns:
            (:py:class:`Secret`): The secret.

        Raises:
            :py:class:`docker.errors.NotFound`
                If the secret does not exist.
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.prepare_model(self.client.api.inspect_secret(secret_id))

    def list(self, **kwargs):
        """
        List secrets. Similar to the ``docker secret ls`` command.

        Args:
            filters (dict): Server-side list filtering options.

        Returns:
            (list of :py:class:`Secret`): The secrets.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        resp = self.client.api.secrets(**kwargs)
        return [self.prepare_model(obj) for obj in resp]
