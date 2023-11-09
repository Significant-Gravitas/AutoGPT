import base64

from .. import errors
from .. import utils


class SecretApiMixin:
    @utils.minimum_version('1.25')
    def create_secret(self, name, data, labels=None, driver=None):
        """
            Create a secret

            Args:
                name (string): Name of the secret
                data (bytes): Secret data to be stored
                labels (dict): A mapping of labels to assign to the secret
                driver (DriverConfig): A custom driver configuration. If
                    unspecified, the default ``internal`` driver will be used

            Returns (dict): ID of the newly created secret
        """
        if not isinstance(data, bytes):
            data = data.encode('utf-8')

        data = base64.b64encode(data)
        data = data.decode('ascii')
        body = {
            'Data': data,
            'Name': name,
            'Labels': labels
        }

        if driver is not None:
            if utils.version_lt(self._version, '1.31'):
                raise errors.InvalidVersion(
                    'Secret driver is only available for API version > 1.31'
                )

            body['Driver'] = driver

        url = self._url('/secrets/create')
        return self._result(
            self._post_json(url, data=body), True
        )

    @utils.minimum_version('1.25')
    @utils.check_resource('id')
    def inspect_secret(self, id):
        """
            Retrieve secret metadata

            Args:
                id (string): Full ID of the secret to inspect

            Returns (dict): A dictionary of metadata

            Raises:
                :py:class:`docker.errors.NotFound`
                    if no secret with that ID exists
        """
        url = self._url('/secrets/{0}', id)
        return self._result(self._get(url), True)

    @utils.minimum_version('1.25')
    @utils.check_resource('id')
    def remove_secret(self, id):
        """
            Remove a secret

            Args:
                id (string): Full ID of the secret to remove

            Returns (boolean): True if successful

            Raises:
                :py:class:`docker.errors.NotFound`
                    if no secret with that ID exists
        """
        url = self._url('/secrets/{0}', id)
        res = self._delete(url)
        self._raise_for_status(res)
        return True

    @utils.minimum_version('1.25')
    def secrets(self, filters=None):
        """
            List secrets

            Args:
                filters (dict): A map of filters to process on the secrets
                list. Available filters: ``names``

            Returns (list): A list of secrets
        """
        url = self._url('/secrets')
        params = {}
        if filters:
            params['filters'] = utils.convert_filters(filters)
        return self._result(self._get(url, params=params), True)
