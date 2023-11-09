import base64

from .. import utils


class ConfigApiMixin:
    @utils.minimum_version('1.30')
    def create_config(self, name, data, labels=None, templating=None):
        """
            Create a config

            Args:
                name (string): Name of the config
                data (bytes): Config data to be stored
                labels (dict): A mapping of labels to assign to the config
                templating (dict): dictionary containing the name of the
                                   templating driver to be used expressed as
                                   { name: <templating_driver_name>}

            Returns (dict): ID of the newly created config
        """
        if not isinstance(data, bytes):
            data = data.encode('utf-8')

        data = base64.b64encode(data)
        data = data.decode('ascii')
        body = {
            'Data': data,
            'Name': name,
            'Labels': labels,
            'Templating': templating
        }

        url = self._url('/configs/create')
        return self._result(
            self._post_json(url, data=body), True
        )

    @utils.minimum_version('1.30')
    @utils.check_resource('id')
    def inspect_config(self, id):
        """
            Retrieve config metadata

            Args:
                id (string): Full ID of the config to inspect

            Returns (dict): A dictionary of metadata

            Raises:
                :py:class:`docker.errors.NotFound`
                    if no config with that ID exists
        """
        url = self._url('/configs/{0}', id)
        return self._result(self._get(url), True)

    @utils.minimum_version('1.30')
    @utils.check_resource('id')
    def remove_config(self, id):
        """
            Remove a config

            Args:
                id (string): Full ID of the config to remove

            Returns (boolean): True if successful

            Raises:
                :py:class:`docker.errors.NotFound`
                    if no config with that ID exists
        """
        url = self._url('/configs/{0}', id)
        res = self._delete(url)
        self._raise_for_status(res)
        return True

    @utils.minimum_version('1.30')
    def configs(self, filters=None):
        """
            List configs

            Args:
                filters (dict): A map of filters to process on the configs
                list. Available filters: ``names``

            Returns (list): A list of configs
        """
        url = self._url('/configs')
        params = {}
        if filters:
            params['filters'] = utils.convert_filters(filters)
        return self._result(self._get(url, params=params), True)
