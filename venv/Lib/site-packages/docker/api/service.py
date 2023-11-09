from .. import auth, errors, utils
from ..types import ServiceMode


def _check_api_features(version, task_template, update_config, endpoint_spec,
                        rollback_config):

    def raise_version_error(param, min_version):
        raise errors.InvalidVersion(
            '{} is not supported in API version < {}'.format(
                param, min_version
            )
        )

    if update_config is not None:
        if utils.version_lt(version, '1.25'):
            if 'MaxFailureRatio' in update_config:
                raise_version_error('UpdateConfig.max_failure_ratio', '1.25')
            if 'Monitor' in update_config:
                raise_version_error('UpdateConfig.monitor', '1.25')

        if utils.version_lt(version, '1.28'):
            if update_config.get('FailureAction') == 'rollback':
                raise_version_error(
                    'UpdateConfig.failure_action rollback', '1.28'
                )

        if utils.version_lt(version, '1.29'):
            if 'Order' in update_config:
                raise_version_error('UpdateConfig.order', '1.29')

    if rollback_config is not None:
        if utils.version_lt(version, '1.28'):
            raise_version_error('rollback_config', '1.28')

        if utils.version_lt(version, '1.29'):
            if 'Order' in update_config:
                raise_version_error('RollbackConfig.order', '1.29')

    if endpoint_spec is not None:
        if utils.version_lt(version, '1.32') and 'Ports' in endpoint_spec:
            if any(p.get('PublishMode') for p in endpoint_spec['Ports']):
                raise_version_error('EndpointSpec.Ports[].mode', '1.32')

    if task_template is not None:
        if 'ForceUpdate' in task_template and utils.version_lt(
                version, '1.25'):
            raise_version_error('force_update', '1.25')

        if task_template.get('Placement'):
            if utils.version_lt(version, '1.30'):
                if task_template['Placement'].get('Platforms'):
                    raise_version_error('Placement.platforms', '1.30')
            if utils.version_lt(version, '1.27'):
                if task_template['Placement'].get('Preferences'):
                    raise_version_error('Placement.preferences', '1.27')

        if task_template.get('ContainerSpec'):
            container_spec = task_template.get('ContainerSpec')

            if utils.version_lt(version, '1.25'):
                if container_spec.get('TTY'):
                    raise_version_error('ContainerSpec.tty', '1.25')
                if container_spec.get('Hostname') is not None:
                    raise_version_error('ContainerSpec.hostname', '1.25')
                if container_spec.get('Hosts') is not None:
                    raise_version_error('ContainerSpec.hosts', '1.25')
                if container_spec.get('Groups') is not None:
                    raise_version_error('ContainerSpec.groups', '1.25')
                if container_spec.get('DNSConfig') is not None:
                    raise_version_error('ContainerSpec.dns_config', '1.25')
                if container_spec.get('Healthcheck') is not None:
                    raise_version_error('ContainerSpec.healthcheck', '1.25')

            if utils.version_lt(version, '1.28'):
                if container_spec.get('ReadOnly') is not None:
                    raise_version_error('ContainerSpec.dns_config', '1.28')
                if container_spec.get('StopSignal') is not None:
                    raise_version_error('ContainerSpec.stop_signal', '1.28')

            if utils.version_lt(version, '1.30'):
                if container_spec.get('Configs') is not None:
                    raise_version_error('ContainerSpec.configs', '1.30')
                if container_spec.get('Privileges') is not None:
                    raise_version_error('ContainerSpec.privileges', '1.30')

            if utils.version_lt(version, '1.35'):
                if container_spec.get('Isolation') is not None:
                    raise_version_error('ContainerSpec.isolation', '1.35')

            if utils.version_lt(version, '1.38'):
                if container_spec.get('Init') is not None:
                    raise_version_error('ContainerSpec.init', '1.38')

        if task_template.get('Resources'):
            if utils.version_lt(version, '1.32'):
                if task_template['Resources'].get('GenericResources'):
                    raise_version_error('Resources.generic_resources', '1.32')


def _merge_task_template(current, override):
    merged = current.copy()
    if override is not None:
        for ts_key, ts_value in override.items():
            if ts_key == 'ContainerSpec':
                if 'ContainerSpec' not in merged:
                    merged['ContainerSpec'] = {}
                for cs_key, cs_value in override['ContainerSpec'].items():
                    if cs_value is not None:
                        merged['ContainerSpec'][cs_key] = cs_value
            elif ts_value is not None:
                merged[ts_key] = ts_value
    return merged


class ServiceApiMixin:
    @utils.minimum_version('1.24')
    def create_service(
            self, task_template, name=None, labels=None, mode=None,
            update_config=None, networks=None, endpoint_config=None,
            endpoint_spec=None, rollback_config=None
    ):
        """
        Create a service.

        Args:
            task_template (TaskTemplate): Specification of the task to start as
                part of the new service.
            name (string): User-defined name for the service. Optional.
            labels (dict): A map of labels to associate with the service.
                Optional.
            mode (ServiceMode): Scheduling mode for the service (replicated
                or global). Defaults to replicated.
            update_config (UpdateConfig): Specification for the update strategy
                of the service. Default: ``None``
            rollback_config (RollbackConfig): Specification for the rollback
                strategy of the service. Default: ``None``
            networks (:py:class:`list`): List of network names or IDs or
                :py:class:`~docker.types.NetworkAttachmentConfig` to attach the
                service to. Default: ``None``.
            endpoint_spec (EndpointSpec): Properties that can be configured to
                access and load balance a service. Default: ``None``.

        Returns:
            A dictionary containing an ``ID`` key for the newly created
            service.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """

        _check_api_features(
            self._version, task_template, update_config, endpoint_spec,
            rollback_config
        )

        url = self._url('/services/create')
        headers = {}
        image = task_template.get('ContainerSpec', {}).get('Image', None)
        if image is None:
            raise errors.DockerException(
                'Missing mandatory Image key in ContainerSpec'
            )
        if mode and not isinstance(mode, dict):
            mode = ServiceMode(mode)

        registry, repo_name = auth.resolve_repository_name(image)
        auth_header = auth.get_config_header(self, registry)
        if auth_header:
            headers['X-Registry-Auth'] = auth_header
        if utils.version_lt(self._version, '1.25'):
            networks = networks or task_template.pop('Networks', None)
        data = {
            'Name': name,
            'Labels': labels,
            'TaskTemplate': task_template,
            'Mode': mode,
            'Networks': utils.convert_service_networks(networks),
            'EndpointSpec': endpoint_spec
        }

        if update_config is not None:
            data['UpdateConfig'] = update_config

        if rollback_config is not None:
            data['RollbackConfig'] = rollback_config

        return self._result(
            self._post_json(url, data=data, headers=headers), True
        )

    @utils.minimum_version('1.24')
    @utils.check_resource('service')
    def inspect_service(self, service, insert_defaults=None):
        """
        Return information about a service.

        Args:
            service (str): Service name or ID.
            insert_defaults (boolean): If true, default values will be merged
                into the service inspect output.

        Returns:
            (dict): A dictionary of the server-side representation of the
                service, including all relevant properties.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        url = self._url('/services/{0}', service)
        params = {}
        if insert_defaults is not None:
            if utils.version_lt(self._version, '1.29'):
                raise errors.InvalidVersion(
                    'insert_defaults is not supported in API version < 1.29'
                )
            params['insertDefaults'] = insert_defaults

        return self._result(self._get(url, params=params), True)

    @utils.minimum_version('1.24')
    @utils.check_resource('task')
    def inspect_task(self, task):
        """
        Retrieve information about a task.

        Args:
            task (str): Task ID

        Returns:
            (dict): Information about the task.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        url = self._url('/tasks/{0}', task)
        return self._result(self._get(url), True)

    @utils.minimum_version('1.24')
    @utils.check_resource('service')
    def remove_service(self, service):
        """
        Stop and remove a service.

        Args:
            service (str): Service name or ID

        Returns:
            ``True`` if successful.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """

        url = self._url('/services/{0}', service)
        resp = self._delete(url)
        self._raise_for_status(resp)
        return True

    @utils.minimum_version('1.24')
    def services(self, filters=None):
        """
        List services.

        Args:
            filters (dict): Filters to process on the nodes list. Valid
                filters: ``id``, ``name`` , ``label`` and ``mode``.
                Default: ``None``.

        Returns:
            A list of dictionaries containing data about each service.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        params = {
            'filters': utils.convert_filters(filters) if filters else None
        }
        url = self._url('/services')
        return self._result(self._get(url, params=params), True)

    @utils.minimum_version('1.25')
    @utils.check_resource('service')
    def service_logs(self, service, details=False, follow=False, stdout=False,
                     stderr=False, since=0, timestamps=False, tail='all',
                     is_tty=None):
        """
            Get log stream for a service.
            Note: This endpoint works only for services with the ``json-file``
            or ``journald`` logging drivers.

            Args:
                service (str): ID or name of the service
                details (bool): Show extra details provided to logs.
                    Default: ``False``
                follow (bool): Keep connection open to read logs as they are
                    sent by the Engine. Default: ``False``
                stdout (bool): Return logs from ``stdout``. Default: ``False``
                stderr (bool): Return logs from ``stderr``. Default: ``False``
                since (int): UNIX timestamp for the logs staring point.
                    Default: 0
                timestamps (bool): Add timestamps to every log line.
                tail (string or int): Number of log lines to be returned,
                    counting from the current end of the logs. Specify an
                    integer or ``'all'`` to output all log lines.
                    Default: ``all``
                is_tty (bool): Whether the service's :py:class:`ContainerSpec`
                    enables the TTY option. If omitted, the method will query
                    the Engine for the information, causing an additional
                    roundtrip.

            Returns (generator): Logs for the service.
        """
        params = {
            'details': details,
            'follow': follow,
            'stdout': stdout,
            'stderr': stderr,
            'since': since,
            'timestamps': timestamps,
            'tail': tail
        }

        url = self._url('/services/{0}/logs', service)
        res = self._get(url, params=params, stream=True)
        if is_tty is None:
            is_tty = self.inspect_service(
                service
            )['Spec']['TaskTemplate']['ContainerSpec'].get('TTY', False)
        return self._get_result_tty(True, res, is_tty)

    @utils.minimum_version('1.24')
    def tasks(self, filters=None):
        """
        Retrieve a list of tasks.

        Args:
            filters (dict): A map of filters to process on the tasks list.
                Valid filters: ``id``, ``name``, ``service``, ``node``,
                ``label`` and ``desired-state``.

        Returns:
            (:py:class:`list`): List of task dictionaries.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """

        params = {
            'filters': utils.convert_filters(filters) if filters else None
        }
        url = self._url('/tasks')
        return self._result(self._get(url, params=params), True)

    @utils.minimum_version('1.24')
    @utils.check_resource('service')
    def update_service(self, service, version, task_template=None, name=None,
                       labels=None, mode=None, update_config=None,
                       networks=None, endpoint_config=None,
                       endpoint_spec=None, fetch_current_spec=False,
                       rollback_config=None):
        """
        Update a service.

        Args:
            service (string): A service identifier (either its name or service
                ID).
            version (int): The version number of the service object being
                updated. This is required to avoid conflicting writes.
            task_template (TaskTemplate): Specification of the updated task to
                start as part of the service.
            name (string): New name for the service. Optional.
            labels (dict): A map of labels to associate with the service.
                Optional.
            mode (ServiceMode): Scheduling mode for the service (replicated
                or global). Defaults to replicated.
            update_config (UpdateConfig): Specification for the update strategy
                of the service. Default: ``None``.
            rollback_config (RollbackConfig): Specification for the rollback
                strategy of the service. Default: ``None``
            networks (:py:class:`list`): List of network names or IDs or
                :py:class:`~docker.types.NetworkAttachmentConfig` to attach the
                service to. Default: ``None``.
            endpoint_spec (EndpointSpec): Properties that can be configured to
                access and load balance a service. Default: ``None``.
            fetch_current_spec (boolean): Use the undefined settings from the
                current specification of the service. Default: ``False``

        Returns:
            A dictionary containing a ``Warnings`` key.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """

        _check_api_features(
            self._version, task_template, update_config, endpoint_spec,
            rollback_config
        )

        if fetch_current_spec:
            inspect_defaults = True
            if utils.version_lt(self._version, '1.29'):
                inspect_defaults = None
            current = self.inspect_service(
                service, insert_defaults=inspect_defaults
            )['Spec']

        else:
            current = {}

        url = self._url('/services/{0}/update', service)
        data = {}
        headers = {}

        data['Name'] = current.get('Name') if name is None else name

        data['Labels'] = current.get('Labels') if labels is None else labels

        if mode is not None:
            if not isinstance(mode, dict):
                mode = ServiceMode(mode)
            data['Mode'] = mode
        else:
            data['Mode'] = current.get('Mode')

        data['TaskTemplate'] = _merge_task_template(
            current.get('TaskTemplate', {}), task_template
        )

        container_spec = data['TaskTemplate'].get('ContainerSpec', {})
        image = container_spec.get('Image', None)
        if image is not None:
            registry, repo_name = auth.resolve_repository_name(image)
            auth_header = auth.get_config_header(self, registry)
            if auth_header:
                headers['X-Registry-Auth'] = auth_header

        if update_config is not None:
            data['UpdateConfig'] = update_config
        else:
            data['UpdateConfig'] = current.get('UpdateConfig')

        if rollback_config is not None:
            data['RollbackConfig'] = rollback_config
        else:
            data['RollbackConfig'] = current.get('RollbackConfig')

        if networks is not None:
            converted_networks = utils.convert_service_networks(networks)
            if utils.version_lt(self._version, '1.25'):
                data['Networks'] = converted_networks
            else:
                data['TaskTemplate']['Networks'] = converted_networks
        elif utils.version_lt(self._version, '1.25'):
            data['Networks'] = current.get('Networks')
        elif data['TaskTemplate'].get('Networks') is None:
            current_task_template = current.get('TaskTemplate', {})
            current_networks = current_task_template.get('Networks')
            if current_networks is None:
                current_networks = current.get('Networks')
            if current_networks is not None:
                data['TaskTemplate']['Networks'] = current_networks

        if endpoint_spec is not None:
            data['EndpointSpec'] = endpoint_spec
        else:
            data['EndpointSpec'] = current.get('EndpointSpec')

        resp = self._post_json(
            url, data=data, params={'version': version}, headers=headers
        )
        return self._result(resp, json=True)
