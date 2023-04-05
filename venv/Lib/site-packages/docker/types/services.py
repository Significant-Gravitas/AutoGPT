from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
    check_resource, format_environment, format_extra_hosts, parse_bytes,
    split_command, convert_service_networks,
)


class TaskTemplate(dict):
    """
    Describe the task specification to be used when creating or updating a
    service.

    Args:

        container_spec (ContainerSpec): Container settings for containers
          started as part of this task.
        log_driver (DriverConfig): Log configuration for containers created as
          part of the service.
        resources (Resources): Resource requirements which apply to each
          individual container created as part of the service.
        restart_policy (RestartPolicy): Specification for the restart policy
          which applies to containers created as part of this service.
        placement (Placement): Placement instructions for the scheduler.
            If a list is passed instead, it is assumed to be a list of
            constraints as part of a :py:class:`Placement` object.
        networks (:py:class:`list`): List of network names or IDs or
            :py:class:`NetworkAttachmentConfig` to attach the service to.
        force_update (int): A counter that triggers an update even if no
            relevant parameters have been changed.
    """

    def __init__(self, container_spec, resources=None, restart_policy=None,
                 placement=None, log_driver=None, networks=None,
                 force_update=None):
        self['ContainerSpec'] = container_spec
        if resources:
            self['Resources'] = resources
        if restart_policy:
            self['RestartPolicy'] = restart_policy
        if placement:
            if isinstance(placement, list):
                placement = Placement(constraints=placement)
            self['Placement'] = placement
        if log_driver:
            self['LogDriver'] = log_driver
        if networks:
            self['Networks'] = convert_service_networks(networks)

        if force_update is not None:
            if not isinstance(force_update, int):
                raise TypeError('force_update must be an integer')
            self['ForceUpdate'] = force_update

    @property
    def container_spec(self):
        return self.get('ContainerSpec')

    @property
    def resources(self):
        return self.get('Resources')

    @property
    def restart_policy(self):
        return self.get('RestartPolicy')

    @property
    def placement(self):
        return self.get('Placement')


class ContainerSpec(dict):
    """
    Describes the behavior of containers that are part of a task, and is used
    when declaring a :py:class:`~docker.types.TaskTemplate`.

    Args:

        image (string): The image name to use for the container.
        command (string or list):  The command to be run in the image.
        args (:py:class:`list`): Arguments to the command.
        hostname (string): The hostname to set on the container.
        env (dict): Environment variables.
        workdir (string): The working directory for commands to run in.
        user (string): The user inside the container.
        labels (dict): A map of labels to associate with the service.
        mounts (:py:class:`list`): A list of specifications for mounts to be
            added to containers created as part of the service. See the
            :py:class:`~docker.types.Mount` class for details.
        stop_grace_period (int): Amount of time to wait for the container to
            terminate before forcefully killing it.
        secrets (:py:class:`list`): List of :py:class:`SecretReference` to be
            made available inside the containers.
        tty (boolean): Whether a pseudo-TTY should be allocated.
        groups (:py:class:`list`): A list of additional groups that the
            container process will run as.
        open_stdin (boolean): Open ``stdin``
        read_only (boolean): Mount the container's root filesystem as read
            only.
        stop_signal (string): Set signal to stop the service's containers
        healthcheck (Healthcheck): Healthcheck
            configuration for this service.
        hosts (:py:class:`dict`): A set of host to IP mappings to add to
            the container's ``hosts`` file.
        dns_config (DNSConfig): Specification for DNS
            related configurations in resolver configuration file.
        configs (:py:class:`list`): List of :py:class:`ConfigReference` that
            will be exposed to the service.
        privileges (Privileges): Security options for the service's containers.
        isolation (string): Isolation technology used by the service's
            containers. Only used for Windows containers.
        init (boolean): Run an init inside the container that forwards signals
            and reaps processes.
        cap_add (:py:class:`list`): A list of kernel capabilities to add to the
            default set for the container.
        cap_drop (:py:class:`list`): A list of kernel capabilities to drop from
            the default set for the container.
        sysctls (:py:class:`dict`): A dict of sysctl values to add to
            the container
    """

    def __init__(self, image, command=None, args=None, hostname=None, env=None,
                 workdir=None, user=None, labels=None, mounts=None,
                 stop_grace_period=None, secrets=None, tty=None, groups=None,
                 open_stdin=None, read_only=None, stop_signal=None,
                 healthcheck=None, hosts=None, dns_config=None, configs=None,
                 privileges=None, isolation=None, init=None, cap_add=None,
                 cap_drop=None, sysctls=None):
        self['Image'] = image

        if isinstance(command, str):
            command = split_command(command)
        self['Command'] = command
        self['Args'] = args

        if hostname is not None:
            self['Hostname'] = hostname
        if env is not None:
            if isinstance(env, dict):
                self['Env'] = format_environment(env)
            else:
                self['Env'] = env
        if workdir is not None:
            self['Dir'] = workdir
        if user is not None:
            self['User'] = user
        if groups is not None:
            self['Groups'] = groups
        if stop_signal is not None:
            self['StopSignal'] = stop_signal
        if stop_grace_period is not None:
            self['StopGracePeriod'] = stop_grace_period
        if labels is not None:
            self['Labels'] = labels
        if hosts is not None:
            self['Hosts'] = format_extra_hosts(hosts, task=True)

        if mounts is not None:
            parsed_mounts = []
            for mount in mounts:
                if isinstance(mount, str):
                    parsed_mounts.append(Mount.parse_mount_string(mount))
                else:
                    # If mount already parsed
                    parsed_mounts.append(mount)
            self['Mounts'] = parsed_mounts

        if secrets is not None:
            if not isinstance(secrets, list):
                raise TypeError('secrets must be a list')
            self['Secrets'] = secrets

        if configs is not None:
            if not isinstance(configs, list):
                raise TypeError('configs must be a list')
            self['Configs'] = configs

        if dns_config is not None:
            self['DNSConfig'] = dns_config
        if privileges is not None:
            self['Privileges'] = privileges
        if healthcheck is not None:
            self['Healthcheck'] = healthcheck

        if tty is not None:
            self['TTY'] = tty
        if open_stdin is not None:
            self['OpenStdin'] = open_stdin
        if read_only is not None:
            self['ReadOnly'] = read_only

        if isolation is not None:
            self['Isolation'] = isolation

        if init is not None:
            self['Init'] = init

        if cap_add is not None:
            if not isinstance(cap_add, list):
                raise TypeError('cap_add must be a list')

            self['CapabilityAdd'] = cap_add

        if cap_drop is not None:
            if not isinstance(cap_drop, list):
                raise TypeError('cap_drop must be a list')

            self['CapabilityDrop'] = cap_drop

        if sysctls is not None:
            if not isinstance(sysctls, dict):
                raise TypeError('sysctls must be a dict')

            self['Sysctls'] = sysctls


class Mount(dict):
    """
    Describes a mounted folder's configuration inside a container. A list of
    :py:class:`Mount` would be used as part of a
    :py:class:`~docker.types.ContainerSpec`.

    Args:

        target (string): Container path.
        source (string): Mount source (e.g. a volume name or a host path).
        type (string): The mount type (``bind`` / ``volume`` / ``tmpfs`` /
            ``npipe``). Default: ``volume``.
        read_only (bool): Whether the mount should be read-only.
        consistency (string): The consistency requirement for the mount. One of
        ``default```, ``consistent``, ``cached``, ``delegated``.
        propagation (string): A propagation mode with the value ``[r]private``,
          ``[r]shared``, or ``[r]slave``. Only valid for the ``bind`` type.
        no_copy (bool): False if the volume should be populated with the data
          from the target. Default: ``False``. Only valid for the ``volume``
          type.
        labels (dict): User-defined name and labels for the volume. Only valid
          for the ``volume`` type.
        driver_config (DriverConfig): Volume driver configuration. Only valid
          for the ``volume`` type.
        tmpfs_size (int or string): The size for the tmpfs mount in bytes.
        tmpfs_mode (int): The permission mode for the tmpfs mount.
    """

    def __init__(self, target, source, type='volume', read_only=False,
                 consistency=None, propagation=None, no_copy=False,
                 labels=None, driver_config=None, tmpfs_size=None,
                 tmpfs_mode=None):
        self['Target'] = target
        self['Source'] = source
        if type not in ('bind', 'volume', 'tmpfs', 'npipe'):
            raise errors.InvalidArgument(
                f'Unsupported mount type: "{type}"'
            )
        self['Type'] = type
        self['ReadOnly'] = read_only

        if consistency:
            self['Consistency'] = consistency

        if type == 'bind':
            if propagation is not None:
                self['BindOptions'] = {
                    'Propagation': propagation
                }
            if any([labels, driver_config, no_copy, tmpfs_size, tmpfs_mode]):
                raise errors.InvalidArgument(
                    'Incompatible options have been provided for the bind '
                    'type mount.'
                )
        elif type == 'volume':
            volume_opts = {}
            if no_copy:
                volume_opts['NoCopy'] = True
            if labels:
                volume_opts['Labels'] = labels
            if driver_config:
                volume_opts['DriverConfig'] = driver_config
            if volume_opts:
                self['VolumeOptions'] = volume_opts
            if any([propagation, tmpfs_size, tmpfs_mode]):
                raise errors.InvalidArgument(
                    'Incompatible options have been provided for the volume '
                    'type mount.'
                )
        elif type == 'tmpfs':
            tmpfs_opts = {}
            if tmpfs_mode:
                if not isinstance(tmpfs_mode, int):
                    raise errors.InvalidArgument(
                        'tmpfs_mode must be an integer'
                    )
                tmpfs_opts['Mode'] = tmpfs_mode
            if tmpfs_size:
                tmpfs_opts['SizeBytes'] = parse_bytes(tmpfs_size)
            if tmpfs_opts:
                self['TmpfsOptions'] = tmpfs_opts
            if any([propagation, labels, driver_config, no_copy]):
                raise errors.InvalidArgument(
                    'Incompatible options have been provided for the tmpfs '
                    'type mount.'
                )

    @classmethod
    def parse_mount_string(cls, string):
        parts = string.split(':')
        if len(parts) > 3:
            raise errors.InvalidArgument(
                f'Invalid mount format "{string}"'
            )
        if len(parts) == 1:
            return cls(target=parts[0], source=None)
        else:
            target = parts[1]
            source = parts[0]
            mount_type = 'volume'
            if source.startswith('/') or (
                IS_WINDOWS_PLATFORM and source[0].isalpha() and
                source[1] == ':'
            ):
                # FIXME: That windows condition will fail earlier since we
                # split on ':'. We should look into doing a smarter split
                # if we detect we are on Windows.
                mount_type = 'bind'
            read_only = not (len(parts) == 2 or parts[2] == 'rw')
            return cls(target, source, read_only=read_only, type=mount_type)


class Resources(dict):
    """
    Configures resource allocation for containers when made part of a
    :py:class:`~docker.types.ContainerSpec`.

    Args:

        cpu_limit (int): CPU limit in units of 10^9 CPU shares.
        mem_limit (int): Memory limit in Bytes.
        cpu_reservation (int): CPU reservation in units of 10^9 CPU shares.
        mem_reservation (int): Memory reservation in Bytes.
        generic_resources (dict or :py:class:`list`): Node level generic
          resources, for example a GPU, using the following format:
          ``{ resource_name: resource_value }``. Alternatively, a list of
          of resource specifications as defined by the Engine API.
    """

    def __init__(self, cpu_limit=None, mem_limit=None, cpu_reservation=None,
                 mem_reservation=None, generic_resources=None):
        limits = {}
        reservation = {}
        if cpu_limit is not None:
            limits['NanoCPUs'] = cpu_limit
        if mem_limit is not None:
            limits['MemoryBytes'] = mem_limit
        if cpu_reservation is not None:
            reservation['NanoCPUs'] = cpu_reservation
        if mem_reservation is not None:
            reservation['MemoryBytes'] = mem_reservation
        if generic_resources is not None:
            reservation['GenericResources'] = (
                _convert_generic_resources_dict(generic_resources)
            )
        if limits:
            self['Limits'] = limits
        if reservation:
            self['Reservations'] = reservation


def _convert_generic_resources_dict(generic_resources):
    if isinstance(generic_resources, list):
        return generic_resources
    if not isinstance(generic_resources, dict):
        raise errors.InvalidArgument(
            'generic_resources must be a dict or a list'
            ' (found {})'.format(type(generic_resources))
        )
    resources = []
    for kind, value in generic_resources.items():
        resource_type = None
        if isinstance(value, int):
            resource_type = 'DiscreteResourceSpec'
        elif isinstance(value, str):
            resource_type = 'NamedResourceSpec'
        else:
            raise errors.InvalidArgument(
                'Unsupported generic resource reservation '
                'type: {}'.format({kind: value})
            )
        resources.append({
            resource_type: {'Kind': kind, 'Value': value}
        })
    return resources


class UpdateConfig(dict):
    """

    Used to specify the way container updates should be performed by a service.

    Args:

        parallelism (int): Maximum number of tasks to be updated in one
          iteration (0 means unlimited parallelism). Default: 0.
        delay (int): Amount of time between updates, in nanoseconds.
        failure_action (string): Action to take if an updated task fails to
          run, or stops running during the update. Acceptable values are
          ``continue``, ``pause``, as well as ``rollback`` since API v1.28.
          Default: ``continue``
        monitor (int): Amount of time to monitor each updated task for
          failures, in nanoseconds.
        max_failure_ratio (float): The fraction of tasks that may fail during
          an update before the failure action is invoked, specified as a
          floating point number between 0 and 1. Default: 0
        order (string): Specifies the order of operations when rolling out an
          updated task. Either ``start-first`` or ``stop-first`` are accepted.
    """

    def __init__(self, parallelism=0, delay=None, failure_action='continue',
                 monitor=None, max_failure_ratio=None, order=None):
        self['Parallelism'] = parallelism
        if delay is not None:
            self['Delay'] = delay
        if failure_action not in ('pause', 'continue', 'rollback'):
            raise errors.InvalidArgument(
                'failure_action must be one of `pause`, `continue`, `rollback`'
            )
        self['FailureAction'] = failure_action

        if monitor is not None:
            if not isinstance(monitor, int):
                raise TypeError('monitor must be an integer')
            self['Monitor'] = monitor

        if max_failure_ratio is not None:
            if not isinstance(max_failure_ratio, (float, int)):
                raise TypeError('max_failure_ratio must be a float')
            if max_failure_ratio > 1 or max_failure_ratio < 0:
                raise errors.InvalidArgument(
                    'max_failure_ratio must be a number between 0 and 1'
                )
            self['MaxFailureRatio'] = max_failure_ratio

        if order is not None:
            if order not in ('start-first', 'stop-first'):
                raise errors.InvalidArgument(
                    'order must be either `start-first` or `stop-first`'
                )
            self['Order'] = order


class RollbackConfig(UpdateConfig):
    """
    Used to specify the way container rollbacks should be performed by a
    service

    Args:
        parallelism (int): Maximum number of tasks to be rolled back in one
          iteration (0 means unlimited parallelism). Default: 0
        delay (int): Amount of time between rollbacks, in nanoseconds.
        failure_action (string): Action to take if a rolled back task fails to
          run, or stops running during the rollback. Acceptable values are
          ``continue``, ``pause`` or ``rollback``.
          Default: ``continue``
        monitor (int): Amount of time to monitor each rolled back task for
          failures, in nanoseconds.
        max_failure_ratio (float): The fraction of tasks that may fail during
          a rollback before the failure action is invoked, specified as a
          floating point number between 0 and 1. Default: 0
        order (string): Specifies the order of operations when rolling out a
          rolled back task. Either ``start-first`` or ``stop-first`` are
          accepted.
    """
    pass


class RestartConditionTypesEnum:
    _values = (
        'none',
        'on-failure',
        'any',
    )
    NONE, ON_FAILURE, ANY = _values


class RestartPolicy(dict):
    """
    Used when creating a :py:class:`~docker.types.ContainerSpec`,
    dictates whether a container should restart after stopping or failing.

    Args:

        condition (string): Condition for restart (``none``, ``on-failure``,
          or ``any``). Default: `none`.
        delay (int): Delay between restart attempts. Default: 0
        max_attempts (int): Maximum attempts to restart a given container
          before giving up. Default value is 0, which is ignored.
        window (int): Time window used to evaluate the restart policy. Default
          value is 0, which is unbounded.
    """

    condition_types = RestartConditionTypesEnum

    def __init__(self, condition=RestartConditionTypesEnum.NONE, delay=0,
                 max_attempts=0, window=0):
        if condition not in self.condition_types._values:
            raise TypeError(
                f'Invalid RestartPolicy condition {condition}'
            )

        self['Condition'] = condition
        self['Delay'] = delay
        self['MaxAttempts'] = max_attempts
        self['Window'] = window


class DriverConfig(dict):
    """
    Indicates which driver to use, as well as its configuration. Can be used
    as ``log_driver`` in a :py:class:`~docker.types.ContainerSpec`,
    for the `driver_config` in a volume :py:class:`~docker.types.Mount`, or
    as the driver object in
    :py:meth:`create_secret`.

    Args:

        name (string): Name of the driver to use.
        options (dict): Driver-specific options. Default: ``None``.
    """

    def __init__(self, name, options=None):
        self['Name'] = name
        if options:
            self['Options'] = options


class EndpointSpec(dict):
    """
    Describes properties to access and load-balance a service.

    Args:

        mode (string): The mode of resolution to use for internal load
          balancing between tasks (``'vip'`` or ``'dnsrr'``). Defaults to
          ``'vip'`` if not provided.
        ports (dict): Exposed ports that this service is accessible on from the
          outside, in the form of ``{ published_port: target_port }`` or
          ``{ published_port: <port_config_tuple> }``. Port config tuple format
          is ``(target_port [, protocol [, publish_mode]])``.
          Ports can only be provided if the ``vip`` resolution mode is used.
    """

    def __init__(self, mode=None, ports=None):
        if ports:
            self['Ports'] = convert_service_ports(ports)
        if mode:
            self['Mode'] = mode


def convert_service_ports(ports):
    if isinstance(ports, list):
        return ports
    if not isinstance(ports, dict):
        raise TypeError(
            'Invalid type for ports, expected dict or list'
        )

    result = []
    for k, v in ports.items():
        port_spec = {
            'Protocol': 'tcp',
            'PublishedPort': k
        }

        if isinstance(v, tuple):
            port_spec['TargetPort'] = v[0]
            if len(v) >= 2 and v[1] is not None:
                port_spec['Protocol'] = v[1]
            if len(v) == 3:
                port_spec['PublishMode'] = v[2]
            if len(v) > 3:
                raise ValueError(
                    'Service port configuration can have at most 3 elements: '
                    '(target_port, protocol, mode)'
                )
        else:
            port_spec['TargetPort'] = v

        result.append(port_spec)
    return result


class ServiceMode(dict):
    """
        Indicate whether a service or a job should be deployed as a replicated
        or global service, and associated parameters

        Args:
            mode (string): Can be either ``replicated``, ``global``,
              ``replicated-job`` or ``global-job``
            replicas (int): Number of replicas. For replicated services only.
            concurrency (int): Number of concurrent jobs. For replicated job
              services only.
    """

    def __init__(self, mode, replicas=None, concurrency=None):
        replicated_modes = ('replicated', 'replicated-job')
        supported_modes = replicated_modes + ('global', 'global-job')

        if mode not in supported_modes:
            raise errors.InvalidArgument(
                'mode must be either "replicated", "global", "replicated-job"'
                ' or "global-job"'
            )

        if mode not in replicated_modes:
            if replicas is not None:
                raise errors.InvalidArgument(
                    'replicas can only be used for "replicated" or'
                    ' "replicated-job" mode'
                )

            if concurrency is not None:
                raise errors.InvalidArgument(
                    'concurrency can only be used for "replicated-job" mode'
                )

        service_mode = self._convert_mode(mode)
        self.mode = service_mode
        self[service_mode] = {}

        if replicas is not None:
            if mode == 'replicated':
                self[service_mode]['Replicas'] = replicas

            if mode == 'replicated-job':
                self[service_mode]['MaxConcurrent'] = concurrency or 1
                self[service_mode]['TotalCompletions'] = replicas

    @staticmethod
    def _convert_mode(original_mode):
        if original_mode == 'global-job':
            return 'GlobalJob'

        if original_mode == 'replicated-job':
            return 'ReplicatedJob'

        return original_mode

    @property
    def replicas(self):
        if 'replicated' in self:
            return self['replicated'].get('Replicas')

        if 'ReplicatedJob' in self:
            return self['ReplicatedJob'].get('TotalCompletions')

        return None


class SecretReference(dict):
    """
        Secret reference to be used as part of a :py:class:`ContainerSpec`.
        Describes how a secret is made accessible inside the service's
        containers.

        Args:
            secret_id (string): Secret's ID
            secret_name (string): Secret's name as defined at its creation.
            filename (string): Name of the file containing the secret. Defaults
                to the secret's name if not specified.
            uid (string): UID of the secret file's owner. Default: 0
            gid (string): GID of the secret file's group. Default: 0
            mode (int): File access mode inside the container. Default: 0o444
    """
    @check_resource('secret_id')
    def __init__(self, secret_id, secret_name, filename=None, uid=None,
                 gid=None, mode=0o444):
        self['SecretName'] = secret_name
        self['SecretID'] = secret_id
        self['File'] = {
            'Name': filename or secret_name,
            'UID': uid or '0',
            'GID': gid or '0',
            'Mode': mode
        }


class ConfigReference(dict):
    """
        Config reference to be used as part of a :py:class:`ContainerSpec`.
        Describes how a config is made accessible inside the service's
        containers.

        Args:
            config_id (string): Config's ID
            config_name (string): Config's name as defined at its creation.
            filename (string): Name of the file containing the config. Defaults
                to the config's name if not specified.
            uid (string): UID of the config file's owner. Default: 0
            gid (string): GID of the config file's group. Default: 0
            mode (int): File access mode inside the container. Default: 0o444
    """
    @check_resource('config_id')
    def __init__(self, config_id, config_name, filename=None, uid=None,
                 gid=None, mode=0o444):
        self['ConfigName'] = config_name
        self['ConfigID'] = config_id
        self['File'] = {
            'Name': filename or config_name,
            'UID': uid or '0',
            'GID': gid or '0',
            'Mode': mode
        }


class Placement(dict):
    """
        Placement constraints to be used as part of a :py:class:`TaskTemplate`

        Args:
            constraints (:py:class:`list` of str): A list of constraints
            preferences (:py:class:`list` of tuple): Preferences provide a way
                to make the scheduler aware of factors such as topology. They
                are provided in order from highest to lowest precedence and
                are expressed as ``(strategy, descriptor)`` tuples. See
                :py:class:`PlacementPreference` for details.
            maxreplicas (int): Maximum number of replicas per node
            platforms (:py:class:`list` of tuple): A list of platforms
                expressed as ``(arch, os)`` tuples
    """

    def __init__(self, constraints=None, preferences=None, platforms=None,
                 maxreplicas=None):
        if constraints is not None:
            self['Constraints'] = constraints
        if preferences is not None:
            self['Preferences'] = []
            for pref in preferences:
                if isinstance(pref, tuple):
                    pref = PlacementPreference(*pref)
                self['Preferences'].append(pref)
        if maxreplicas is not None:
            self['MaxReplicas'] = maxreplicas
        if platforms:
            self['Platforms'] = []
            for plat in platforms:
                self['Platforms'].append({
                    'Architecture': plat[0], 'OS': plat[1]
                })


class PlacementPreference(dict):
    """
        Placement preference to be used as an element in the list of
        preferences for :py:class:`Placement` objects.

        Args:
            strategy (string): The placement strategy to implement. Currently,
                the only supported strategy is ``spread``.
            descriptor (string): A label descriptor. For the spread strategy,
                the scheduler will try to spread tasks evenly over groups of
                nodes identified by this label.
    """

    def __init__(self, strategy, descriptor):
        if strategy != 'spread':
            raise errors.InvalidArgument(
                'PlacementPreference strategy value is invalid ({}):'
                ' must be "spread".'.format(strategy)
            )
        self['Spread'] = {'SpreadDescriptor': descriptor}


class DNSConfig(dict):
    """
        Specification for DNS related configurations in resolver configuration
        file (``resolv.conf``). Part of a :py:class:`ContainerSpec` definition.

        Args:
            nameservers (:py:class:`list`): The IP addresses of the name
                servers.
            search (:py:class:`list`): A search list for host-name lookup.
            options (:py:class:`list`): A list of internal resolver variables
                to be modified (e.g., ``debug``, ``ndots:3``, etc.).
    """

    def __init__(self, nameservers=None, search=None, options=None):
        self['Nameservers'] = nameservers
        self['Search'] = search
        self['Options'] = options


class Privileges(dict):
    r"""
        Security options for a service's containers.
        Part of a :py:class:`ContainerSpec` definition.

        Args:
            credentialspec_file (str): Load credential spec from this file.
                The file is read by the daemon, and must be present in the
                CredentialSpecs subdirectory in the docker data directory,
                which defaults to ``C:\ProgramData\Docker\`` on Windows.
                Can not be combined with credentialspec_registry.

            credentialspec_registry (str): Load credential spec from this value
                in the Windows registry. The specified registry value must be
                located in: ``HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion
                \Virtualization\Containers\CredentialSpecs``.
                Can not be combined with credentialspec_file.

            selinux_disable (boolean): Disable SELinux
            selinux_user (string): SELinux user label
            selinux_role (string): SELinux role label
            selinux_type (string): SELinux type label
            selinux_level (string): SELinux level label
    """

    def __init__(self, credentialspec_file=None, credentialspec_registry=None,
                 selinux_disable=None, selinux_user=None, selinux_role=None,
                 selinux_type=None, selinux_level=None):
        credential_spec = {}
        if credentialspec_registry is not None:
            credential_spec['Registry'] = credentialspec_registry
        if credentialspec_file is not None:
            credential_spec['File'] = credentialspec_file

        if len(credential_spec) > 1:
            raise errors.InvalidArgument(
                'credentialspec_file and credentialspec_registry are mutually'
                ' exclusive'
            )

        selinux_context = {
            'Disable': selinux_disable,
            'User': selinux_user,
            'Role': selinux_role,
            'Type': selinux_type,
            'Level': selinux_level,
        }

        if len(credential_spec) > 0:
            self['CredentialSpec'] = credential_spec

        if len(selinux_context) > 0:
            self['SELinuxContext'] = selinux_context


class NetworkAttachmentConfig(dict):
    """
        Network attachment options for a service.

        Args:
            target (str): The target network for attachment.
                Can be a network name or ID.
            aliases (:py:class:`list`): A list of discoverable alternate names
                for the service.
            options (:py:class:`dict`): Driver attachment options for the
                network target.
    """

    def __init__(self, target, aliases=None, options=None):
        self['Target'] = target
        self['Aliases'] = aliases
        self['DriverOpts'] = options
