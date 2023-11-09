from ..errors import InvalidVersion
from ..utils import version_lt


class SwarmSpec(dict):
    """
        Describe a Swarm's configuration and options. Use
        :py:meth:`~docker.api.swarm.SwarmApiMixin.create_swarm_spec`
        to instantiate.
    """
    def __init__(self, version, task_history_retention_limit=None,
                 snapshot_interval=None, keep_old_snapshots=None,
                 log_entries_for_slow_followers=None, heartbeat_tick=None,
                 election_tick=None, dispatcher_heartbeat_period=None,
                 node_cert_expiry=None, external_cas=None, name=None,
                 labels=None, signing_ca_cert=None, signing_ca_key=None,
                 ca_force_rotate=None, autolock_managers=None,
                 log_driver=None):
        if task_history_retention_limit is not None:
            self['Orchestration'] = {
                'TaskHistoryRetentionLimit': task_history_retention_limit
            }
        if any([snapshot_interval,
                keep_old_snapshots,
                log_entries_for_slow_followers,
                heartbeat_tick,
                election_tick]):
            self['Raft'] = {
                'SnapshotInterval': snapshot_interval,
                'KeepOldSnapshots': keep_old_snapshots,
                'LogEntriesForSlowFollowers': log_entries_for_slow_followers,
                'HeartbeatTick': heartbeat_tick,
                'ElectionTick': election_tick
            }

        if dispatcher_heartbeat_period:
            self['Dispatcher'] = {
                'HeartbeatPeriod': dispatcher_heartbeat_period
            }

        ca_config = {}
        if node_cert_expiry is not None:
            ca_config['NodeCertExpiry'] = node_cert_expiry
        if external_cas:
            if version_lt(version, '1.25'):
                if len(external_cas) > 1:
                    raise InvalidVersion(
                        'Support for multiple external CAs is not available '
                        'for API version < 1.25'
                    )
                ca_config['ExternalCA'] = external_cas[0]
            else:
                ca_config['ExternalCAs'] = external_cas
        if signing_ca_key:
            if version_lt(version, '1.30'):
                raise InvalidVersion(
                    'signing_ca_key is not supported in API version < 1.30'
                )
            ca_config['SigningCAKey'] = signing_ca_key
        if signing_ca_cert:
            if version_lt(version, '1.30'):
                raise InvalidVersion(
                    'signing_ca_cert is not supported in API version < 1.30'
                )
            ca_config['SigningCACert'] = signing_ca_cert
        if ca_force_rotate is not None:
            if version_lt(version, '1.30'):
                raise InvalidVersion(
                    'force_rotate is not supported in API version < 1.30'
                )
            ca_config['ForceRotate'] = ca_force_rotate
        if ca_config:
            self['CAConfig'] = ca_config

        if autolock_managers is not None:
            if version_lt(version, '1.25'):
                raise InvalidVersion(
                    'autolock_managers is not supported in API version < 1.25'
                )

            self['EncryptionConfig'] = {'AutoLockManagers': autolock_managers}

        if log_driver is not None:
            if version_lt(version, '1.25'):
                raise InvalidVersion(
                    'log_driver is not supported in API version < 1.25'
                )

            self['TaskDefaults'] = {'LogDriver': log_driver}

        if name is not None:
            self['Name'] = name
        if labels is not None:
            self['Labels'] = labels


class SwarmExternalCA(dict):
    """
        Configuration for forwarding signing requests to an external
        certificate authority.

        Args:
            url (string): URL where certificate signing requests should be
                sent.
            protocol (string): Protocol for communication with the external CA.
            options (dict): An object with key/value pairs that are interpreted
                as protocol-specific options for the external CA driver.
            ca_cert (string): The root CA certificate (in PEM format) this
                external CA uses to issue TLS certificates (assumed to be to
                the current swarm root CA certificate if not provided).



    """
    def __init__(self, url, protocol=None, options=None, ca_cert=None):
        self['URL'] = url
        self['Protocol'] = protocol
        self['Options'] = options
        self['CACert'] = ca_cert
