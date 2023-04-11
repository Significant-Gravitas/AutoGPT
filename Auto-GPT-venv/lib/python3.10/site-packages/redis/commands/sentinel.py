import warnings


class SentinelCommands:
    """
    A class containing the commands specific to redis sentinel. This class is
    to be used as a mixin.
    """

    def sentinel(self, *args):
        """Redis Sentinel's SENTINEL command."""
        warnings.warn(DeprecationWarning("Use the individual sentinel_* methods"))

    def sentinel_get_master_addr_by_name(self, service_name):
        """Returns a (host, port) pair for the given ``service_name``"""
        return self.execute_command("SENTINEL GET-MASTER-ADDR-BY-NAME", service_name)

    def sentinel_master(self, service_name):
        """Returns a dictionary containing the specified masters state."""
        return self.execute_command("SENTINEL MASTER", service_name)

    def sentinel_masters(self):
        """Returns a list of dictionaries containing each master's state."""
        return self.execute_command("SENTINEL MASTERS")

    def sentinel_monitor(self, name, ip, port, quorum):
        """Add a new master to Sentinel to be monitored"""
        return self.execute_command("SENTINEL MONITOR", name, ip, port, quorum)

    def sentinel_remove(self, name):
        """Remove a master from Sentinel's monitoring"""
        return self.execute_command("SENTINEL REMOVE", name)

    def sentinel_sentinels(self, service_name):
        """Returns a list of sentinels for ``service_name``"""
        return self.execute_command("SENTINEL SENTINELS", service_name)

    def sentinel_set(self, name, option, value):
        """Set Sentinel monitoring parameters for a given master"""
        return self.execute_command("SENTINEL SET", name, option, value)

    def sentinel_slaves(self, service_name):
        """Returns a list of slaves for ``service_name``"""
        return self.execute_command("SENTINEL SLAVES", service_name)

    def sentinel_reset(self, pattern):
        """
        This command will reset all the masters with matching name.
        The pattern argument is a glob-style pattern.

        The reset process clears any previous state in a master (including a
        failover in progress), and removes every slave and sentinel already
        discovered and associated with the master.
        """
        return self.execute_command("SENTINEL RESET", pattern, once=True)

    def sentinel_failover(self, new_master_name):
        """
        Force a failover as if the master was not reachable, and without
        asking for agreement to other Sentinels (however a new version of the
        configuration will be published so that the other Sentinels will
        update their configurations).
        """
        return self.execute_command("SENTINEL FAILOVER", new_master_name)

    def sentinel_ckquorum(self, new_master_name):
        """
        Check if the current Sentinel configuration is able to reach the
        quorum needed to failover a master, and the majority needed to
        authorize the failover.

        This command should be used in monitoring systems to check if a
        Sentinel deployment is ok.
        """
        return self.execute_command("SENTINEL CKQUORUM", new_master_name, once=True)

    def sentinel_flushconfig(self):
        """
        Force Sentinel to rewrite its configuration on disk, including the
        current Sentinel state.

        Normally Sentinel rewrites the configuration every time something
        changes in its state (in the context of the subset of the state which
        is persisted on disk across restart).
        However sometimes it is possible that the configuration file is lost
        because of operation errors, disk failures, package upgrade scripts or
        configuration managers. In those cases a way to to force Sentinel to
        rewrite the configuration file is handy.

        This command works even if the previous configuration file is
        completely missing.
        """
        return self.execute_command("SENTINEL FLUSHCONFIG")


class AsyncSentinelCommands(SentinelCommands):
    async def sentinel(self, *args) -> None:
        """Redis Sentinel's SENTINEL command."""
        super().sentinel(*args)
