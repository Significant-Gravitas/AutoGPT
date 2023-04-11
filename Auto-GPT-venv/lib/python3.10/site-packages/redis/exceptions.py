"Core exceptions raised by the Redis client"


class RedisError(Exception):
    pass


class ConnectionError(RedisError):
    pass


class TimeoutError(RedisError):
    pass


class AuthenticationError(ConnectionError):
    pass


class AuthorizationError(ConnectionError):
    pass


class BusyLoadingError(ConnectionError):
    pass


class InvalidResponse(RedisError):
    pass


class ResponseError(RedisError):
    pass


class DataError(RedisError):
    pass


class PubSubError(RedisError):
    pass


class WatchError(RedisError):
    pass


class NoScriptError(ResponseError):
    pass


class ExecAbortError(ResponseError):
    pass


class ReadOnlyError(ResponseError):
    pass


class NoPermissionError(ResponseError):
    pass


class ModuleError(ResponseError):
    pass


class LockError(RedisError, ValueError):
    "Errors acquiring or releasing a lock"
    # NOTE: For backwards compatibility, this class derives from ValueError.
    # This was originally chosen to behave like threading.Lock.
    pass


class LockNotOwnedError(LockError):
    "Error trying to extend or release a lock that is (no longer) owned"
    pass


class ChildDeadlockedError(Exception):
    "Error indicating that a child process is deadlocked after a fork()"
    pass


class AuthenticationWrongNumberOfArgsError(ResponseError):
    """
    An error to indicate that the wrong number of args
    were sent to the AUTH command
    """

    pass


class RedisClusterException(Exception):
    """
    Base exception for the RedisCluster client
    """

    pass


class ClusterError(RedisError):
    """
    Cluster errors occurred multiple times, resulting in an exhaustion of the
    command execution TTL
    """

    pass


class ClusterDownError(ClusterError, ResponseError):
    """
    Error indicated CLUSTERDOWN error received from cluster.
    By default Redis Cluster nodes stop accepting queries if they detect there
    is at least a hash slot uncovered (no available node is serving it).
    This way if the cluster is partially down (for example a range of hash
    slots are no longer covered) the entire cluster eventually becomes
    unavailable. It automatically returns available as soon as all the slots
    are covered again.
    """

    def __init__(self, resp):
        self.args = (resp,)
        self.message = resp


class AskError(ResponseError):
    """
    Error indicated ASK error received from cluster.
    When a slot is set as MIGRATING, the node will accept all queries that
    pertain to this hash slot, but only if the key in question exists,
    otherwise the query is forwarded using a -ASK redirection to the node that
    is target of the migration.
    src node: MIGRATING to dst node
        get > ASK error
        ask dst node > ASKING command
    dst node: IMPORTING from src node
        asking command only affects next command
        any op will be allowed after asking command
    """

    def __init__(self, resp):
        """should only redirect to master node"""
        self.args = (resp,)
        self.message = resp
        slot_id, new_node = resp.split(" ")
        host, port = new_node.rsplit(":", 1)
        self.slot_id = int(slot_id)
        self.node_addr = self.host, self.port = host, int(port)


class TryAgainError(ResponseError):
    """
    Error indicated TRYAGAIN error received from cluster.
    Operations on keys that don't exist or are - during resharding - split
    between the source and destination nodes, will generate a -TRYAGAIN error.
    """

    def __init__(self, *args, **kwargs):
        pass


class ClusterCrossSlotError(ResponseError):
    """
    Error indicated CROSSSLOT error received from cluster.
    A CROSSSLOT error is generated when keys in a request don't hash to the
    same slot.
    """

    message = "Keys in request don't hash to the same slot"


class MovedError(AskError):
    """
    Error indicated MOVED error received from cluster.
    A request sent to a node that doesn't serve this key will be replayed with
    a MOVED error that points to the correct node.
    """

    pass


class MasterDownError(ClusterDownError):
    """
    Error indicated MASTERDOWN error received from cluster.
    Link with MASTER is down and replica-serve-stale-data is set to 'no'.
    """

    pass


class SlotNotCoveredError(RedisClusterException):
    """
    This error only happens in the case where the connection pool will try to
    fetch what node that is covered by a given slot.

    If this error is raised the client should drop the current node layout and
    attempt to reconnect and refresh the node layout again
    """

    pass


class MaxConnectionsError(ConnectionError):
    ...
