import asyncio
import collections
import random
import socket
import warnings
from typing import (
    Any,
    Deque,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
)

from redis.asyncio.client import ResponseCallbackT
from redis.asyncio.connection import (
    Connection,
    DefaultParser,
    Encoder,
    SSLConnection,
    parse_url,
)
from redis.asyncio.lock import Lock
from redis.asyncio.parser import CommandsParser
from redis.asyncio.retry import Retry
from redis.backoff import default_backoff
from redis.client import EMPTY_RESPONSE, NEVER_DECODE, AbstractRedis
from redis.cluster import (
    PIPELINE_BLOCKED_COMMANDS,
    PRIMARY,
    REPLICA,
    SLOT_ID,
    AbstractRedisCluster,
    LoadBalancer,
    block_pipeline_command,
    get_node_name,
    parse_cluster_slots,
)
from redis.commands import READ_COMMANDS, AsyncRedisClusterCommands
from redis.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from redis.credentials import CredentialProvider
from redis.exceptions import (
    AskError,
    BusyLoadingError,
    ClusterCrossSlotError,
    ClusterDownError,
    ClusterError,
    ConnectionError,
    DataError,
    MasterDownError,
    MaxConnectionsError,
    MovedError,
    RedisClusterException,
    ResponseError,
    SlotNotCoveredError,
    TimeoutError,
    TryAgainError,
)
from redis.typing import AnyKeyT, EncodableT, KeyT
from redis.utils import dict_merge, safe_str, str_if_bytes

TargetNodesT = TypeVar(
    "TargetNodesT", str, "ClusterNode", List["ClusterNode"], Dict[Any, "ClusterNode"]
)


class ClusterParser(DefaultParser):
    EXCEPTION_CLASSES = dict_merge(
        DefaultParser.EXCEPTION_CLASSES,
        {
            "ASK": AskError,
            "CLUSTERDOWN": ClusterDownError,
            "CROSSSLOT": ClusterCrossSlotError,
            "MASTERDOWN": MasterDownError,
            "MOVED": MovedError,
            "TRYAGAIN": TryAgainError,
        },
    )


class RedisCluster(AbstractRedis, AbstractRedisCluster, AsyncRedisClusterCommands):
    """
    Create a new RedisCluster client.

    Pass one of parameters:

      - `host` & `port`
      - `startup_nodes`

    | Use ``await`` :meth:`initialize` to find cluster nodes & create connections.
    | Use ``await`` :meth:`close` to disconnect connections & close client.

    Many commands support the target_nodes kwarg. It can be one of the
    :attr:`NODE_FLAGS`:

      - :attr:`PRIMARIES`
      - :attr:`REPLICAS`
      - :attr:`ALL_NODES`
      - :attr:`RANDOM`
      - :attr:`DEFAULT_NODE`

    Note: This client is not thread/process/fork safe.

    :param host:
        | Can be used to point to a startup node
    :param port:
        | Port used if **host** is provided
    :param startup_nodes:
        | :class:`~.ClusterNode` to used as a startup node
    :param require_full_coverage:
        | When set to ``False``: the client will not require a full coverage of
          the slots. However, if not all slots are covered, and at least one node
          has ``cluster-require-full-coverage`` set to ``yes``, the server will throw
          a :class:`~.ClusterDownError` for some key-based commands.
        | When set to ``True``: all slots must be covered to construct the cluster
          client. If not all slots are covered, :class:`~.RedisClusterException` will be
          thrown.
        | See:
          https://redis.io/docs/manual/scaling/#redis-cluster-configuration-parameters
    :param read_from_replicas:
        | Enable read from replicas in READONLY mode. You can read possibly stale data.
          When set to true, read commands will be assigned between the primary and
          its replications in a Round-Robin manner.
    :param reinitialize_steps:
        | Specifies the number of MOVED errors that need to occur before reinitializing
          the whole cluster topology. If a MOVED error occurs and the cluster does not
          need to be reinitialized on this current error handling, only the MOVED slot
          will be patched with the redirected node.
          To reinitialize the cluster on every MOVED error, set reinitialize_steps to 1.
          To avoid reinitializing the cluster on moved errors, set reinitialize_steps to
          0.
    :param cluster_error_retry_attempts:
        | Number of times to retry before raising an error when :class:`~.TimeoutError`
          or :class:`~.ConnectionError` or :class:`~.ClusterDownError` are encountered
    :param connection_error_retry_attempts:
        | Number of times to retry before reinitializing when :class:`~.TimeoutError`
          or :class:`~.ConnectionError` are encountered.
          The default backoff strategy will be set if Retry object is not passed (see
          default_backoff in backoff.py). To change it, pass a custom Retry object
          using the "retry" keyword.
    :param max_connections:
        | Maximum number of connections per node. If there are no free connections & the
          maximum number of connections are already created, a
          :class:`~.MaxConnectionsError` is raised. This error may be retried as defined
          by :attr:`connection_error_retry_attempts`

    | Rest of the arguments will be passed to the
      :class:`~redis.asyncio.connection.Connection` instances when created

    :raises RedisClusterException:
        if any arguments are invalid or unknown. Eg:

        - `db` != 0 or None
        - `path` argument for unix socket connection
        - none of the `host`/`port` & `startup_nodes` were provided

    """

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> "RedisCluster":
        """
        Return a Redis client object configured from the given URL.

        For example::

            redis://[[username]:[password]]@localhost:6379/0
            rediss://[[username]:[password]]@localhost:6379/0

        Three URL schemes are supported:

        - `redis://` creates a TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/redis>
        - `rediss://` creates a SSL wrapped TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/rediss>

        The username, password, hostname, path and all querystring values are passed
        through ``urllib.parse.unquote`` in order to replace any percent-encoded values
        with their corresponding characters.

        All querystring options are cast to their appropriate Python types. Boolean
        arguments can be specified with string values "True"/"False" or "Yes"/"No".
        Values that cannot be properly cast cause a ``ValueError`` to be raised. Once
        parsed, the querystring arguments and keyword arguments are passed to
        :class:`~redis.asyncio.connection.Connection` when created.
        In the case of conflicting arguments, querystring arguments are used.
        """
        kwargs.update(parse_url(url))
        if kwargs.pop("connection_class", None) is SSLConnection:
            kwargs["ssl"] = True
        return cls(**kwargs)

    __slots__ = (
        "_initialize",
        "_lock",
        "cluster_error_retry_attempts",
        "command_flags",
        "commands_parser",
        "connection_error_retry_attempts",
        "connection_kwargs",
        "encoder",
        "node_flags",
        "nodes_manager",
        "read_from_replicas",
        "reinitialize_counter",
        "reinitialize_steps",
        "response_callbacks",
        "result_callbacks",
    )

    def __init__(
        self,
        host: Optional[str] = None,
        port: Union[str, int] = 6379,
        # Cluster related kwargs
        startup_nodes: Optional[List["ClusterNode"]] = None,
        require_full_coverage: bool = True,
        read_from_replicas: bool = False,
        reinitialize_steps: int = 5,
        cluster_error_retry_attempts: int = 3,
        connection_error_retry_attempts: int = 3,
        max_connections: int = 2**31,
        # Client related kwargs
        db: Union[str, int] = 0,
        path: Optional[str] = None,
        credential_provider: Optional[CredentialProvider] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_name: Optional[str] = None,
        # Encoding related kwargs
        encoding: str = "utf-8",
        encoding_errors: str = "strict",
        decode_responses: bool = False,
        # Connection related kwargs
        health_check_interval: float = 0,
        socket_connect_timeout: Optional[float] = None,
        socket_keepalive: bool = False,
        socket_keepalive_options: Optional[Mapping[int, Union[int, bytes]]] = None,
        socket_timeout: Optional[float] = None,
        retry: Optional["Retry"] = None,
        retry_on_error: Optional[List[Exception]] = None,
        # SSL related kwargs
        ssl: bool = False,
        ssl_ca_certs: Optional[str] = None,
        ssl_ca_data: Optional[str] = None,
        ssl_cert_reqs: str = "required",
        ssl_certfile: Optional[str] = None,
        ssl_check_hostname: bool = False,
        ssl_keyfile: Optional[str] = None,
    ) -> None:
        if db:
            raise RedisClusterException(
                "Argument 'db' must be 0 or None in cluster mode"
            )

        if path:
            raise RedisClusterException(
                "Unix domain socket is not supported in cluster mode"
            )

        if (not host or not port) and not startup_nodes:
            raise RedisClusterException(
                "RedisCluster requires at least one node to discover the cluster.\n"
                "Please provide one of the following or use RedisCluster.from_url:\n"
                '   - host and port: RedisCluster(host="localhost", port=6379)\n'
                "   - startup_nodes: RedisCluster(startup_nodes=["
                'ClusterNode("localhost", 6379), ClusterNode("localhost", 6380)])'
            )

        kwargs: Dict[str, Any] = {
            "max_connections": max_connections,
            "connection_class": Connection,
            "parser_class": ClusterParser,
            # Client related kwargs
            "credential_provider": credential_provider,
            "username": username,
            "password": password,
            "client_name": client_name,
            # Encoding related kwargs
            "encoding": encoding,
            "encoding_errors": encoding_errors,
            "decode_responses": decode_responses,
            # Connection related kwargs
            "health_check_interval": health_check_interval,
            "socket_connect_timeout": socket_connect_timeout,
            "socket_keepalive": socket_keepalive,
            "socket_keepalive_options": socket_keepalive_options,
            "socket_timeout": socket_timeout,
            "retry": retry,
        }

        if ssl:
            # SSL related kwargs
            kwargs.update(
                {
                    "connection_class": SSLConnection,
                    "ssl_ca_certs": ssl_ca_certs,
                    "ssl_ca_data": ssl_ca_data,
                    "ssl_cert_reqs": ssl_cert_reqs,
                    "ssl_certfile": ssl_certfile,
                    "ssl_check_hostname": ssl_check_hostname,
                    "ssl_keyfile": ssl_keyfile,
                }
            )

        if read_from_replicas:
            # Call our on_connect function to configure READONLY mode
            kwargs["redis_connect_func"] = self.on_connect

        self.retry = retry
        if retry or retry_on_error or connection_error_retry_attempts > 0:
            # Set a retry object for all cluster nodes
            self.retry = retry or Retry(
                default_backoff(), connection_error_retry_attempts
            )
            if not retry_on_error:
                # Default errors for retrying
                retry_on_error = [ConnectionError, TimeoutError]
            self.retry.update_supported_errors(retry_on_error)
            kwargs.update({"retry": self.retry})

        kwargs["response_callbacks"] = self.__class__.RESPONSE_CALLBACKS.copy()
        self.connection_kwargs = kwargs

        if startup_nodes:
            passed_nodes = []
            for node in startup_nodes:
                passed_nodes.append(
                    ClusterNode(node.host, node.port, **self.connection_kwargs)
                )
            startup_nodes = passed_nodes
        else:
            startup_nodes = []
        if host and port:
            startup_nodes.append(ClusterNode(host, port, **self.connection_kwargs))

        self.nodes_manager = NodesManager(startup_nodes, require_full_coverage, kwargs)
        self.encoder = Encoder(encoding, encoding_errors, decode_responses)
        self.read_from_replicas = read_from_replicas
        self.reinitialize_steps = reinitialize_steps
        self.cluster_error_retry_attempts = cluster_error_retry_attempts
        self.connection_error_retry_attempts = connection_error_retry_attempts
        self.reinitialize_counter = 0
        self.commands_parser = CommandsParser()
        self.node_flags = self.__class__.NODE_FLAGS.copy()
        self.command_flags = self.__class__.COMMAND_FLAGS.copy()
        self.response_callbacks = kwargs["response_callbacks"]
        self.result_callbacks = self.__class__.RESULT_CALLBACKS.copy()
        self.result_callbacks[
            "CLUSTER SLOTS"
        ] = lambda cmd, res, **kwargs: parse_cluster_slots(
            list(res.values())[0], **kwargs
        )

        self._initialize = True
        self._lock: Optional[asyncio.Lock] = None

    async def initialize(self) -> "RedisCluster":
        """Get all nodes from startup nodes & creates connections if not initialized."""
        if self._initialize:
            if not self._lock:
                self._lock = asyncio.Lock()
            async with self._lock:
                if self._initialize:
                    try:
                        await self.nodes_manager.initialize()
                        await self.commands_parser.initialize(
                            self.nodes_manager.default_node
                        )
                        self._initialize = False
                    except BaseException:
                        await self.nodes_manager.close()
                        await self.nodes_manager.close("startup_nodes")
                        raise
        return self

    async def close(self) -> None:
        """Close all connections & client if initialized."""
        if not self._initialize:
            if not self._lock:
                self._lock = asyncio.Lock()
            async with self._lock:
                if not self._initialize:
                    self._initialize = True
                    await self.nodes_manager.close()
                    await self.nodes_manager.close("startup_nodes")

    async def __aenter__(self) -> "RedisCluster":
        return await self.initialize()

    async def __aexit__(self, exc_type: None, exc_value: None, traceback: None) -> None:
        await self.close()

    def __await__(self) -> Generator[Any, None, "RedisCluster"]:
        return self.initialize().__await__()

    _DEL_MESSAGE = "Unclosed RedisCluster client"

    def __del__(self) -> None:
        if hasattr(self, "_initialize") and not self._initialize:
            warnings.warn(f"{self._DEL_MESSAGE} {self!r}", ResourceWarning, source=self)
            try:
                context = {"client": self, "message": self._DEL_MESSAGE}
                asyncio.get_running_loop().call_exception_handler(context)
            except RuntimeError:
                ...

    async def on_connect(self, connection: Connection) -> None:
        await connection.on_connect()

        # Sending READONLY command to server to configure connection as
        # readonly. Since each cluster node may change its server type due
        # to a failover, we should establish a READONLY connection
        # regardless of the server type. If this is a primary connection,
        # READONLY would not affect executing write commands.
        await connection.send_command("READONLY")
        if str_if_bytes(await connection.read_response()) != "OK":
            raise ConnectionError("READONLY command failed")

    def get_nodes(self) -> List["ClusterNode"]:
        """Get all nodes of the cluster."""
        return list(self.nodes_manager.nodes_cache.values())

    def get_primaries(self) -> List["ClusterNode"]:
        """Get the primary nodes of the cluster."""
        return self.nodes_manager.get_nodes_by_server_type(PRIMARY)

    def get_replicas(self) -> List["ClusterNode"]:
        """Get the replica nodes of the cluster."""
        return self.nodes_manager.get_nodes_by_server_type(REPLICA)

    def get_random_node(self) -> "ClusterNode":
        """Get a random node of the cluster."""
        return random.choice(list(self.nodes_manager.nodes_cache.values()))

    def get_default_node(self) -> "ClusterNode":
        """Get the default node of the client."""
        return self.nodes_manager.default_node

    def set_default_node(self, node: "ClusterNode") -> None:
        """
        Set the default node of the client.

        :raises DataError: if None is passed or node does not exist in cluster.
        """
        if not node or not self.get_node(node_name=node.name):
            raise DataError("The requested node does not exist in the cluster.")

        self.nodes_manager.default_node = node

    def get_node(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        node_name: Optional[str] = None,
    ) -> Optional["ClusterNode"]:
        """Get node by (host, port) or node_name."""
        return self.nodes_manager.get_node(host, port, node_name)

    def get_node_from_key(
        self, key: str, replica: bool = False
    ) -> Optional["ClusterNode"]:
        """
        Get the cluster node corresponding to the provided key.

        :param key:
        :param replica:
            | Indicates if a replica should be returned
            |
              None will returned if no replica holds this key

        :raises SlotNotCoveredError: if the key is not covered by any slot.
        """
        slot = self.keyslot(key)
        slot_cache = self.nodes_manager.slots_cache.get(slot)
        if not slot_cache:
            raise SlotNotCoveredError(f'Slot "{slot}" is not covered by the cluster.')

        if replica:
            if len(self.nodes_manager.slots_cache[slot]) < 2:
                return None
            node_idx = 1
        else:
            node_idx = 0

        return slot_cache[node_idx]

    def keyslot(self, key: EncodableT) -> int:
        """
        Find the keyslot for a given key.

        See: https://redis.io/docs/manual/scaling/#redis-cluster-data-sharding
        """
        return key_slot(self.encoder.encode(key))

    def get_encoder(self) -> Encoder:
        """Get the encoder object of the client."""
        return self.encoder

    def get_connection_kwargs(self) -> Dict[str, Optional[Any]]:
        """Get the kwargs passed to :class:`~redis.asyncio.connection.Connection`."""
        return self.connection_kwargs

    def get_retry(self) -> Optional["Retry"]:
        return self.retry

    def set_retry(self, retry: "Retry") -> None:
        self.retry = retry
        for node in self.get_nodes():
            node.connection_kwargs.update({"retry": retry})
            for conn in node._connections:
                conn.retry = retry

    def set_response_callback(self, command: str, callback: ResponseCallbackT) -> None:
        """Set a custom response callback."""
        self.response_callbacks[command] = callback

    async def _determine_nodes(
        self, command: str, *args: Any, node_flag: Optional[str] = None
    ) -> List["ClusterNode"]:
        # Determine which nodes should be executed the command on.
        # Returns a list of target nodes.
        if not node_flag:
            # get the nodes group for this command if it was predefined
            node_flag = self.command_flags.get(command)

        if node_flag in self.node_flags:
            if node_flag == self.__class__.DEFAULT_NODE:
                # return the cluster's default node
                return [self.nodes_manager.default_node]
            if node_flag == self.__class__.PRIMARIES:
                # return all primaries
                return self.nodes_manager.get_nodes_by_server_type(PRIMARY)
            if node_flag == self.__class__.REPLICAS:
                # return all replicas
                return self.nodes_manager.get_nodes_by_server_type(REPLICA)
            if node_flag == self.__class__.ALL_NODES:
                # return all nodes
                return list(self.nodes_manager.nodes_cache.values())
            if node_flag == self.__class__.RANDOM:
                # return a random node
                return [random.choice(list(self.nodes_manager.nodes_cache.values()))]

        # get the node that holds the key's slot
        return [
            self.nodes_manager.get_node_from_slot(
                await self._determine_slot(command, *args),
                self.read_from_replicas and command in READ_COMMANDS,
            )
        ]

    async def _determine_slot(self, command: str, *args: Any) -> int:
        if self.command_flags.get(command) == SLOT_ID:
            # The command contains the slot ID
            return int(args[0])

        # Get the keys in the command

        # EVAL and EVALSHA are common enough that it's wasteful to go to the
        # redis server to parse the keys. Besides, there is a bug in redis<7.0
        # where `self._get_command_keys()` fails anyway. So, we special case
        # EVAL/EVALSHA.
        # - issue: https://github.com/redis/redis/issues/9493
        # - fix: https://github.com/redis/redis/pull/9733
        if command in ("EVAL", "EVALSHA"):
            # command syntax: EVAL "script body" num_keys ...
            if len(args) < 2:
                raise RedisClusterException(
                    f"Invalid args in command: {command, *args}"
                )
            keys = args[2 : 2 + args[1]]
            # if there are 0 keys, that means the script can be run on any node
            # so we can just return a random slot
            if not keys:
                return random.randrange(0, REDIS_CLUSTER_HASH_SLOTS)
        else:
            keys = await self.commands_parser.get_keys(command, *args)
            if not keys:
                # FCALL can call a function with 0 keys, that means the function
                #  can be run on any node so we can just return a random slot
                if command in ("FCALL", "FCALL_RO"):
                    return random.randrange(0, REDIS_CLUSTER_HASH_SLOTS)
                raise RedisClusterException(
                    "No way to dispatch this command to Redis Cluster. "
                    "Missing key.\nYou can execute the command by specifying "
                    f"target nodes.\nCommand: {args}"
                )

        # single key command
        if len(keys) == 1:
            return self.keyslot(keys[0])

        # multi-key command; we need to make sure all keys are mapped to
        # the same slot
        slots = {self.keyslot(key) for key in keys}
        if len(slots) != 1:
            raise RedisClusterException(
                f"{command} - all keys must map to the same key slot"
            )

        return slots.pop()

    def _is_node_flag(self, target_nodes: Any) -> bool:
        return isinstance(target_nodes, str) and target_nodes in self.node_flags

    def _parse_target_nodes(self, target_nodes: Any) -> List["ClusterNode"]:
        if isinstance(target_nodes, list):
            nodes = target_nodes
        elif isinstance(target_nodes, ClusterNode):
            # Supports passing a single ClusterNode as a variable
            nodes = [target_nodes]
        elif isinstance(target_nodes, dict):
            # Supports dictionaries of the format {node_name: node}.
            # It enables to execute commands with multi nodes as follows:
            # rc.cluster_save_config(rc.get_primaries())
            nodes = list(target_nodes.values())
        else:
            raise TypeError(
                "target_nodes type can be one of the following: "
                "node_flag (PRIMARIES, REPLICAS, RANDOM, ALL_NODES),"
                "ClusterNode, list<ClusterNode>, or dict<any, ClusterNode>. "
                f"The passed type is {type(target_nodes)}"
            )
        return nodes

    async def execute_command(self, *args: EncodableT, **kwargs: Any) -> Any:
        """
        Execute a raw command on the appropriate cluster node or target_nodes.

        It will retry the command as specified by :attr:`cluster_error_retry_attempts` &
        then raise an exception.

        :param args:
            | Raw command args
        :param kwargs:

            - target_nodes: :attr:`NODE_FLAGS` or :class:`~.ClusterNode`
              or List[:class:`~.ClusterNode`] or Dict[Any, :class:`~.ClusterNode`]
            - Rest of the kwargs are passed to the Redis connection

        :raises RedisClusterException: if target_nodes is not provided & the command
            can't be mapped to a slot
        """
        command = args[0]
        target_nodes = []
        target_nodes_specified = False
        retry_attempts = self.cluster_error_retry_attempts

        passed_targets = kwargs.pop("target_nodes", None)
        if passed_targets and not self._is_node_flag(passed_targets):
            target_nodes = self._parse_target_nodes(passed_targets)
            target_nodes_specified = True
            retry_attempts = 0

        # Add one for the first execution
        execute_attempts = 1 + retry_attempts
        for _ in range(execute_attempts):
            if self._initialize:
                await self.initialize()
                if (
                    len(target_nodes) == 1
                    and target_nodes[0] == self.get_default_node()
                ):
                    # Replace the default cluster node
                    self.replace_default_node()
            try:
                if not target_nodes_specified:
                    # Determine the nodes to execute the command on
                    target_nodes = await self._determine_nodes(
                        *args, node_flag=passed_targets
                    )
                    if not target_nodes:
                        raise RedisClusterException(
                            f"No targets were found to execute {args} command on"
                        )

                if len(target_nodes) == 1:
                    # Return the processed result
                    ret = await self._execute_command(target_nodes[0], *args, **kwargs)
                    if command in self.result_callbacks:
                        return self.result_callbacks[command](
                            command, {target_nodes[0].name: ret}, **kwargs
                        )
                    return ret
                else:
                    keys = [node.name for node in target_nodes]
                    values = await asyncio.gather(
                        *(
                            asyncio.create_task(
                                self._execute_command(node, *args, **kwargs)
                            )
                            for node in target_nodes
                        )
                    )
                    if command in self.result_callbacks:
                        return self.result_callbacks[command](
                            command, dict(zip(keys, values)), **kwargs
                        )
                    return dict(zip(keys, values))
            except Exception as e:
                if retry_attempts > 0 and type(e) in self.__class__.ERRORS_ALLOW_RETRY:
                    # The nodes and slots cache were should be reinitialized.
                    # Try again with the new cluster setup.
                    retry_attempts -= 1
                    continue
                else:
                    # raise the exception
                    raise e

    async def _execute_command(
        self, target_node: "ClusterNode", *args: Union[KeyT, EncodableT], **kwargs: Any
    ) -> Any:
        asking = moved = False
        redirect_addr = None
        ttl = self.RedisClusterRequestTTL

        while ttl > 0:
            ttl -= 1
            try:
                if asking:
                    target_node = self.get_node(node_name=redirect_addr)
                    await target_node.execute_command("ASKING")
                    asking = False
                elif moved:
                    # MOVED occurred and the slots cache was updated,
                    # refresh the target node
                    slot = await self._determine_slot(*args)
                    target_node = self.nodes_manager.get_node_from_slot(
                        slot, self.read_from_replicas and args[0] in READ_COMMANDS
                    )
                    moved = False

                return await target_node.execute_command(*args, **kwargs)
            except (BusyLoadingError, MaxConnectionsError):
                raise
            except (ConnectionError, TimeoutError):
                # Connection retries are being handled in the node's
                # Retry object.
                # Remove the failed node from the startup nodes before we try
                # to reinitialize the cluster
                self.nodes_manager.startup_nodes.pop(target_node.name, None)
                # Hard force of reinitialize of the node/slots setup
                # and try again with the new setup
                await self.close()
                raise
            except ClusterDownError:
                # ClusterDownError can occur during a failover and to get
                # self-healed, we will try to reinitialize the cluster layout
                # and retry executing the command
                await self.close()
                await asyncio.sleep(0.25)
                raise
            except MovedError as e:
                # First, we will try to patch the slots/nodes cache with the
                # redirected node output and try again. If MovedError exceeds
                # 'reinitialize_steps' number of times, we will force
                # reinitializing the tables, and then try again.
                # 'reinitialize_steps' counter will increase faster when
                # the same client object is shared between multiple threads. To
                # reduce the frequency you can set this variable in the
                # RedisCluster constructor.
                self.reinitialize_counter += 1
                if (
                    self.reinitialize_steps
                    and self.reinitialize_counter % self.reinitialize_steps == 0
                ):
                    await self.close()
                    # Reset the counter
                    self.reinitialize_counter = 0
                else:
                    self.nodes_manager._moved_exception = e
                moved = True
            except AskError as e:
                redirect_addr = get_node_name(host=e.host, port=e.port)
                asking = True
            except TryAgainError:
                if ttl < self.RedisClusterRequestTTL / 2:
                    await asyncio.sleep(0.05)

        raise ClusterError("TTL exhausted.")

    def pipeline(
        self, transaction: Optional[Any] = None, shard_hint: Optional[Any] = None
    ) -> "ClusterPipeline":
        """
        Create & return a new :class:`~.ClusterPipeline` object.

        Cluster implementation of pipeline does not support transaction or shard_hint.

        :raises RedisClusterException: if transaction or shard_hint are truthy values
        """
        if shard_hint:
            raise RedisClusterException("shard_hint is deprecated in cluster mode")

        if transaction:
            raise RedisClusterException("transaction is deprecated in cluster mode")

        return ClusterPipeline(self)

    def lock(
        self,
        name: KeyT,
        timeout: Optional[float] = None,
        sleep: float = 0.1,
        blocking: bool = True,
        blocking_timeout: Optional[float] = None,
        lock_class: Optional[Type[Lock]] = None,
        thread_local: bool = True,
    ) -> Lock:
        """
        Return a new Lock object using key ``name`` that mimics
        the behavior of threading.Lock.

        If specified, ``timeout`` indicates a maximum life for the lock.
        By default, it will remain locked until release() is called.

        ``sleep`` indicates the amount of time to sleep per loop iteration
        when the lock is in blocking mode and another client is currently
        holding the lock.

        ``blocking`` indicates whether calling ``acquire`` should block until
        the lock has been acquired or to fail immediately, causing ``acquire``
        to return False and the lock not being acquired. Defaults to True.
        Note this value can be overridden by passing a ``blocking``
        argument to ``acquire``.

        ``blocking_timeout`` indicates the maximum amount of time in seconds to
        spend trying to acquire the lock. A value of ``None`` indicates
        continue trying forever. ``blocking_timeout`` can be specified as a
        float or integer, both representing the number of seconds to wait.

        ``lock_class`` forces the specified lock implementation. Note that as
        of redis-py 3.0, the only lock class we implement is ``Lock`` (which is
        a Lua-based lock). So, it's unlikely you'll need this parameter, unless
        you have created your own custom lock class.

        ``thread_local`` indicates whether the lock token is placed in
        thread-local storage. By default, the token is placed in thread local
        storage so that a thread only sees its token, not a token set by
        another thread. Consider the following timeline:

            time: 0, thread-1 acquires `my-lock`, with a timeout of 5 seconds.
                     thread-1 sets the token to "abc"
            time: 1, thread-2 blocks trying to acquire `my-lock` using the
                     Lock instance.
            time: 5, thread-1 has not yet completed. redis expires the lock
                     key.
            time: 5, thread-2 acquired `my-lock` now that it's available.
                     thread-2 sets the token to "xyz"
            time: 6, thread-1 finishes its work and calls release(). if the
                     token is *not* stored in thread local storage, then
                     thread-1 would see the token value as "xyz" and would be
                     able to successfully release the thread-2's lock.

        In some use cases it's necessary to disable thread local storage. For
        example, if you have code where one thread acquires a lock and passes
        that lock instance to a worker thread to release later. If thread
        local storage isn't disabled in this case, the worker thread won't see
        the token set by the thread that acquired the lock. Our assumption
        is that these cases aren't common and as such default to using
        thread local storage."""
        if lock_class is None:
            lock_class = Lock
        return lock_class(
            self,
            name,
            timeout=timeout,
            sleep=sleep,
            blocking=blocking,
            blocking_timeout=blocking_timeout,
            thread_local=thread_local,
        )


class ClusterNode:
    """
    Create a new ClusterNode.

    Each ClusterNode manages multiple :class:`~redis.asyncio.connection.Connection`
    objects for the (host, port).
    """

    __slots__ = (
        "_connections",
        "_free",
        "connection_class",
        "connection_kwargs",
        "host",
        "max_connections",
        "name",
        "port",
        "response_callbacks",
        "server_type",
    )

    def __init__(
        self,
        host: str,
        port: Union[str, int],
        server_type: Optional[str] = None,
        *,
        max_connections: int = 2**31,
        connection_class: Type[Connection] = Connection,
        **connection_kwargs: Any,
    ) -> None:
        if host == "localhost":
            host = socket.gethostbyname(host)

        connection_kwargs["host"] = host
        connection_kwargs["port"] = port
        self.host = host
        self.port = port
        self.name = get_node_name(host, port)
        self.server_type = server_type

        self.max_connections = max_connections
        self.connection_class = connection_class
        self.connection_kwargs = connection_kwargs
        self.response_callbacks = connection_kwargs.pop("response_callbacks", {})

        self._connections: List[Connection] = []
        self._free: Deque[Connection] = collections.deque(maxlen=self.max_connections)

    def __repr__(self) -> str:
        return (
            f"[host={self.host}, port={self.port}, "
            f"name={self.name}, server_type={self.server_type}]"
        )

    def __eq__(self, obj: Any) -> bool:
        return isinstance(obj, ClusterNode) and obj.name == self.name

    _DEL_MESSAGE = "Unclosed ClusterNode object"

    def __del__(self) -> None:
        for connection in self._connections:
            if connection.is_connected:
                warnings.warn(
                    f"{self._DEL_MESSAGE} {self!r}", ResourceWarning, source=self
                )
                try:
                    context = {"client": self, "message": self._DEL_MESSAGE}
                    asyncio.get_running_loop().call_exception_handler(context)
                except RuntimeError:
                    ...
                break

    async def disconnect(self) -> None:
        ret = await asyncio.gather(
            *(
                asyncio.create_task(connection.disconnect())
                for connection in self._connections
            ),
            return_exceptions=True,
        )
        exc = next((res for res in ret if isinstance(res, Exception)), None)
        if exc:
            raise exc

    def acquire_connection(self) -> Connection:
        try:
            return self._free.popleft()
        except IndexError:
            if len(self._connections) < self.max_connections:
                connection = self.connection_class(**self.connection_kwargs)
                self._connections.append(connection)
                return connection

            raise MaxConnectionsError()

    async def parse_response(
        self, connection: Connection, command: str, **kwargs: Any
    ) -> Any:
        try:
            if NEVER_DECODE in kwargs:
                response = await connection.read_response(disable_decoding=True)
                kwargs.pop(NEVER_DECODE)
            else:
                response = await connection.read_response()
        except ResponseError:
            if EMPTY_RESPONSE in kwargs:
                return kwargs[EMPTY_RESPONSE]
            raise

        if EMPTY_RESPONSE in kwargs:
            kwargs.pop(EMPTY_RESPONSE)

        # Return response
        if command in self.response_callbacks:
            return self.response_callbacks[command](response, **kwargs)

        return response

    async def execute_command(self, *args: Any, **kwargs: Any) -> Any:
        # Acquire connection
        connection = self.acquire_connection()

        # Execute command
        await connection.send_packed_command(connection.pack_command(*args), False)

        # Read response
        return await asyncio.shield(
            self._parse_and_release(connection, args[0], **kwargs)
        )

    async def _parse_and_release(self, connection, *args, **kwargs):
        try:
            return await self.parse_response(connection, *args, **kwargs)
        except asyncio.CancelledError:
            # should not be possible
            await connection.disconnect(nowait=True)
            raise
        finally:
            self._free.append(connection)

    async def _try_parse_response(self, cmd, connection, ret):
        try:
            cmd.result = await asyncio.shield(
                self.parse_response(connection, cmd.args[0], **cmd.kwargs)
            )
        except asyncio.CancelledError:
            await connection.disconnect(nowait=True)
            raise
        except Exception as e:
            cmd.result = e
            ret = True
        return ret

    async def execute_pipeline(self, commands: List["PipelineCommand"]) -> bool:
        # Acquire connection
        connection = self.acquire_connection()

        # Execute command
        await connection.send_packed_command(
            connection.pack_commands(cmd.args for cmd in commands), False
        )

        # Read responses
        ret = False
        for cmd in commands:
            ret = await asyncio.shield(self._try_parse_response(cmd, connection, ret))

        # Release connection
        self._free.append(connection)

        return ret


class NodesManager:
    __slots__ = (
        "_moved_exception",
        "connection_kwargs",
        "default_node",
        "nodes_cache",
        "read_load_balancer",
        "require_full_coverage",
        "slots_cache",
        "startup_nodes",
    )

    def __init__(
        self,
        startup_nodes: List["ClusterNode"],
        require_full_coverage: bool,
        connection_kwargs: Dict[str, Any],
    ) -> None:
        self.startup_nodes = {node.name: node for node in startup_nodes}
        self.require_full_coverage = require_full_coverage
        self.connection_kwargs = connection_kwargs

        self.default_node: "ClusterNode" = None
        self.nodes_cache: Dict[str, "ClusterNode"] = {}
        self.slots_cache: Dict[int, List["ClusterNode"]] = {}
        self.read_load_balancer = LoadBalancer()
        self._moved_exception: MovedError = None

    def get_node(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        node_name: Optional[str] = None,
    ) -> Optional["ClusterNode"]:
        if host and port:
            # the user passed host and port
            if host == "localhost":
                host = socket.gethostbyname(host)
            return self.nodes_cache.get(get_node_name(host=host, port=port))
        elif node_name:
            return self.nodes_cache.get(node_name)
        else:
            raise DataError(
                "get_node requires one of the following: "
                "1. node name "
                "2. host and port"
            )

    def set_nodes(
        self,
        old: Dict[str, "ClusterNode"],
        new: Dict[str, "ClusterNode"],
        remove_old: bool = False,
    ) -> None:
        if remove_old:
            for name in list(old.keys()):
                if name not in new:
                    asyncio.create_task(old.pop(name).disconnect())

        for name, node in new.items():
            if name in old:
                if old[name] is node:
                    continue
                asyncio.create_task(old[name].disconnect())
            old[name] = node

    def _update_moved_slots(self) -> None:
        e = self._moved_exception
        redirected_node = self.get_node(host=e.host, port=e.port)
        if redirected_node:
            # The node already exists
            if redirected_node.server_type != PRIMARY:
                # Update the node's server type
                redirected_node.server_type = PRIMARY
        else:
            # This is a new node, we will add it to the nodes cache
            redirected_node = ClusterNode(
                e.host, e.port, PRIMARY, **self.connection_kwargs
            )
            self.set_nodes(self.nodes_cache, {redirected_node.name: redirected_node})
        if redirected_node in self.slots_cache[e.slot_id]:
            # The MOVED error resulted from a failover, and the new slot owner
            # had previously been a replica.
            old_primary = self.slots_cache[e.slot_id][0]
            # Update the old primary to be a replica and add it to the end of
            # the slot's node list
            old_primary.server_type = REPLICA
            self.slots_cache[e.slot_id].append(old_primary)
            # Remove the old replica, which is now a primary, from the slot's
            # node list
            self.slots_cache[e.slot_id].remove(redirected_node)
            # Override the old primary with the new one
            self.slots_cache[e.slot_id][0] = redirected_node
            if self.default_node == old_primary:
                # Update the default node with the new primary
                self.default_node = redirected_node
        else:
            # The new slot owner is a new server, or a server from a different
            # shard. We need to remove all current nodes from the slot's list
            # (including replications) and add just the new node.
            self.slots_cache[e.slot_id] = [redirected_node]
        # Reset moved_exception
        self._moved_exception = None

    def get_node_from_slot(
        self, slot: int, read_from_replicas: bool = False
    ) -> "ClusterNode":
        if self._moved_exception:
            self._update_moved_slots()

        try:
            if read_from_replicas:
                # get the server index in a Round-Robin manner
                primary_name = self.slots_cache[slot][0].name
                node_idx = self.read_load_balancer.get_server_index(
                    primary_name, len(self.slots_cache[slot])
                )
                return self.slots_cache[slot][node_idx]
            return self.slots_cache[slot][0]
        except (IndexError, TypeError):
            raise SlotNotCoveredError(
                f'Slot "{slot}" not covered by the cluster. '
                f'"require_full_coverage={self.require_full_coverage}"'
            )

    def get_nodes_by_server_type(self, server_type: str) -> List["ClusterNode"]:
        return [
            node
            for node in self.nodes_cache.values()
            if node.server_type == server_type
        ]

    async def initialize(self) -> None:
        self.read_load_balancer.reset()
        tmp_nodes_cache: Dict[str, "ClusterNode"] = {}
        tmp_slots: Dict[int, List["ClusterNode"]] = {}
        disagreements = []
        startup_nodes_reachable = False
        fully_covered = False
        exception = None
        for startup_node in self.startup_nodes.values():
            try:
                # Make sure cluster mode is enabled on this node
                if not (await startup_node.execute_command("INFO")).get(
                    "cluster_enabled"
                ):
                    raise RedisClusterException(
                        "Cluster mode is not enabled on this node"
                    )
                cluster_slots = await startup_node.execute_command("CLUSTER SLOTS")
                startup_nodes_reachable = True
            except Exception as e:
                # Try the next startup node.
                # The exception is saved and raised only if we have no more nodes.
                exception = e
                continue

            # CLUSTER SLOTS command results in the following output:
            # [[slot_section[from_slot,to_slot,master,replica1,...,replicaN]]]
            # where each node contains the following list: [IP, port, node_id]
            # Therefore, cluster_slots[0][2][0] will be the IP address of the
            # primary node of the first slot section.
            # If there's only one server in the cluster, its ``host`` is ''
            # Fix it to the host in startup_nodes
            if (
                len(cluster_slots) == 1
                and not cluster_slots[0][2][0]
                and len(self.startup_nodes) == 1
            ):
                cluster_slots[0][2][0] = startup_node.host

            for slot in cluster_slots:
                for i in range(2, len(slot)):
                    slot[i] = [str_if_bytes(val) for val in slot[i]]
                primary_node = slot[2]
                host = primary_node[0]
                if host == "":
                    host = startup_node.host
                port = int(primary_node[1])

                target_node = tmp_nodes_cache.get(get_node_name(host, port))
                if not target_node:
                    target_node = ClusterNode(
                        host, port, PRIMARY, **self.connection_kwargs
                    )
                # add this node to the nodes cache
                tmp_nodes_cache[target_node.name] = target_node

                for i in range(int(slot[0]), int(slot[1]) + 1):
                    if i not in tmp_slots:
                        tmp_slots[i] = []
                        tmp_slots[i].append(target_node)
                        replica_nodes = [slot[j] for j in range(3, len(slot))]

                        for replica_node in replica_nodes:
                            host = replica_node[0]
                            port = replica_node[1]

                            target_replica_node = tmp_nodes_cache.get(
                                get_node_name(host, port)
                            )
                            if not target_replica_node:
                                target_replica_node = ClusterNode(
                                    host, port, REPLICA, **self.connection_kwargs
                                )
                            tmp_slots[i].append(target_replica_node)
                            # add this node to the nodes cache
                            tmp_nodes_cache[
                                target_replica_node.name
                            ] = target_replica_node
                    else:
                        # Validate that 2 nodes want to use the same slot cache
                        # setup
                        tmp_slot = tmp_slots[i][0]
                        if tmp_slot.name != target_node.name:
                            disagreements.append(
                                f"{tmp_slot.name} vs {target_node.name} on slot: {i}"
                            )

                            if len(disagreements) > 5:
                                raise RedisClusterException(
                                    f"startup_nodes could not agree on a valid "
                                    f'slots cache: {", ".join(disagreements)}'
                                )

            # Validate if all slots are covered or if we should try next startup node
            fully_covered = True
            for i in range(REDIS_CLUSTER_HASH_SLOTS):
                if i not in tmp_slots:
                    fully_covered = False
                    break
            if fully_covered:
                break

        if not startup_nodes_reachable:
            raise RedisClusterException(
                f"Redis Cluster cannot be connected. Please provide at least "
                f"one reachable node: {str(exception)}"
            ) from exception

        # Check if the slots are not fully covered
        if not fully_covered and self.require_full_coverage:
            # Despite the requirement that the slots be covered, there
            # isn't a full coverage
            raise RedisClusterException(
                f"All slots are not covered after query all startup_nodes. "
                f"{len(tmp_slots)} of {REDIS_CLUSTER_HASH_SLOTS} "
                f"covered..."
            )

        # Set the tmp variables to the real variables
        self.slots_cache = tmp_slots
        self.set_nodes(self.nodes_cache, tmp_nodes_cache, remove_old=True)
        # Populate the startup nodes with all discovered nodes
        self.set_nodes(self.startup_nodes, self.nodes_cache, remove_old=True)

        # Set the default node
        self.default_node = self.get_nodes_by_server_type(PRIMARY)[0]
        # If initialize was called after a MovedError, clear it
        self._moved_exception = None

    async def close(self, attr: str = "nodes_cache") -> None:
        self.default_node = None
        await asyncio.gather(
            *(
                asyncio.create_task(node.disconnect())
                for node in getattr(self, attr).values()
            )
        )


class ClusterPipeline(AbstractRedis, AbstractRedisCluster, AsyncRedisClusterCommands):
    """
    Create a new ClusterPipeline object.

    Usage::

        result = await (
            rc.pipeline()
            .set("A", 1)
            .get("A")
            .hset("K", "F", "V")
            .hgetall("K")
            .mset_nonatomic({"A": 2, "B": 3})
            .get("A")
            .get("B")
            .delete("A", "B", "K")
            .execute()
        )
        # result = [True, "1", 1, {"F": "V"}, True, True, "2", "3", 1, 1, 1]

    Note: For commands `DELETE`, `EXISTS`, `TOUCH`, `UNLINK`, `mset_nonatomic`, which
    are split across multiple nodes, you'll get multiple results for them in the array.

    Retryable errors:
        - :class:`~.ClusterDownError`
        - :class:`~.ConnectionError`
        - :class:`~.TimeoutError`

    Redirection errors:
        - :class:`~.TryAgainError`
        - :class:`~.MovedError`
        - :class:`~.AskError`

    :param client:
        | Existing :class:`~.RedisCluster` client
    """

    __slots__ = ("_command_stack", "_client")

    def __init__(self, client: RedisCluster) -> None:
        self._client = client

        self._command_stack: List["PipelineCommand"] = []

    async def initialize(self) -> "ClusterPipeline":
        if self._client._initialize:
            await self._client.initialize()
        self._command_stack = []
        return self

    async def __aenter__(self) -> "ClusterPipeline":
        return await self.initialize()

    async def __aexit__(self, exc_type: None, exc_value: None, traceback: None) -> None:
        self._command_stack = []

    def __await__(self) -> Generator[Any, None, "ClusterPipeline"]:
        return self.initialize().__await__()

    def __enter__(self) -> "ClusterPipeline":
        self._command_stack = []
        return self

    def __exit__(self, exc_type: None, exc_value: None, traceback: None) -> None:
        self._command_stack = []

    def __bool__(self) -> bool:
        return bool(self._command_stack)

    def __len__(self) -> int:
        return len(self._command_stack)

    def execute_command(
        self, *args: Union[KeyT, EncodableT], **kwargs: Any
    ) -> "ClusterPipeline":
        """
        Append a raw command to the pipeline.

        :param args:
            | Raw command args
        :param kwargs:

            - target_nodes: :attr:`NODE_FLAGS` or :class:`~.ClusterNode`
              or List[:class:`~.ClusterNode`] or Dict[Any, :class:`~.ClusterNode`]
            - Rest of the kwargs are passed to the Redis connection
        """
        self._command_stack.append(
            PipelineCommand(len(self._command_stack), *args, **kwargs)
        )
        return self

    async def execute(
        self, raise_on_error: bool = True, allow_redirections: bool = True
    ) -> List[Any]:
        """
        Execute the pipeline.

        It will retry the commands as specified by :attr:`cluster_error_retry_attempts`
        & then raise an exception.

        :param raise_on_error:
            | Raise the first error if there are any errors
        :param allow_redirections:
            | Whether to retry each failed command individually in case of redirection
              errors

        :raises RedisClusterException: if target_nodes is not provided & the command
            can't be mapped to a slot
        """
        if not self._command_stack:
            return []

        try:
            for _ in range(self._client.cluster_error_retry_attempts):
                if self._client._initialize:
                    await self._client.initialize()

                try:
                    return await self._execute(
                        self._client,
                        self._command_stack,
                        raise_on_error=raise_on_error,
                        allow_redirections=allow_redirections,
                    )
                except BaseException as e:
                    if type(e) in self.__class__.ERRORS_ALLOW_RETRY:
                        # Try again with the new cluster setup.
                        exception = e
                        await self._client.close()
                        await asyncio.sleep(0.25)
                    else:
                        # All other errors should be raised.
                        raise

            # If it fails the configured number of times then raise an exception
            raise exception
        finally:
            self._command_stack = []

    async def _execute(
        self,
        client: "RedisCluster",
        stack: List["PipelineCommand"],
        raise_on_error: bool = True,
        allow_redirections: bool = True,
    ) -> List[Any]:
        todo = [
            cmd for cmd in stack if not cmd.result or isinstance(cmd.result, Exception)
        ]

        nodes = {}
        for cmd in todo:
            passed_targets = cmd.kwargs.pop("target_nodes", None)
            if passed_targets and not client._is_node_flag(passed_targets):
                target_nodes = client._parse_target_nodes(passed_targets)
            else:
                target_nodes = await client._determine_nodes(
                    *cmd.args, node_flag=passed_targets
                )
                if not target_nodes:
                    raise RedisClusterException(
                        f"No targets were found to execute {cmd.args} command on"
                    )
            if len(target_nodes) > 1:
                raise RedisClusterException(f"Too many targets for command {cmd.args}")
            node = target_nodes[0]
            if node.name not in nodes:
                nodes[node.name] = (node, [])
            nodes[node.name][1].append(cmd)

        errors = await asyncio.gather(
            *(
                asyncio.create_task(node[0].execute_pipeline(node[1]))
                for node in nodes.values()
            )
        )

        if any(errors):
            if allow_redirections:
                # send each errored command individually
                for cmd in todo:
                    if isinstance(cmd.result, (TryAgainError, MovedError, AskError)):
                        try:
                            cmd.result = await client.execute_command(
                                *cmd.args, **cmd.kwargs
                            )
                        except Exception as e:
                            cmd.result = e

            if raise_on_error:
                for cmd in todo:
                    result = cmd.result
                    if isinstance(result, Exception):
                        command = " ".join(map(safe_str, cmd.args))
                        msg = (
                            f"Command # {cmd.position + 1} ({command}) of pipeline "
                            f"caused error: {result.args}"
                        )
                        result.args = (msg,) + result.args[1:]
                        raise result

            default_node = nodes.get(client.get_default_node().name)
            if default_node is not None:
                # This pipeline execution used the default node, check if we need
                # to replace it.
                # Note: when the error is raised we'll reset the default node in the
                # caller function.
                for cmd in default_node[1]:
                    # Check if it has a command that failed with a relevant
                    # exception
                    if type(cmd.result) in self.__class__.ERRORS_ALLOW_RETRY:
                        client.replace_default_node()
                        break

        return [cmd.result for cmd in stack]

    def _split_command_across_slots(
        self, command: str, *keys: KeyT
    ) -> "ClusterPipeline":
        for slot_keys in self._client._partition_keys_by_slot(keys).values():
            self.execute_command(command, *slot_keys)

        return self

    def mset_nonatomic(
        self, mapping: Mapping[AnyKeyT, EncodableT]
    ) -> "ClusterPipeline":
        encoder = self._client.encoder

        slots_pairs = {}
        for pair in mapping.items():
            slot = key_slot(encoder.encode(pair[0]))
            slots_pairs.setdefault(slot, []).extend(pair)

        for pairs in slots_pairs.values():
            self.execute_command("MSET", *pairs)

        return self


for command in PIPELINE_BLOCKED_COMMANDS:
    command = command.replace(" ", "_").lower()
    if command == "mset_nonatomic":
        continue

    setattr(ClusterPipeline, command, block_pipeline_command(command))


class PipelineCommand:
    def __init__(self, position: int, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.position = position
        self.result: Union[Any, Exception] = None

    def __repr__(self) -> str:
        return f"[{self.position}] {self.args} ({self.kwargs})"
