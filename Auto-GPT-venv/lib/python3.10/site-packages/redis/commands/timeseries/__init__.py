import redis

from ..helpers import parse_to_list
from .commands import (
    ALTER_CMD,
    CREATE_CMD,
    CREATERULE_CMD,
    DEL_CMD,
    DELETERULE_CMD,
    GET_CMD,
    INFO_CMD,
    MGET_CMD,
    MRANGE_CMD,
    MREVRANGE_CMD,
    QUERYINDEX_CMD,
    RANGE_CMD,
    REVRANGE_CMD,
    TimeSeriesCommands,
)
from .info import TSInfo
from .utils import parse_get, parse_m_get, parse_m_range, parse_range


class TimeSeries(TimeSeriesCommands):
    """
    This class subclasses redis-py's `Redis` and implements RedisTimeSeries's
    commands (prefixed with "ts").
    The client allows to interact with RedisTimeSeries and use all of it's
    functionality.
    """

    def __init__(self, client=None, **kwargs):
        """Create a new RedisTimeSeries client."""
        # Set the module commands' callbacks
        self.MODULE_CALLBACKS = {
            CREATE_CMD: redis.client.bool_ok,
            ALTER_CMD: redis.client.bool_ok,
            CREATERULE_CMD: redis.client.bool_ok,
            DEL_CMD: int,
            DELETERULE_CMD: redis.client.bool_ok,
            RANGE_CMD: parse_range,
            REVRANGE_CMD: parse_range,
            MRANGE_CMD: parse_m_range,
            MREVRANGE_CMD: parse_m_range,
            GET_CMD: parse_get,
            MGET_CMD: parse_m_get,
            INFO_CMD: TSInfo,
            QUERYINDEX_CMD: parse_to_list,
        }

        self.client = client
        self.execute_command = client.execute_command

        for key, value in self.MODULE_CALLBACKS.items():
            self.client.set_response_callback(key, value)

    def pipeline(self, transaction=True, shard_hint=None):
        """Creates a pipeline for the TimeSeries module, that can be used
        for executing only TimeSeries commands and core commands.

        Usage example:

        r = redis.Redis()
        pipe = r.ts().pipeline()
        for i in range(100):
            pipeline.add("with_pipeline", i, 1.1 * i)
        pipeline.execute()

        """
        if isinstance(self.client, redis.RedisCluster):
            p = ClusterPipeline(
                nodes_manager=self.client.nodes_manager,
                commands_parser=self.client.commands_parser,
                startup_nodes=self.client.nodes_manager.startup_nodes,
                result_callbacks=self.client.result_callbacks,
                cluster_response_callbacks=self.client.cluster_response_callbacks,
                cluster_error_retry_attempts=self.client.cluster_error_retry_attempts,
                read_from_replicas=self.client.read_from_replicas,
                reinitialize_steps=self.client.reinitialize_steps,
                lock=self.client._lock,
            )

        else:
            p = Pipeline(
                connection_pool=self.client.connection_pool,
                response_callbacks=self.MODULE_CALLBACKS,
                transaction=transaction,
                shard_hint=shard_hint,
            )
        return p


class ClusterPipeline(TimeSeriesCommands, redis.cluster.ClusterPipeline):
    """Cluster pipeline for the module."""


class Pipeline(TimeSeriesCommands, redis.client.Pipeline):
    """Pipeline for the module."""
