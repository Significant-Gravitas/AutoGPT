from redis import DataError
from redis.exceptions import ResponseError

from .exceptions import VersionMismatchException
from .execution_plan import ExecutionPlan
from .query_result import AsyncQueryResult, QueryResult

PROFILE_CMD = "GRAPH.PROFILE"
RO_QUERY_CMD = "GRAPH.RO_QUERY"
QUERY_CMD = "GRAPH.QUERY"
DELETE_CMD = "GRAPH.DELETE"
SLOWLOG_CMD = "GRAPH.SLOWLOG"
CONFIG_CMD = "GRAPH.CONFIG"
LIST_CMD = "GRAPH.LIST"
EXPLAIN_CMD = "GRAPH.EXPLAIN"


class GraphCommands:
    """RedisGraph Commands"""

    def commit(self):
        """
        Create entire graph.
        """
        if len(self.nodes) == 0 and len(self.edges) == 0:
            return None

        query = "CREATE "
        for _, node in self.nodes.items():
            query += str(node) + ","

        query += ",".join([str(edge) for edge in self.edges])

        # Discard leading comma.
        if query[-1] == ",":
            query = query[:-1]

        return self.query(query)

    def query(self, q, params=None, timeout=None, read_only=False, profile=False):
        """
        Executes a query against the graph.
        For more information see `GRAPH.QUERY <https://redis.io/commands/graph.query>`_. # noqa

        Args:

        q : str
            The query.
        params : dict
            Query parameters.
        timeout : int
            Maximum runtime for read queries in milliseconds.
        read_only : bool
            Executes a readonly query if set to True.
        profile : bool
            Return details on results produced by and time
            spent in each operation.
        """

        # maintain original 'q'
        query = q

        # handle query parameters
        query = self._build_params_header(params) + query

        # construct query command
        # ask for compact result-set format
        # specify known graph version
        if profile:
            cmd = PROFILE_CMD
        else:
            cmd = RO_QUERY_CMD if read_only else QUERY_CMD
        command = [cmd, self.name, query, "--compact"]

        # include timeout is specified
        if isinstance(timeout, int):
            command.extend(["timeout", timeout])
        elif timeout is not None:
            raise Exception("Timeout argument must be a positive integer")

        # issue query
        try:
            response = self.execute_command(*command)
            return QueryResult(self, response, profile)
        except ResponseError as e:
            if "unknown command" in str(e) and read_only:
                # `GRAPH.RO_QUERY` is unavailable in older versions.
                return self.query(q, params, timeout, read_only=False)
            raise e
        except VersionMismatchException as e:
            # client view over the graph schema is out of sync
            # set client version and refresh local schema
            self.version = e.version
            self._refresh_schema()
            # re-issue query
            return self.query(q, params, timeout, read_only)

    def merge(self, pattern):
        """
        Merge pattern.
        """
        query = "MERGE "
        query += str(pattern)

        return self.query(query)

    def delete(self):
        """
        Deletes graph.
        For more information see `DELETE <https://redis.io/commands/graph.delete>`_. # noqa
        """
        self._clear_schema()
        return self.execute_command(DELETE_CMD, self.name)

    # declared here, to override the built in redis.db.flush()
    def flush(self):
        """
        Commit the graph and reset the edges and the nodes to zero length.
        """
        self.commit()
        self.nodes = {}
        self.edges = []

    def bulk(self, **kwargs):
        """Internal only. Not supported."""
        raise NotImplementedError(
            "GRAPH.BULK is internal only. "
            "Use https://github.com/redisgraph/redisgraph-bulk-loader."
        )

    def profile(self, query):
        """
        Execute a query and produce an execution plan augmented with metrics
        for each operation's execution. Return a string representation of a
        query execution plan, with details on results produced by and time
        spent in each operation.
        For more information see `GRAPH.PROFILE <https://redis.io/commands/graph.profile>`_. # noqa
        """
        return self.query(query, profile=True)

    def slowlog(self):
        """
        Get a list containing up to 10 of the slowest queries issued
        against the given graph ID.
        For more information see `GRAPH.SLOWLOG <https://redis.io/commands/graph.slowlog>`_. # noqa

        Each item in the list has the following structure:
        1. A unix timestamp at which the log entry was processed.
        2. The issued command.
        3. The issued query.
        4. The amount of time needed for its execution, in milliseconds.
        """
        return self.execute_command(SLOWLOG_CMD, self.name)

    def config(self, name, value=None, set=False):
        """
        Retrieve or update a RedisGraph configuration.
        For more information see `https://redis.io/commands/graph.config-get/>`_. # noqa

        Args:

        name : str
            The name of the configuration
        value :
            The value we want to set (can be used only when `set` is on)
        set : bool
            Turn on to set a configuration. Default behavior is get.
        """
        params = ["SET" if set else "GET", name]
        if value is not None:
            if set:
                params.append(value)
            else:
                raise DataError(
                    "``value`` can be provided only when ``set`` is True"
                )  # noqa
        return self.execute_command(CONFIG_CMD, *params)

    def list_keys(self):
        """
        Lists all graph keys in the keyspace.
        For more information see `GRAPH.LIST <https://redis.io/commands/graph.list>`_. # noqa
        """
        return self.execute_command(LIST_CMD)

    def execution_plan(self, query, params=None):
        """
        Get the execution plan for given query,
        GRAPH.EXPLAIN returns an array of operations.

        Args:
            query: the query that will be executed
            params: query parameters
        """
        query = self._build_params_header(params) + query

        plan = self.execute_command(EXPLAIN_CMD, self.name, query)
        if isinstance(plan[0], bytes):
            plan = [b.decode() for b in plan]
        return "\n".join(plan)

    def explain(self, query, params=None):
        """
        Get the execution plan for given query,
        GRAPH.EXPLAIN returns ExecutionPlan object.
        For more information see `GRAPH.EXPLAIN <https://redis.io/commands/graph.explain>`_. # noqa

        Args:
            query: the query that will be executed
            params: query parameters
        """
        query = self._build_params_header(params) + query

        plan = self.execute_command(EXPLAIN_CMD, self.name, query)
        return ExecutionPlan(plan)


class AsyncGraphCommands(GraphCommands):
    async def query(self, q, params=None, timeout=None, read_only=False, profile=False):
        """
        Executes a query against the graph.
        For more information see `GRAPH.QUERY <https://oss.redis.com/redisgraph/master/commands/#graphquery>`_. # noqa

        Args:

        q : str
            The query.
        params : dict
            Query parameters.
        timeout : int
            Maximum runtime for read queries in milliseconds.
        read_only : bool
            Executes a readonly query if set to True.
        profile : bool
            Return details on results produced by and time
            spent in each operation.
        """

        # maintain original 'q'
        query = q

        # handle query parameters
        query = self._build_params_header(params) + query

        # construct query command
        # ask for compact result-set format
        # specify known graph version
        if profile:
            cmd = PROFILE_CMD
        else:
            cmd = RO_QUERY_CMD if read_only else QUERY_CMD
        command = [cmd, self.name, query, "--compact"]

        # include timeout is specified
        if isinstance(timeout, int):
            command.extend(["timeout", timeout])
        elif timeout is not None:
            raise Exception("Timeout argument must be a positive integer")

        # issue query
        try:
            response = await self.execute_command(*command)
            return await AsyncQueryResult().initialize(self, response, profile)
        except ResponseError as e:
            if "unknown command" in str(e) and read_only:
                # `GRAPH.RO_QUERY` is unavailable in older versions.
                return await self.query(q, params, timeout, read_only=False)
            raise e
        except VersionMismatchException as e:
            # client view over the graph schema is out of sync
            # set client version and refresh local schema
            self.version = e.version
            self._refresh_schema()
            # re-issue query
            return await self.query(q, params, timeout, read_only)

    async def execution_plan(self, query, params=None):
        """
        Get the execution plan for given query,
        GRAPH.EXPLAIN returns an array of operations.

        Args:
            query: the query that will be executed
            params: query parameters
        """
        query = self._build_params_header(params) + query

        plan = await self.execute_command(EXPLAIN_CMD, self.name, query)
        if isinstance(plan[0], bytes):
            plan = [b.decode() for b in plan]
        return "\n".join(plan)

    async def explain(self, query, params=None):
        """
        Get the execution plan for given query,
        GRAPH.EXPLAIN returns ExecutionPlan object.

        Args:
            query: the query that will be executed
            params: query parameters
        """
        query = self._build_params_header(params) + query

        plan = await self.execute_command(EXPLAIN_CMD, self.name, query)
        return ExecutionPlan(plan)

    async def flush(self):
        """
        Commit the graph and reset the edges and the nodes to zero length.
        """
        await self.commit()
        self.nodes = {}
        self.edges = []
