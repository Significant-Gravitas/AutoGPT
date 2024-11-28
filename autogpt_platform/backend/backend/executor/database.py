from functools import wraps
from typing import Any, Callable, Concatenate, Coroutine, ParamSpec, TypeVar, cast

from backend.data.credit import get_user_credit_model
from backend.data.execution import (
    ExecutionResult,
    RedisExecutionEventBus,
    create_graph_execution,
    get_execution_results,
    get_incomplete_executions,
    get_latest_execution,
    update_execution_status,
    update_graph_execution_stats,
    update_node_execution_stats,
    upsert_execution_input,
    upsert_execution_output,
)
from backend.data.graph import get_graph, get_node
from backend.data.user import (
    get_user_integrations,
    get_user_metadata,
    update_user_integrations,
    update_user_metadata,
)
from backend.util.service import AppService, expose, register_pydantic_serializers
from backend.util.settings import Config

P = ParamSpec("P")
R = TypeVar("R")
config = Config()


class DatabaseManager(AppService):
    def __init__(self):
        super().__init__()
        self.use_db = True
        self.use_redis = True
        self.event_queue = RedisExecutionEventBus()

    @classmethod
    def get_port(cls) -> int:
        return config.database_api_port

    @expose
    def send_execution_update(self, execution_result: ExecutionResult):
        self.event_queue.publish(execution_result)

    @staticmethod
    def exposed_run_and_wait(
        f: Callable[P, Coroutine[None, None, R]]
    ) -> Callable[Concatenate[object, P], R]:
        @expose
        @wraps(f)
        def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:
            coroutine = f(*args, **kwargs)
            res = self.run_and_wait(coroutine)
            return res

        # Register serializers for annotations on bare function
        register_pydantic_serializers(f)

        return wrapper

    # Executions
    create_graph_execution = exposed_run_and_wait(create_graph_execution)
    get_execution_results = exposed_run_and_wait(get_execution_results)
    get_incomplete_executions = exposed_run_and_wait(get_incomplete_executions)
    get_latest_execution = exposed_run_and_wait(get_latest_execution)
    update_execution_status = exposed_run_and_wait(update_execution_status)
    update_graph_execution_stats = exposed_run_and_wait(update_graph_execution_stats)
    update_node_execution_stats = exposed_run_and_wait(update_node_execution_stats)
    upsert_execution_input = exposed_run_and_wait(upsert_execution_input)
    upsert_execution_output = exposed_run_and_wait(upsert_execution_output)

    # Graphs
    get_node = exposed_run_and_wait(get_node)
    get_graph = exposed_run_and_wait(get_graph)

    # Credits
    user_credit_model = get_user_credit_model()
    get_or_refill_credit = cast(
        Callable[[Any, str], int],
        exposed_run_and_wait(user_credit_model.get_or_refill_credit),
    )
    spend_credits = cast(
        Callable[[Any, str, int, str, dict[str, str], float, float], int],
        exposed_run_and_wait(user_credit_model.spend_credits),
    )

    # User + User Metadata + User Integrations
    get_user_metadata = exposed_run_and_wait(get_user_metadata)
    update_user_metadata = exposed_run_and_wait(update_user_metadata)
    get_user_integrations = exposed_run_and_wait(get_user_integrations)
    update_user_integrations = exposed_run_and_wait(update_user_integrations)
