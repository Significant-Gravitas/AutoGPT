from functools import wraps
from typing import Any, Callable, Concatenate, Coroutine, ParamSpec, TypeVar, cast

from backend.data import credit as C
from backend.data import execution as E
from backend.data import graph as G
from backend.data.queue import RedisEventQueue
from backend.util.service import AppService, expose
from backend.util.settings import Config

P = ParamSpec("P")
R = TypeVar("R")


class DatabaseAPI(AppService):

    def __init__(self):
        super().__init__(port=Config().database_api_port)
        self.use_db = True
        self.use_redis = True
        self.event_queue = RedisEventQueue()

    @expose
    def send_execution_update(self, execution_result_dict: dict[Any, Any]):
        self.event_queue.put(E.ExecutionResult(**execution_result_dict))

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

        return wrapper

    # Executions
    create_graph_execution = exposed_run_and_wait(E.create_graph_execution)
    get_execution_results = exposed_run_and_wait(E.get_execution_results)
    get_incomplete_executions = exposed_run_and_wait(E.get_incomplete_executions)
    get_latest_execution = exposed_run_and_wait(E.get_latest_execution)
    update_execution_status = exposed_run_and_wait(E.update_execution_status)
    update_graph_execution_stats = exposed_run_and_wait(E.update_graph_execution_stats)
    update_node_execution_stats = exposed_run_and_wait(E.update_node_execution_stats)
    upsert_execution_input = exposed_run_and_wait(E.upsert_execution_input)
    upsert_execution_output = exposed_run_and_wait(E.upsert_execution_output)

    # Graphs
    get_node = exposed_run_and_wait(G.get_node)
    get_graph = exposed_run_and_wait(G.get_graph)

    # Credits
    user_credit_model = C.get_user_credit_model()
    get_or_refill_credit = cast(
        Callable[[Any, str], int],
        exposed_run_and_wait(user_credit_model.get_or_refill_credit),
    )
    spend_credits = cast(
        Callable[[Any, str, int, str, dict[str, str], float, float], int],
        exposed_run_and_wait(user_credit_model.spend_credits),
    )
