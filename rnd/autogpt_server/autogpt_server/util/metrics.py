import sentry_sdk
from sentry_sdk import metrics

from autogpt_server.util.settings import Settings

sentry_dsn = Settings().secrets.sentry_dsn
sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0, profiles_sample_rate=1.0)


def emit_distribution(
    name: str,
    key: str,
    value: float,
    unit: str = "none",
    tags: dict[str, str] | None = None,
):
    metrics.distribution(
        key=f"{name}__{key}",
        value=value,
        unit=unit,
        tags=tags or {},
    )


def metric_node_payload(key: str, value: float, tags: dict[str, str]):
    emit_distribution("NODE_EXECUTION", key, value, unit="byte", tags=tags)


def metric_node_timing(key: str, value: float, tags: dict[str, str]):
    emit_distribution("NODE_EXECUTION", key, value, unit="second", tags=tags)


def metric_graph_count(key: str, value: int, tags: dict[str, str]):
    emit_distribution("GRAPH_EXECUTION", key, value, tags=tags)


def metric_graph_timing(key: str, value: float, tags: dict[str, str]):
    emit_distribution("GRAPH_EXECUTION", key, value, unit="second", tags=tags)
