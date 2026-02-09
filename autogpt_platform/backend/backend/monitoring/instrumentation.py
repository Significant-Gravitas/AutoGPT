"""
Prometheus instrumentation for FastAPI services.

This module provides centralized metrics collection and instrumentation
for all FastAPI services in the AutoGPT platform.
"""

import logging
from typing import Optional

from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, Info
from prometheus_fastapi_instrumentator import Instrumentator, metrics

logger = logging.getLogger(__name__)

# Custom business metrics with controlled cardinality
GRAPH_EXECUTIONS = Counter(
    "autogpt_graph_executions_total",
    "Total number of graph executions",
    labelnames=[
        "status"
    ],  # Removed graph_id and user_id to prevent cardinality explosion
)

GRAPH_EXECUTIONS_BY_USER = Counter(
    "autogpt_graph_executions_by_user_total",
    "Total number of graph executions by user (sampled)",
    labelnames=["status"],  # Only status, user_id tracked separately when needed
)

BLOCK_EXECUTIONS = Counter(
    "autogpt_block_executions_total",
    "Total number of block executions",
    labelnames=["block_type", "status"],  # block_type is bounded
)

BLOCK_DURATION = Histogram(
    "autogpt_block_duration_seconds",
    "Duration of block executions in seconds",
    labelnames=["block_type"],
    buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60],
)

WEBSOCKET_CONNECTIONS = Gauge(
    "autogpt_websocket_connections_total",
    "Total number of active WebSocket connections",
    # Removed user_id label - track total only to prevent cardinality explosion
)

SCHEDULER_JOBS = Gauge(
    "autogpt_scheduler_jobs",
    "Current number of scheduled jobs",
    labelnames=["job_type", "status"],
)

DATABASE_QUERIES = Histogram(
    "autogpt_database_query_duration_seconds",
    "Duration of database queries in seconds",
    labelnames=["operation", "table"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
)

RABBITMQ_MESSAGES = Counter(
    "autogpt_rabbitmq_messages_total",
    "Total number of RabbitMQ messages",
    labelnames=["queue", "status"],
)

AUTHENTICATION_ATTEMPTS = Counter(
    "autogpt_auth_attempts_total",
    "Total number of authentication attempts",
    labelnames=["method", "status"],
)

API_KEY_USAGE = Counter(
    "autogpt_api_key_usage_total",
    "API key usage by provider",
    labelnames=["provider", "block_type", "status"],
)

# Function/operation level metrics with controlled cardinality
GRAPH_OPERATIONS = Counter(
    "autogpt_graph_operations_total",
    "Graph operations by type",
    labelnames=["operation", "status"],  # create, update, delete, execute, etc.
)

USER_OPERATIONS = Counter(
    "autogpt_user_operations_total",
    "User operations by type",
    labelnames=["operation", "status"],  # login, register, update_profile, etc.
)

RATE_LIMIT_HITS = Counter(
    "autogpt_rate_limit_hits_total",
    "Number of rate limit hits",
    labelnames=["endpoint"],  # Removed user_id to prevent cardinality explosion
)

SERVICE_INFO = Info(
    "autogpt_service",
    "Service information",
)


def instrument_fastapi(
    app: FastAPI,
    service_name: str,
    expose_endpoint: bool = True,
    endpoint: str = "/metrics",
    include_in_schema: bool = False,
    excluded_handlers: Optional[list] = None,
) -> Instrumentator:
    """
    Instrument a FastAPI application with Prometheus metrics.

    Args:
        app: FastAPI application instance
        service_name: Name of the service for metrics labeling
        expose_endpoint: Whether to expose /metrics endpoint
        endpoint: Path for metrics endpoint
        include_in_schema: Whether to include metrics endpoint in OpenAPI schema
        excluded_handlers: List of paths to exclude from metrics

    Returns:
        Configured Instrumentator instance
    """

    # Set service info
    try:
        from importlib.metadata import version

        service_version = version("autogpt-platform-backend")
    except Exception:
        service_version = "unknown"

    SERVICE_INFO.info(
        {
            "service": service_name,
            "version": service_version,
        }
    )

    # Create instrumentator with default metrics
    # Use service-specific inprogress_name to avoid duplicate registration
    # when multiple FastAPI apps are instrumented in the same process
    service_subsystem = service_name.replace("-", "_")
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=excluded_handlers or ["/health", "/readiness"],
        env_var_name="ENABLE_METRICS",
        inprogress_name=f"autogpt_{service_subsystem}_http_requests_inprogress",
        inprogress_labels=True,
    )

    # Add default HTTP metrics
    instrumentator.add(
        metrics.default(
            metric_namespace="autogpt",
            metric_subsystem=service_name.replace("-", "_"),
        )
    )

    # Add request size metrics
    instrumentator.add(
        metrics.request_size(
            metric_namespace="autogpt",
            metric_subsystem=service_name.replace("-", "_"),
        )
    )

    # Add response size metrics
    instrumentator.add(
        metrics.response_size(
            metric_namespace="autogpt",
            metric_subsystem=service_name.replace("-", "_"),
        )
    )

    # Add latency metrics with custom buckets for better granularity
    instrumentator.add(
        metrics.latency(
            metric_namespace="autogpt",
            metric_subsystem=service_name.replace("-", "_"),
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60],
        )
    )

    # Add combined metrics (requests by method and status)
    instrumentator.add(
        metrics.combined_size(
            metric_namespace="autogpt",
            metric_subsystem=service_name.replace("-", "_"),
        )
    )

    # Instrument the app
    instrumentator.instrument(app)

    # Expose metrics endpoint if requested
    if expose_endpoint:
        instrumentator.expose(
            app,
            endpoint=endpoint,
            include_in_schema=include_in_schema,
            tags=["monitoring"] if include_in_schema else None,
        )
        logger.info(f"Metrics endpoint exposed at {endpoint} for {service_name}")

    return instrumentator


def record_graph_execution(graph_id: str, status: str, user_id: str):
    """Record a graph execution event.

    Args:
        graph_id: Graph identifier (kept for future sampling/debugging)
        status: Execution status (success/error/validation_error)
        user_id: User identifier (kept for future sampling/debugging)
    """
    # Track overall executions without high-cardinality labels
    GRAPH_EXECUTIONS.labels(status=status).inc()

    # Optionally track per-user executions (implement sampling if needed)
    # For now, just track status to avoid cardinality explosion
    GRAPH_EXECUTIONS_BY_USER.labels(status=status).inc()


def record_block_execution(block_type: str, status: str, duration: float):
    """Record a block execution event with duration."""
    BLOCK_EXECUTIONS.labels(block_type=block_type, status=status).inc()
    BLOCK_DURATION.labels(block_type=block_type).observe(duration)


def update_websocket_connections(user_id: str, delta: int):
    """Update the number of active WebSocket connections.

    Args:
        user_id: User identifier (kept for future sampling/debugging)
        delta: Change in connection count (+1 for connect, -1 for disconnect)
    """
    # Track total connections without user_id to prevent cardinality explosion
    if delta > 0:
        WEBSOCKET_CONNECTIONS.inc(delta)
    else:
        WEBSOCKET_CONNECTIONS.dec(abs(delta))


def record_database_query(operation: str, table: str, duration: float):
    """Record a database query with duration."""
    DATABASE_QUERIES.labels(operation=operation, table=table).observe(duration)


def record_rabbitmq_message(queue: str, status: str):
    """Record a RabbitMQ message event."""
    RABBITMQ_MESSAGES.labels(queue=queue, status=status).inc()


def record_authentication_attempt(method: str, status: str):
    """Record an authentication attempt."""
    AUTHENTICATION_ATTEMPTS.labels(method=method, status=status).inc()


def record_api_key_usage(provider: str, block_type: str, status: str):
    """Record API key usage by provider and block."""
    API_KEY_USAGE.labels(provider=provider, block_type=block_type, status=status).inc()


def record_rate_limit_hit(endpoint: str, user_id: str):
    """Record a rate limit hit.

    Args:
        endpoint: API endpoint that was rate limited
        user_id: User identifier (kept for future sampling/debugging)
    """
    RATE_LIMIT_HITS.labels(endpoint=endpoint).inc()


def record_graph_operation(operation: str, status: str):
    """Record a graph operation (create, update, delete, execute, etc.)."""
    GRAPH_OPERATIONS.labels(operation=operation, status=status).inc()


def record_user_operation(operation: str, status: str):
    """Record a user operation (login, register, etc.)."""
    USER_OPERATIONS.labels(operation=operation, status=status).inc()
