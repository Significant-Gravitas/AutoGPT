from contextlib import contextmanager

from fastapi import HTTPException

from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError


@contextmanager
def raise_400_on_activation_error():
    """Map `GraphActivationError` (raised by `before_graph_activate` or by
    library-DB helpers that call it) to an HTTP 400 with the error's message,
    so every route surfacing graph activation errors does the same thing.

    Usage:
        with raise_400_on_activation_error():
            graph = await before_graph_activate(graph, user_id=user_id)
    """
    try:
        yield
    except GraphActivationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
