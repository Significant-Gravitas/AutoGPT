from backend.data import db


def chat_db():
    if db.is_connected():
        from backend.copilot import db as _chat_db

        chat_db = _chat_db
    else:
        from backend.util.clients import get_database_manager_async_client

        chat_db = get_database_manager_async_client()

    return chat_db


def graph_db():
    if db.is_connected():
        from backend.data import graph as _graph_db

        graph_db = _graph_db
    else:
        from backend.util.clients import get_database_manager_async_client

        graph_db = get_database_manager_async_client()

    return graph_db


def library_db():
    if db.is_connected():
        from backend.api.features.library import db as _library_db

        library_db = _library_db
    else:
        from backend.util.clients import get_database_manager_async_client

        library_db = get_database_manager_async_client()

    return library_db


def store_db():
    if db.is_connected():
        from backend.api.features.store import db as _store_db

        store_db = _store_db
    else:
        from backend.util.clients import get_database_manager_async_client

        store_db = get_database_manager_async_client()

    return store_db


def search():
    if db.is_connected():
        from backend.api.features.store import hybrid_search as _search

        search = _search
    else:
        from backend.util.clients import get_database_manager_async_client

        search = get_database_manager_async_client()

    return search


def execution_db():
    if db.is_connected():
        from backend.data import execution as _execution_db

        execution_db = _execution_db
    else:
        from backend.util.clients import get_database_manager_async_client

        execution_db = get_database_manager_async_client()

    return execution_db


def user_db():
    if db.is_connected():
        from backend.data import user as _user_db

        user_db = _user_db
    else:
        from backend.util.clients import get_database_manager_async_client

        user_db = get_database_manager_async_client()

    return user_db


def understanding_db():
    if db.is_connected():
        from backend.data import understanding as _understanding_db

        understanding_db = _understanding_db
    else:
        from backend.util.clients import get_database_manager_async_client

        understanding_db = get_database_manager_async_client()

    return understanding_db


def workspace_db():
    if db.is_connected():
        from backend.data import workspace as _workspace_db

        workspace_db = _workspace_db
    else:
        from backend.util.clients import get_database_manager_async_client

        workspace_db = get_database_manager_async_client()

    return workspace_db
