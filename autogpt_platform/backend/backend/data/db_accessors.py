from types import ModuleType
from typing import TYPE_CHECKING, TypeVar

from backend.data import db
from backend.util import clients

if TYPE_CHECKING:
    from backend.data.db_manager import DatabaseManagerAsyncClient

M = TypeVar("M", bound=ModuleType)


def db_for(module: M, *, should_retry: bool = True) -> "M | DatabaseManagerAsyncClient":
    """``module``'s database functions, wherever this process can run them.

    The module itself in a process with a Prisma connection, else the database
    manager's RPC client, which offers the functions the manager exposes.
    """
    if db.is_connected():
        return module
    if should_retry:
        # No argument, so every default caller shares one thread-cached client.
        return clients.get_database_manager_async_client()
    return clients.get_database_manager_async_client(should_retry=False)


def auth_session_db():
    from backend.data.auth import session as module

    return db_for(module)


def background_task_db():
    from backend.copilot import background_task_db as module

    return db_for(module)


def chat_db():
    from backend.copilot import db as module

    return db_for(module)


def chat_db_is_local() -> bool:
    """Whether ``chat_db()`` is this process's own Prisma client, not the
    database manager's RPC."""
    return db.is_connected()


def experts_db():
    from backend.api.features.experts import experts_db as module

    return db_for(module)


def expert_kickoff_db():
    from backend.api.features.experts import kickoff as module

    return db_for(module)


def expert_check_db():
    from backend.api.features.experts import experts_db as module

    # One attempt: a database manager from before check_private_expert
    # answers at once that it has no such route, where the retrying
    # client would wait up to five minutes for its replacement. The
    # executor's expert check then reads the expert another way.
    return db_for(module, should_retry=False)


def feedback_prompts_db():
    from backend.data import feedback_prompts as module

    return db_for(module)


def spend_approval_db():
    from backend.api.features.experts import spend_approval as module

    return db_for(module)


def graph_db():
    from backend.data import graph as module

    return db_for(module)


def library_db():
    from backend.api.features.library import db as module

    return db_for(module)


def store_db():
    from backend.api.features.store import db as module

    return db_for(module)


def triggers_db():
    from backend.api.features.library import triggers as module

    return db_for(module)


def search():
    from backend.api.features.search import hybrid_search as module

    return db_for(module)


def embeddings_db():
    from backend.api.features.search import embeddings as module

    return db_for(module)


def execution_db():
    from backend.data import execution as module

    return db_for(module)


def user_db():
    from backend.data import user as module

    return db_for(module)


def memory_schedule_db():
    from backend.data import memory_schedule as module

    return db_for(module)


def understanding_db():
    from backend.data import understanding as module

    return db_for(module)


def workspace_db():
    from backend.data import workspace as module

    return db_for(module)


def workspace_folder_db():
    from backend.data import workspace_folder as module

    return db_for(module)


def skill_db():
    """Marketplace skill listings: active versions and packages, read by the
    copy reconcile in copilot.tools.skills."""
    from backend.api.features.store import skill_db as module

    return db_for(module)


def workspace_skill_db():
    from backend.data import workspace_skill as module

    return db_for(module)


def review_db():
    from backend.data import human_review as module

    return db_for(module)


def credit_db():
    from backend.data import db_manager as module

    return db_for(module)


def platform_cost_db():
    from backend.data import platform_cost as module

    return db_for(module)


def activity_event_db():
    from backend.data import activity_event as module

    return db_for(module)


def orgs_db():
    from backend.api.features.orgs import db as module

    return db_for(module)


def platform_linking_db():
    from backend.platform_linking import db as module

    return db_for(module)


def bot_analytics_db():
    from backend.data import bot_analytics as module

    return db_for(module)


def skill_learning_db():
    from backend.data import skill_learning as module

    return db_for(module)


def skill_reviews_db():
    from backend.data import skill_reviews as module

    return db_for(module)


def skill_publication_db():
    from backend.data import skill_publication as module

    return db_for(module)


def skill_use_db():
    from backend.data import skill_use as module

    return db_for(module)


def skill_deletion_db():
    from backend.data import skill_deletion as module

    return db_for(module)


def skill_versions_db():
    from backend.data import skill_versions as module

    return db_for(module)


def bot_installs_db():
    from backend.data import bot_installs as module

    return db_for(module)


def dream_db():
    from backend.data import dream_pass as module

    # The dream record is written under a short deadline that already
    # covers retrying (copilot/dream/store.py); a retrying client would
    # only keep going on a request the deadline has given up on.
    return db_for(module, should_retry=False)


def pro_activation_db():
    from backend.data import pro_activation as module

    return db_for(module)


def posthog_lifecycle_db():
    from backend.data import posthog_lifecycle_sync as module

    # Best-effort and asked for after the work is queued: fail fast.
    return db_for(module, should_retry=False)


def scheduled_chats_db():
    from backend.copilot.scheduled_chats import forget as module

    return db_for(module)
