"""Do-not-merge probe (#15739 integration preview), v4: where a turn's time
goes between the executor picking it up and the query reaching the CLI.

Every awaited step ``stream_chat_completion_sdk`` makes before it sends the
query is timed (``perf-step <name> <ms>``): the setup gather's legs (system
prompt, graphiti warm context, CLI session restore, E2B sandbox), the steps
after StreamStart (model route, turn budget, tier, CLI prelaunch, pending
drain, session and skills context, user context, budget block, query build,
attachments, builder context, skills update, transcript seed), and inside
memory recall the query embedding, the reranker and graphiti's edge-search
methods, plus the follow-up refresh and its join. Each line names the asyncio
task it ran in, so a turn's steps can be told from background ingestion's.
"""

import asyncio
import inspect
import logging
import time

logger = logging.getLogger(__name__)

_SERVICE_STEPS = (
    "_build_system_prompt",
    "_fetch_graphiti_context",
    "_restore_cli_session_for_turn",
    "get_or_create_sandbox",
    "active_mode",
    "_resolve_sdk_model_for_request",
    "resolve_turn_budget",
    "get_user_tier",
    "drain_pending_safe",
    "build_session_context",
    "build_skills_context",
    "inject_user_context",
    "persist_pending_as_user_rows",
    "build_turn_budget_block",
    "_build_query_message",
    "_prepare_file_attachments",
    "_maybe_prepend_builder_context",
    "_maybe_prepend_skills_update",
    "_seed_transcript",
)
_SEARCH_METHODS = ("edge_fulltext_search", "edge_similarity_search", "edge_bfs_search")
_installed = False


def install(service_module) -> None:
    global _installed
    if _installed:
        return
    _installed = True
    for name in _SERVICE_STEPS:
        _wrap_function(service_module, name, name)
    _wrap_prelaunch(service_module)
    _wrap_memory()
    logger.info("perf-step installed")


def _wrap_function(module, attribute: str, label: str) -> None:
    original = getattr(module, attribute, None)
    if original is None or not inspect.iscoroutinefunction(original):
        logger.info("perf-step skip %s (missing or not async)", label)
        return

    async def timed(*args, **kwargs):
        started = time.perf_counter()
        try:
            return await original(*args, **kwargs)
        finally:
            _log(label, started)

    setattr(module, attribute, timed)


def _wrap_prelaunch(service_module) -> None:
    prelaunch = getattr(service_module, "CLIPrelaunch", None)
    start = getattr(prelaunch, "start", None)
    if start is None:
        logger.info("perf-step skip CLIPrelaunch.start")
        return
    original = start.__func__

    async def timed(cls, *args, **kwargs):
        started = time.perf_counter()
        try:
            return await original(cls, *args, **kwargs)
        finally:
            _log("CLIPrelaunch.start", started)

    prelaunch.start = classmethod(timed)


def _wrap_memory() -> None:
    from graphiti_core.embedder import openai as graphiti_embedder
    from graphiti_core.search import search as graphiti_search

    from backend.copilot.graphiti import context_refresh, reranker

    _wrap_function(context_refresh, "refresh_warm_context", "refresh_warm_context")
    _wrap_function(context_refresh, "join_refresh", "join_refresh")
    _wrap_function(graphiti_search, "search", "graphiti.search")
    for method in _SEARCH_METHODS:
        _wrap_function(graphiti_search, method, method)
    _wrap_method(graphiti_embedder.OpenAIEmbedder, "create", "embed")
    _wrap_method(reranker.CompatOpenAIRerankerClient, "rank", "rerank")


def _wrap_method(cls, attribute: str, label: str) -> None:
    original = getattr(cls, attribute, None)
    if original is None:
        logger.info("perf-step skip %s", label)
        return

    async def timed(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return await original(self, *args, **kwargs)
        finally:
            _log(label, started)

    setattr(cls, attribute, timed)


def _log(label: str, started: float) -> None:
    task = asyncio.current_task()
    logger.info(
        "perf-step %s %.0f task=%s",
        label,
        (time.perf_counter() - started) * 1000,
        task.get_name()[:28] if task else "-",
    )
