from graphiti_core.driver.falkordb import STOPWORDS
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.helpers import validate_group_ids


class AutoGPTFalkorDriver(FalkorDriver):
    """FalkorDriver subclass with two AutoGPT-specific tweaks.

    1. ``build_fulltext_query`` adds the per-user ``group_id`` filter so
       multi-tenant searches don't cross user graphs.

    2. ``build_indices`` parameter (defaults True for upstream-compatible
       behaviour) opts out of the fire-and-forget
       ``build_indices_and_constraints`` background task that
       graphiti-core's ``FalkorDriver.__init__`` always spawns.
       That task is fine for long-lived drivers (chat ingest path) but
       generates "Connection closed by server" / "Buffer is closed" log
       spam when the driver is created per short-lived request — most
       notably the admin memory visualizer's per-request driver opens,
       where the indexing task's sequential CREATE INDEX statements
       race the route's own queries and the closing of the connection
       when the route returns. Pass ``build_indices=False`` for
       read-only paths against an existing user's graph; the indices
       are already there from the long-lived chat-write client.
    """

    def __init__(self, *args, build_indices: bool = True, **kwargs):
        # Stash the flag BEFORE super().__init__ runs because
        # FalkorDriver.__init__ fires
        # ``loop.create_task(self.build_indices_and_constraints())``
        # synchronously; our override below reads this attribute when
        # the task actually ticks on the loop.
        self._build_indices_at_init = build_indices
        super().__init__(*args, **kwargs)

    async def build_indices_and_constraints(self) -> None:  # type: ignore[override]
        if not getattr(self, "_build_indices_at_init", True):
            # Caller asserted indices already exist (or will be built by
            # someone else) — skip the multi-CREATE-INDEX race that
            # produces the log spam.
            return
        await super().build_indices_and_constraints()

    def build_fulltext_query(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_query_length: int = 128,
    ) -> str:
        validate_group_ids(group_ids)

        group_filter = ""
        if group_ids:
            group_filter = f"(@group_id:{'|'.join(group_ids)})"

        sanitized_query = self.sanitize(query)
        query_words = sanitized_query.split()
        filtered_words = [word for word in query_words if word.lower() not in STOPWORDS]
        sanitized_query = " | ".join(filtered_words)

        if not sanitized_query:
            fulltext_query = group_filter
        elif not group_filter:
            fulltext_query = f"({sanitized_query})"
        else:
            fulltext_query = f"{group_filter} ({sanitized_query})"

        if len(fulltext_query) >= max_query_length:
            return ""

        return fulltext_query
