"""How a shared cache's operations show up on the caller's Sentry transaction."""

import sentry_sdk

from backend.util import cache as cache_module
from backend.util import tracing
from backend.util.cache import cached


class _SpanPerPageRedis:
    """A shared-cache Redis with Sentry's integration on: each SCAN page and
    pipeline execution is a ``db.redis`` span."""

    def __init__(self, keys: list[str], page_size: int):
        self._keys = keys
        self._page_size = page_size
        self.deleted: list[str] = []

    def scan_iter(self, match, target_nodes=None):
        for start in range(0, len(self._keys), self._page_size):
            with sentry_sdk.start_span(op="db.redis", name=f"SCAN {start}"):
                page = self._keys[start : start + self._page_size]
            yield from page

    def pipeline(self):
        deleted = self.deleted

        class _Pipeline:
            def delete(self, key):
                deleted.append(key)

            def execute(self):
                with sentry_sdk.start_span(op="db.redis", name="pipeline"):
                    pass

        return _Pipeline()


def test_clearing_a_shared_cache_is_one_span_however_many_pages(
    sentry_transactions, monkeypatch
):
    """The sweep reads every primary page by page; a span per page took the
    caller's transaction past the SDK's span limit."""
    keys = [f"cache:swept:{i}" for i in range(50)]
    redis = _SpanPerPageRedis(keys, page_size=2)
    monkeypatch.setattr(cache_module, "_get_redis", lambda: redis)

    @cached(ttl_seconds=30, shared_cache=True)
    def swept(x: int) -> int:
        return x

    with tracing.start_transaction("http.server", "/api/auth/user/timezone"):
        swept.cache_clear()

    assert redis.deleted == keys
    [request] = sentry_transactions
    assert [span["description"] for span in request["spans"]] == [
        "SCAN cache keys",
        "pipeline",
    ]
