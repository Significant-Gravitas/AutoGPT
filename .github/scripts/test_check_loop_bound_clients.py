import io
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import check_loop_bound_clients as checker


class CheckLoopBoundClientsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def check(self, source, filename="module.py"):
        (self.root / filename).write_text(textwrap.dedent(source))
        output = io.StringIO()
        with redirect_stderr(output):
            code = checker.main([str(self.root)])
        return code, output.getvalue()

    def assert_flagged(self, source, *expected):
        code, output = self.check(source)
        self.assertEqual(code, 1, output)
        for text in expected:
            self.assertIn(text, output)
        self.assertIn("decorate the factory with @per_loop_cached", output)

    def test_a_global_set_by_its_getter(self):
        # How backend/copilot/service.py kept its model clients.
        self.assert_flagged(
            """
            from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI

            _main_client = None

            def _get_main_client():
                global _main_client
                if _main_client is None:
                    _main_client = LangfuseAsyncOpenAI(api_key="key")
                return _main_client
            """,
            "_get_main_client() stores AsyncOpenAI(...) in the global _main_client",
        )

    def test_a_factory_cached_for_the_process(self):
        # How backend/util/clients.py kept get_openai_client's client.
        self.assert_flagged(
            """
            from backend.util.cache import cached

            @cached(ttl_seconds=3600)
            def get_openai_client(*, prefer_openrouter=False) -> "AsyncOpenAI | None":
                from openai import AsyncOpenAI
                return AsyncOpenAI(api_key="key")
            """,
            "@cached get_openai_client() keeps one AsyncOpenAI for the process",
        )

    def test_functools_caches_around_a_client(self):
        self.assert_flagged(
            """
            import functools

            import aiohttp
            import httpx

            @functools.cache
            def session():
                return aiohttp.ClientSession()

            @functools.lru_cache(maxsize=1)
            async def http():
                return httpx.AsyncClient()
            """,
            "@cache session() keeps one ClientSession",
            "@lru_cache http() keeps one AsyncClient",
        )

    def test_a_client_built_at_import(self):
        self.assert_flagged(
            """
            import httpx
            from anthropic import AsyncAnthropic

            CLIENT = AsyncAnthropic()

            class Api:
                http: httpx.AsyncClient = httpx.AsyncClient()

            try:
                SESSION = httpx.AsyncClient()
            except ImportError:
                SESSION = None
            """,
            "CLIENT = AsyncAnthropic(...) is built once for the process",
            "http = AsyncClient(...) is built once for the process",
            "SESSION = AsyncClient(...) is built once for the process",
        )

    def test_clients_per_loop_per_call_or_per_object_pass(self):
        code, output = self.check(
            """
            import httpx
            from openai import AsyncOpenAI, DefaultAsyncHttpxClient

            from backend.util.per_loop_cache import per_loop_cached, same_loop_hooks

            @per_loop_cached
            def get_client() -> AsyncOpenAI:
                return AsyncOpenAI(
                    api_key="key",
                    http_client=DefaultAsyncHttpxClient(
                        event_hooks=same_loop_hooks("get_client()")
                    ),
                )

            async def fetch(url):
                async with httpx.AsyncClient() as client:
                    return await client.get(url)

            class Adapter:
                def __init__(self):
                    self._http = httpx.AsyncClient()
            """
        )
        self.assertEqual(code, 0, output)

    def test_tests_are_not_checked(self):
        code, output = self.check(
            "import httpx\nCLIENT = httpx.AsyncClient()\n", filename="module_test.py"
        )
        self.assertEqual(code, 0, output)

    def test_an_allowed_client_passes(self):
        source = "import httpx\nCLIENT = httpx.AsyncClient()\n"
        key = f"{(self.root / 'module.py').resolve().as_posix()}::CLIENT"
        with patch.dict(checker.ALLOWED, {key: "one loop for the process"}):
            code, output = self.check(source)
        self.assertEqual(code, 0, output)


if __name__ == "__main__":
    unittest.main()
