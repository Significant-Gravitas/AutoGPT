"""Which engine a load-test account's chat turns run on.

A load test sends turns from throwaway accounts at a preview environment that
people are using with the real model at the same time, so the choice is made
per account rather than for the whole process (as ``CHAT_TEST_MODE`` does).
``ChatConfig.load_test_email_domain`` names the accounts' email domain, and
the address picks the engine:

- ``loadtest-dummy-…@<domain>``: the dummy engine (``sdk/dummy.py``). The
  route, queue, executor and stream relay all run, with a canned reply and no
  model, sandbox or memory work.
- ``loadtest-scripted-…@<domain>``: the SDK engine, its CLI pointed at
  ``ChatConfig.load_test_model_url``, where the scripted model plays the
  provider. The whole engine runs, sandbox and tool loop included, with no
  model spend.
- Any other address: the account's usual engine and model.

With the domain unset, no account is routed and no email is looked up.
"""

import logging
from typing import Literal

from backend.copilot.config import ChatConfig
from backend.data.db_accessors import user_db
from backend.util.cache import cached

logger = logging.getLogger(__name__)

LoadTestEngine = Literal["dummy", "scripted"]

_ENGINE_BY_PREFIX: dict[str, LoadTestEngine] = {
    "loadtest-dummy-": "dummy",
    "loadtest-scripted-": "scripted",
}


def engine_for_email(email: str, domain: str) -> LoadTestEngine | None:
    """The load-test engine *email* asks for, if it is in *domain*."""
    local, at, email_domain = email.strip().lower().rpartition("@")
    if not at or not domain or email_domain != domain.strip().lower():
        return None
    for prefix, engine in _ENGINE_BY_PREFIX.items():
        if local.startswith(prefix):
            return engine
    return None


# An account's email doesn't change. Only a found address is cached: a
# lookup that raises is retried on the account's next turn.
@cached(maxsize=1000, ttl_seconds=3600)
async def _email_for(user_id: str) -> str:
    fields = await user_db().get_auth_user_flag_fields(user_id)
    if fields is None or not fields.email:
        raise LookupError(f"No email for user {user_id}")
    return fields.email


async def load_test_engine(
    user_id: str | None, config: ChatConfig
) -> LoadTestEngine | None:
    """The engine *user_id*'s turns run on as a load-test account, or None
    for an account that isn't one (and whenever the routing is off)."""
    domain = config.load_test_email_domain
    if not domain or not user_id:
        return None
    try:
        email = await _email_for(user_id)
    except LookupError:
        return None
    engine = engine_for_email(email, domain)
    if engine is not None:
        logger.info("Load-test account %s: %s engine", user_id, engine)
    return engine
