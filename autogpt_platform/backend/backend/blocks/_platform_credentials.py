"""Keep blocks that store objects with a vendor off the platform's shared keys.

A platform-owned key is a "Use Credits for X" system credential or an SDK
``<provider>-default`` credential. Blocks that create objects in the vendor
account, such as an Exa Webset or a SmartLead campaign, keep them under the
user's own key instead. They set ``refuses_platform_credentials = True``, so
graph validation refuses those keys before the graph runs, and call
``refuse_platform_credentials`` before their first vendor call as the backstop
for runs that don't go through graph validation.
"""

from backend.blocks._base import Block
from backend.data.model import Credentials, is_sdk_default, provider_label
from backend.integrations.credentials_store import is_system_credential
from backend.util.exceptions import BlockExecutionError


def refuse_platform_credentials(block: Block, credentials: Credentials) -> None:
    if not is_platform_credential(credentials.id):
        return
    raise BlockExecutionError(
        message=platform_credentials_refused_message(credentials.provider),
        block_name=block.name,
        block_id=block.id,
    )


def platform_credentials_refused_message(provider: str) -> str:
    label = provider_label(provider)
    return (
        f"This block keeps what it creates in the {label} account behind "
        f"its key, so it can't use the platform's shared {label} key. "
        f"Connect your own {label} API key and select it instead."
    )


def is_platform_credential(credentials_id: str) -> bool:
    return is_system_credential(credentials_id) or is_sdk_default(credentials_id)
