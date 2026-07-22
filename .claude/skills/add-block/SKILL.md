---
name: add-block
description: Authoritative checklist for adding a new block or provider integration to the AutoGPT Platform. Covers the full lifecycle from provider config → schemas → run method → test triple → OAuth/webhooks → validation. Use when adding/creating a block, integrating a new service, or planning block work.
user-invocable: true
args: "[what the block should do] — service/provider, action, and auth type if known."
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Adding a New Block

This is the authoritative checklist for adding a block to `autogpt_platform/backend/backend/blocks/`. Every step has a **file path** and a **what to do**. Missing a step causes silent failures — blocks that load but never appear (auth filtered), pass tests but break on real API shapes (unfaithful mocks), or run but can't connect to anything in the graph editor.

Canonical references: [Block SDK Guide](../../../docs/platform/block-sdk-guide.md) (provider/auth/cost patterns) and [new_blocks.md](../../../docs/platform/new_blocks.md) (edge cases: webhooks, files, errors).

**Before you start:** Read the existing block most similar to yours and trace it through every file below. `backend/blocks/linear/` is a good modern exemplar (OAuth + API key, `_config.py` / `_api.py` / `models.py` split).

## Design Philosophy — Blocks Are For Non-Technical Users

A block is a node in a visual graph editor used by people who don't read code.

- **One block, one action.** "Search issues" and "create issue" are two blocks, not one block with a mode switch.
- **Composability over completeness.** Inputs and outputs must connect productively to other blocks. Before finalizing schemas, picture the graph: what feeds this block's inputs? What consumes its outputs? An output type nothing can connect to is a design bug.
- **Names and descriptions are UI copy.** `SchemaField(description=...)` text is read by end users in the builder — write it for them, not for developers.

## File Structure

New provider → new folder. Adding a block to an existing provider → reuse its folder and `_config.py`.

```
backend/blocks/your_provider/
  __init__.py        # exports the block classes
  _config.py         # ProviderBuilder + shared test credentials
  _api.py            # API client wrapper (requests, error mapping)
  models.py          # Pydantic data models
  _oauth.py          # OAuth handler (only if OAuth)
  _webhook.py        # WebhooksManager (only if webhook triggers)
  your_action.py     # one block (or a few tightly related ones) per file
```

## Checklist

### 1. Provider config — `_config.py`

```python
from backend.sdk import BlockCostType, ProviderBuilder

your_provider = (
    ProviderBuilder("your_provider")
    .with_api_key(env_var_name="YOUR_PROVIDER_API_KEY", title="Your Provider API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
```

- Auth variants: `.with_api_key(...)`, `.with_oauth(HandlerClass, scopes=..., client_id_env_var=..., client_secret_env_var=...)`, `.with_user_password(...)`, `.with_managed_api_key()`.
- Cost types: `RUN`, `BYTE`, `SECOND`, `ITEMS`, `COST_USD`, `TOKENS`. If the provider bills variably, also call `self.merge_stats(NodeExecutionStats(provider_cost=..., provider_cost_type="cost_usd"))` inside `run()`.
- Define `TEST_CREDENTIALS` / `TEST_CREDENTIALS_INPUT` here once and share across the provider's blocks.

### 2. Block class — `your_action.py`

```python
class YourProviderDoThingBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = your_provider.credentials_field(
            description="..."
        )
        target: str = SchemaField(description="What the user should put here")

    class Output(BlockSchemaOutput):
        result: str = SchemaField(description="What came back")

    def __init__(self):
        super().__init__(
            id="<uuid4>",  # python3 -c "import uuid; print(uuid.uuid4())"
            description="Does the thing",  # end-user copy
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={...},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("result", "expected")],
            test_mock={"do_thing": lambda *a, **kw: "expected"},
        )
```

Registration rules — `load_all_blocks()` (`backend/blocks/__init__.py`) hard-fails on violations:

| Rule | Detail |
|---|---|
| ID is UUID4, 36 chars | Generate fresh; never copy from another block |
| ID unique | Duplicate = registration error across the whole platform |
| Class name ends in `Block` | (`Base` suffix for abstract bases) |
| Module name | lowercase, alphanumeric + underscores |
| Every field is a `SchemaField` | Both Input and Output |
| Boolean inputs have defaults | No optional-and-undefaulted bools |
| `error` output is `str` | If you declare it at all |
| Credentials field naming | `credentials` or `*_credentials` |
| OAuth env vars present | Missing `*_CLIENT_ID`/`*_CLIENT_SECRET` silently filters the block out of the registry |

`__init__` runs at registry load — keep it fast, no I/O, no side effects.

### 3. The run method

- `async def run(self, input_data: Input, *, credentials: ..., **kwargs) -> BlockOutput` — `yield "field_name", value` per output; the executor injects credentials.
- Put API logic in **static methods** (or `_api.py`) so `test_mock` can patch them by name.
- HTTP through `backend.util.request.Requests` (SSRF-safe) — never raw `httpx`/`requests` for user-influenced URLs.
- Errors: raise `BlockInputError` (bad user input) or `BlockExecutionError` (API/runtime failure) from `backend.util.exceptions`, chaining the original (`from e`). Unhandled non-`ValueError` exceptions surface as system errors and page on-call — don't leak them for user-fixable conditions.
- Files: `store_media_file()` from `backend.util.file` — `for_local_processing` (local tools), `for_external_api` (data URI out), and **always `for_block_output` for outputs** (auto-adapts between CoPilot workspace and graph context).

### 4. The test triple — make mocks honest

`test_input` / `test_output` / `test_mock` drive `execute_block_test()` (`backend/util/test.py`), which runs the block with mocks patched and asserts outputs (by value, or by type if you give a type class).

**The mock must mirror the real API's response shape.** Pull the shape from the provider's API docs (or a real call) — a mock returning a convenient invented shape makes the test pass and the block fail in production. If `_api.py` parses responses into Pydantic models, mock at the client-method boundary and return real model instances.

```bash
poetry run pytest 'backend/blocks/test/test_block.py::test_available_blocks[YourProviderDoThingBlock]' -xvs
poetry run pytest backend/blocks/test/test_block.py -x      # whole registry: IDs, schemas, all blocks
```

Never run two pytest invocations concurrently — they exhaust database connection slots.

### 5. OAuth providers (only if `.with_oauth`)

1. `_oauth.py`: handler extending `BaseOAuthHandler` (`backend/integrations/oauth/base.py`) — authorization URL + code-for-token exchange.
2. Register in `backend/integrations/oauth/__init__.py` (`HANDLERS_BY_NAME`).
3. Add `*_client_id` / `*_client_secret` to `Secrets` in `backend/util/settings.py`; document the env vars in `backend/.env.default`.
4. Follow [oauth-integration-flow.md](../../../docs/platform/contributing/oauth-integration-flow.md) for the end-to-end flow, including the frontend credentials UI if the provider is net-new.

### 6. Webhook trigger blocks (only for event-driven blocks)

1. Choose `BlockWebhookConfig` (platform auto-registers via provider API — needs credentials) or `BlockManualWebhookConfig` (user pastes the URL themselves).
2. Input schema: an events-filter model of booleans, plus `payload: dict = SchemaField(hidden=True, default_factory=dict)` (injected by the platform).
3. `_webhook.py`: manager extending `BaseWebhooksManager` — register/unregister/validate/parse.
4. Register the manager in `backend/integrations/webhooks/__init__.py`.
5. Trace `backend/blocks/github/triggers.py` as the reference implementation.

### 7. Final gauntlet

```bash
poetry run format
poetry run lint
poetry run pytest backend/blocks/test/test_block.py -x
```

Then sanity-check the product surface: block name, description, and field descriptions read like UI copy; inputs/outputs connect to plausible neighbor blocks; categories are right (that's how users find it).

## Common Pitfalls

- **Filtered, not failing:** OAuth block missing client env vars loads zero errors and zero block — check the registry filter before debugging "my block doesn't exist".
- **Mock drift:** provider changed their API; the mock still mirrors the old shape. When touching an existing provider, re-verify the shape against current docs.
- **Monolith blocks:** "do X then Y then Z" belongs in a graph of three blocks, not one block. If your Input schema has a `mode` field, split the block.
- **bool without default**, **non-SchemaField field**, **reused UUID** — all caught at registration, but only when blocks actually load; run the registry test, don't assume.
