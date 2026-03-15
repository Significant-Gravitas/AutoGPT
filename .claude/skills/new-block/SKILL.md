---
name: new-block
description: Create a new backend block following the Block SDK Guide. TRIGGER when user asks to create a new block, add a new integration, or build a new node for the graph editor.
user-invocable: true
metadata:
  author: autogpt-team
  version: "1.1.0"
---

# New Block

Read `docs/platform/block-sdk-guide.md` first for the full guide.

## Steps

1. **Provider config** (if external service): create `_config.py` with `ProviderBuilder`
2. **Block file** in `backend/blocks/` (from `autogpt_platform/backend/`):
   - Generate UUID once with `uuid.uuid4()`, then **hard-code** as `id` (must be stable across imports)
   - `Input(BlockSchema)` and `Output(BlockSchema)` classes
   - `async def run` that `yield`s output fields
3. **Files**: use `store_media_file()` with `"for_block_output"` for outputs
4. **Test**: `poetry run pytest 'backend/blocks/test/test_block.py::test_available_blocks[MyBlock]' -xvs`
5. **Format**: run `/check`

## Rules

- Analyze interfaces: do inputs/outputs connect well with other blocks in a graph?
- Apply /code-style rules (top-level imports, no duck typing, Pydantic models)
- Always use `for_block_output` for block outputs
