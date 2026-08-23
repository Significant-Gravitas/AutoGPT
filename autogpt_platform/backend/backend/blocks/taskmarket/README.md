# TaskMarket requester blocks

These blocks let an AutoGPT graph prepare, authorize, create, and monitor a
TaskMarket requester task. Funding is restricted to canonical USDC on Base.

## Prerequisites

Install the first-party `taskmarket` CLI on the backend host and complete its
wallet and legal setup outside AutoGPT. The executable must be directly
available as `taskmarket` on `PATH`; shell wrapper files are intentionally
rejected. Never place a private key, seed phrase, device token, or keystore
contents in a block input.

The backend process must have access to the existing CLI keystore. These blocks
do not read, return, or log that keystore.

## Workflow

1. **Prepare TaskMarket Task** validates the reward, maximum spend, deadline,
   deliverables, Base chain, and canonical Base USDC contract. It emits an
   immutable preview and SHA-256 fingerprint without making a network write.
2. Connect that preview to **Create TaskMarket Task**. The block always creates
   a new, non-editable human review tied to the current node execution. Normal
   sensitive-action settings and reusable auto-approvals cannot bypass it.
3. Review the exact description, deliverables, reward, deadline, Base network,
   USDC contract, and maximum spend in AutoGPT. Approve or reject it there.
4. After approval, the block runs read-only CLI preflights for wallet address,
   chain ID, USDC contract, current legal acceptance, and balance. It then makes
   one `taskmarket task create` call and reads the new task back. The reviewed
   preview fingerprint is supplied as the write's deterministic idempotency key.
5. Use **Inspect TaskMarket Task** to retrieve live status and submissions. Its
   output always sets `human_review_required` to `true`; it exposes no accept,
   reject, rate, or winner-selection operation.

## Settlement safety

A timeout, malformed response, non-zero exit, or missing task ID during task
creation is classified as unknown settlement. The block never retries the
funding call. Its approval is atomically consumed before the write, so concurrent
or replayed executions cannot fund a second task. The first-party create command
is a single write operation, and its deterministic idempotency key is derived
from the exact reviewed preview. If a task ID was returned but the status read
fails, the block preserves the ID and link with an `unknown` status so an
operator can inspect it manually.

## Reproduction

From `autogpt_platform/backend`:

```text
poetry run pytest backend/blocks/taskmarket/taskmarket_test.py -q
poetry run pytest 'backend/blocks/test/test_block.py::test_available_blocks[PrepareTaskMarketTaskBlock]' -xvs
poetry run pytest 'backend/blocks/test/test_block.py::test_available_blocks[CreateTaskMarketTaskBlock]' -xvs
poetry run pytest 'backend/blocks/test/test_block.py::test_available_blocks[InspectTaskMarketTaskBlock]' -xvs
poetry run format
poetry run lint
```

All automated tests use injected command runners and make no network request or
payment. A live demo must stop at the human review unless the operator explicitly
chooses to fund the displayed task.
