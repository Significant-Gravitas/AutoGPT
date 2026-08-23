# Taskmarket Blocks
<!-- MANUAL: file_description -->
TaskMarket requester blocks for safely preparing, funding, and monitoring Base USDC tasks with explicit human review.
<!-- END MANUAL -->

## Create Task Market Task

### What it is
Creates one Base-funded TaskMarket task only after a fresh, exact human review

### How it works
<!-- MANUAL: how_it_works -->
Consumes an approved review atomically, verifies the reviewed preview fingerprint and spend ceiling, checks the configured TaskMarket wallet on Base, and performs one idempotent task-creation operation. Ambiguous settlement is never retried automatically.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| preview | Exact preview produced by PrepareTaskMarketTaskBlock | TaskMarketTaskPreview | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| task_id | Created TaskMarket task ID | str |
| task_url | Canonical TaskMarket task link | str |
| live_status | Live task state read back after creation | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
Publish a reviewed bounty from an AutoGPT workflow while enforcing an exact maximum USDC spend and retaining the created task ID for audit and monitoring.
<!-- END MANUAL -->

---

## Inspect Task Market Task

### What it is
Reads a TaskMarket task and its submissions without deciding outcomes

### How it works
<!-- MANUAL: how_it_works -->
Uses read-only TaskMarket CLI operations to retrieve the current task state and its submissions. The block never accepts, rejects, rates, or selects a winner.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| task_id | 0x-prefixed TaskMarket task ID | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| task | Current live task state | Dict[str, Any] |
| submissions | Submissions presented for human review | List[Dict[str, Any]] |
| human_review_required | Always true; this block cannot accept or reject work | bool |

### Possible use case
<!-- MANUAL: use_case -->
Monitor a published TaskMarket bounty and surface worker submissions to a human reviewer without allowing the automation to make an irreversible winner decision.
<!-- END MANUAL -->

---

## Prepare Task Market Task

### What it is
Builds an immutable TaskMarket requester preview without moving funds

### How it works
<!-- MANUAL: how_it_works -->
Validates the description, deliverables, reward, hard spend ceiling, deadline, Base network, and canonical Base USDC contract. It emits a frozen preview and SHA-256 fingerprint without making a network write.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| description | Exact public task description | str | Yes |
| deliverables | Exact files or outcomes the worker must deliver | List[str] | Yes |
| reward_usdc | USDC reward to escrow | float \| str | Yes |
| maximum_spend_usdc | Hard operator-approved USDC spend ceiling | float \| str | Yes |
| deadline | Timezone-aware task deadline | str (date-time) | Yes |
| mode | Task selection mode | "bounty" \| "claim" \| "pitch" \| "benchmark" | No |
| tags | Optional discovery tags | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| preview | Immutable Base task preview for human authorization | TaskMarketTaskPreview |
| fingerprint | SHA-256 binding for every preview and spend field | str |

### Possible use case
<!-- MANUAL: use_case -->
Construct a deterministic bounty proposal for human approval before any USDC can be moved, with the exact terms cryptographically bound to the later creation step.
<!-- END MANUAL -->

---
