# Capy Create Thread
<!-- MANUAL: file_description -->
Start a Capy coding-agent thread, on your Capy balance or on a model provider linked in Capy.
<!-- END MANUAL -->

## Capy Create Thread

### What it is
Starts a new Capy agent on a coding task: Capy, an AI software engineer, runs a background coding agent on its own cloud machine against your GitHub repo. Delegate a bug fix, a feature, or work that should open a pull request. Returns at once with a link where the work shows live; follow it with Capy Wait For Thread. The agent then follows its own pull request, fixing failing CI and answering reviews.

### How it works
<!-- MANUAL: how_it_works -->
Calls `POST /api/v1/threads` with the project, the brief, and an optional model, reasoning effort and machine size. The model ID decides who pays: `openai/gpt-6-astra` (or bare `gpt-6-astra`) bills the Capy balance, while `codex/`, `copilot/`, `supergrok/` and `azure/` IDs run the same model through that provider linked in Capy's settings. `model_route` rewrites the prefix for you, so `gpt-6-astra` with the `codex` route becomes `codex/gpt-6-astra`. If the linked provider is disconnected or was never linked, Capy rejects the model and the error says which one to reconnect. With `fall_back_to_capy_balance` on, the block instead reruns the same model on the Capy balance under a derived request ID. Every call carries a `requestId` (generated when left empty), which Capy uses to dedupe, so a retried request never starts a second run. The block returns as soon as the thread exists, with `thread_url` pointing at the thread in the Capy app, where the agent's plan, commands and diff show live. `model_id` and `billed_via` report what it was started with. Once the agent opens a pull request, Capy subscribes the thread to it: a failing check, a review comment or the merge wakes the agent, which fixes and pushes on its own.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| project_id | The Capy project to run in (see Capy List Projects) | str | Yes |
| message | The task for the agent, written as you would brief an engineer: the goal, where to look, what done looks like, and whether to open a pull request | str | Yes |
| title | Thread title. Leave empty to let Capy name it. | str | No |
| model_id | Capy model ID from docs.capy.ai/models-and-pricing, e.g. openai/gpt-6-astra, or a bare name like gpt-6-astra to combine with model_route. Leave empty for the project's default model. | str | No |
| model_route | Who pays for the model. as_given uses model_id exactly as written. capy_balance bills the Capy balance. codex, copilot, supergrok and azure run the model through that provider linked in Capy's settings, so it bills the subscription instead. | "as_given" \| "capy_balance" \| "codex" \| "copilot" \| "supergrok" \| "azure" | No |
| fall_back_to_capy_balance | If the linked provider is disconnected or not linked, run the same model on the Capy balance instead of failing. Off by default, because it moves the cost from the subscription to the balance. | bool | No |
| reasoning | Reasoning effort for the chosen model. Needs model_id. | "" \| "none" \| "instant" \| "minimal" \| "low" \| "medium" \| "high" \| "xhigh" \| "max" | No |
| machine_size | Machine size for the agent's VM. Empty uses Capy's default. | "" \| "small" \| "medium" \| "large" \| "ultra" \| "hyper" \| "bigguy" | No |
| pull_request_author | Who opens the thread's pull requests on GitHub: capy (the Capy GitHub app, so you can approve them yourself where a pull request needs an approving review) or user (the key's owner). Empty follows the Capy settings. Commits keep your Git identity either way. | "" \| "capy" \| "user" | No |
| request_id | Idempotency key. Re-sending the same request_id returns the thread it already created instead of starting a second run. Leave empty to generate one. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| thread | The created thread | Thread |
| thread_id | ID of the created thread | str |
| thread_url | The thread in the Capy app, where the agent's plan, commands and diff show live. Share it when you hand the task off. | str |
| status | The thread's status right after start | str |
| model_id | The model ID the thread was started with; differs from the input when the route rewrote it or the balance fallback ran. Empty means the project's default model. | str |
| billed_via | Who pays for that model: the Capy balance or a linked provider | str |

### Possible use case
<!-- MANUAL: use_case -->
**Bug Fix Handoff**: Hand a support ticket's bug report to a coding agent with the instruction to fix it and open a pull request.

**Subscription Billing**: Run the agent on the team's linked ChatGPT subscription (`model_route: codex`) instead of the Capy balance.

**Resilient Automations**: Turn on `fall_back_to_capy_balance` so a scheduled job still runs when a linked subscription has been disconnected.
<!-- END MANUAL -->

---
