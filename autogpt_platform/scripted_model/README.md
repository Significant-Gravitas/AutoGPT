# Scripted model

A fake model provider for tests. A Playwright journey drives the real copilot
pipeline (engine, tool routing and execution, persistence, streaming and
resume, cost accounting) while this server plays the model, turn by turn, from
a script. `CHAT_TEST_MODE`'s canned reply skips all of that.

It only runs where `docker-compose.journeys.yml` is added: CI's journeys leg
and local test stacks. Self-hosted and hosted deployments never start it.

## How the platform reaches it

`docker-compose.journeys.yml` puts the backend in OpenRouter mode with
`CHAT_BASE_URL=http://scripted_model:4010/api/v1`, so both engines and every
side call come here:

| Caller | Request | Dialect |
| --- | --- | --- |
| SDK engine (the Claude Agent SDK's CLI, the default) | `POST /api/v1/messages` | Anthropic Messages, streamed |
| Baseline engine (`FORCE_FLAG_COPILOT_SDK=false`) | `POST /api/v1/chat/completions` | OpenAI chat completions, streamed |
| Side calls (session title) | `POST /api/v1/chat/completions` | non-streamed, no tools |

The CLI also sends `HEAD /api/api/hello` to check it can reach the server,
and may ask `POST /api/v1/messages/count_tokens`; both are answered.

## Scripts

A journey picks a script with a marker in its chat message:
`[[script:<name>]]`, or `[[script:<name>#<nonce>]]` to find its own requests
in the journal. Scripts are JSON files in
`frontend/src/playwright/journeys/model-scripts/`, mounted at `/scripts` and
read on every request (edits apply without a restart). A journey imports its
script to know what to expect.

```json
{
  "description": "What the journey checks.",
  "title": "Answer to side calls, e.g. the chat's title",
  "turns": [
    {
      "text": "Let me check your library first.",
      "tool_calls": [{ "name": "find_library_agent", "input": { "query": "doubler" } }],
      "usage": { "input_tokens": 3000, "output_tokens": 40,
                 "cache_read_input_tokens": 20000, "cache_creation_input_tokens": 0 },
      "stream": { "chunk_words": 3, "delay_ms": 15 }
    },
    {
      "expect_tool_results": [{ "tool": "find_library_agent", "contains": "No agents" }],
      "text": "You don't have one yet. Want me to build it?",
      "usage": { "input_tokens": 3400, "output_tokens": 20 }
    },
    {
      "expect_user_text": "Yes, build it",
      "text": "On it.",
      "usage": { "input_tokens": 3500, "output_tokens": 5 }
    }
  ]
}
```

- **Which turn answers a request:** turn `n` answers the request whose
  conversation already holds `n` assistant messages. Requests carry the whole
  conversation, so the server keeps no state: concurrent runs of one script,
  a retried turn and a resumed session all land on the right turn. A
  follow-up the user queued during a turn reaches the model after that turn's
  answer, so it gets the next turn.
- **Tools:** name them as the baseline engine does (`find_library_agent`).
  The SDK engine offers them as `mcp__copilot__find_library_agent`; the server
  calls whichever the request offers. Most platform tools (`create_agent`,
  `edit_agent` and others) are deferred: on both engines the model reaches
  them through `run_capability` with `{"id": "tool:<name>", "input": {...}}`,
  and `create_agent` needs the building guide read first
  (`{"id": "skill:agent_building_guide"}`). See
  `model-scripts/build-doubler-agent.json`.
- **Checks** are logged, and a journey asserts on them through the journal:
  `expect_tool_results` against the tool results the platform sends back,
  `expect_user_text` against what the user said since the last answer.
- **Streaming:** text goes out `chunk_words` words at a time, `delay_ms`
  apart. Slow it down to give a journey time to cut the connection.

A request the script can't answer fails the turn with a 400 that says why:
no marker, an unknown script, a turn past the script's end (an extra tool
round or retry), or a tool the platform didn't offer. A journey fails loudly
instead of waiting on a reply that never comes. A request that offers no tools
is a side call and gets the script's `title`.

## Cost

The server keeps the provider's invoice, priced in `invoice.py` from
Anthropic's list prices (the same as `backend/copilot/anthropic_rates.json`):

- chat completions report it as `usage.cost`, as OpenRouter does, which is
  what the baseline engine charges;
- the SDK engine charges the CLI's own estimate, priced from the same list.
  Message ids are `msg_…`, not OpenRouter's `gen-…`, so the platform doesn't
  try to look the cost up on openrouter.ai.

Each journal entry records `usage` and `cost_usd`, so a journey can check that
the platform recorded exactly what the provider billed.

## Journal

`GET http://localhost:4010/__scripted__/journal?nonce=<nonce>` (or
`?script=<name>`) lists every request: dialect, model, which turn answered it,
the tool results it carried, expectation checks, usage, cost, and whether the
whole reply reached the platform (`completed`). Refused requests are there
with their reason. The container log has one line per request.

## Run it

With the stack, add the override file:

```bash
cd autogpt_platform
docker compose -f docker-compose.yml -f docker-compose.journeys.yml up -d
```

On its own (standard library only, Python 3.10+):

```bash
cd autogpt_platform/scripted_model
SCRIPTED_MODEL_SCRIPTS_DIR=../frontend/src/playwright/journeys/model-scripts \
  SCRIPTED_MODEL_PORT=4010 python -m scripted_model
```

Tests, with the backend's environment (they use the real `anthropic` and
`openai` clients):

```bash
cd autogpt_platform/backend
poetry run pytest ../scripted_model/tests
```
