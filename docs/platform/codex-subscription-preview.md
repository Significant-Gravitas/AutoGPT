# ChatGPT/Codex subscription transport preview

This private preview lets a user connect a ChatGPT account and use that
account's Codex allowance for three explicit execution paths:

- The existing **Code Generation** block through the `codex_app_server`
  transport.
- A newly created AutoPilot chat session when **ChatGPT/Codex** is the only
  connected subscription transport, or is selected from multiple connected
  subscription transports.
- An **AutoPilot** graph block with an explicitly selected ChatGPT/Codex
  connection.

`codex` is the credential provider and Codex App Server is the execution
runtime. A ChatGPT plan is not unlimited inference: these calls consume the
connected account's Codex allowance or credits and remain subject to its rate
limits. Subscription-backed usage records tokens and
`billing_mode=user_subscription`; it is not reported as AutoGPT provider USD
spend.

The preview does not route Orchestrator, shared LLM blocks, image generation,
or background system work through the connected account. Existing graphs and
AutoPilot sessions keep their stored route. New AutoPilot sessions use a
connected subscription transport automatically when exactly one is available.

## Run locally

From `autogpt_platform`:

```sh
docker compose up -d --build
```

Codex pairing and its supported execution routes are available without feature
flag environment variables. The stack mounts `/run/autogpt-codex` as a
memory-backed temporary filesystem for REST, executor, and Copilot executor
processes.

Open `http://localhost:3000`, then open **Settings > Integrations** and connect
**Codex**. In the sign-in window, open the ChatGPT verification page and enter
the one-time code.

### Test the Code Generation block

1. Add a **Code Generation** block to a graph.
2. Set **Transport** to **Codex App Server**.
3. Select the connected **ChatGPT for Codex** credential.
4. Run the graph.

Existing graphs continue to default to the OpenAI API transport. The block's
model dropdown applies to that API transport. The subscription transport uses
App Server's current account-compatible default; its live catalog can change
independently of public API model names.

### Test AutoPilot

1. Open **AutoPilot** and start a new task.
2. If ChatGPT/Codex is the only connected subscription transport, it is used
   automatically and no connection selector is shown. If multiple subscription
   transports are connected, choose the desired connection before sending the
   first message.
3. Choose Fast or Thinking and Balanced or Advanced as usual, then send the
   message. Text and permitted AutoGPT tool calls stream through the existing
   AutoPilot event surface.

The route and credential are stored on the new session and are immutable for
that session. Start another task to use a different connected subscription
transport. The server reloads the authoritative session route before each
queued turn; queued messages carry only the credential ID, never the OAuth
tokens. A missing, revoked, or busy Codex credential fails visibly. There is no
silent fallback to an AutoGPT-funded model or another user's credential.

The mode and model controls select a Codex route from the shared model catalog,
then validate it against the models advertised by the connected account. The
preview maps Fast/Balanced to GPT-5.6 Luna, Fast/Advanced and Thinking/Balanced
to GPT-5.6 Terra, and Thinking/Advanced to GPT-5.6 Sol when the account exposes
them. It otherwise uses the visible account default. File attachments,
agent-building tools, and SDK sub-sessions use the same Claude Agent SDK path as
platform-funded AutoPilot. Builder-panel-bound sessions remain platform-funded
in this preview because their persistent session is created without an AI
connection selector.

The Copilot executor keeps one exclusive credential lease and one Codex runtime
per connected account, then multiplexes overlapping chats onto separate Codex
threads. Canceling or finishing one chat does not stop its siblings. This pool
is process-local: a request routed to another executor process or replica cannot
join the owner yet and still fails with the bounded, retryable
`codex_credential_busy` error. A hosted multi-replica rollout therefore needs
credential-owner routing or a dedicated bridge. The normal platform AutoPilot
route is unchanged.

### Test the AutoPilot graph block

1. Add an **AutoPilot** block to a graph.
2. Select the saved connection under **ChatGPT / Codex connection**.
3. Run the graph.

The credential field is a reference-only credential input: graph validation
checks ownership and export strips the credential ID, while the graph executor
does not hold the outer credential lease. The Copilot executor acquires the
single runtime lease when it consumes the queued turn. This preserves normal
credential rebinding without deadlocking the nested execution path. Leaving the
field empty preserves the existing platform-funded behavior.

## Claude Agent SDK transport

AutoPilot keeps the existing Claude Agent SDK and Claude Code CLI as its agent
harness. A request-scoped loopback Anthropic Messages endpoint translates model
rounds to the pinned Codex App Server's experimental `dynamicTools` protocol.
Claude Code still owns the MCP tool loop, permission hooks, builder behavior,
transcript, resume, compaction, and sub-session machinery. Codex supplies the
model response using the selected user's ChatGPT/Codex credential.

The loopback endpoint is bound to `127.0.0.1` on an ephemeral port and accepts
only a random capability generated for that turn. The capability is placed in
the Claude CLI child environment; ChatGPT access, refresh, and ID tokens are
never placed there. Anthropic text and tool-use events are mapped to the same
Claude SDK response adapter used by normal AutoPilot.

The App Server process itself is fail-closed: native shell, filesystem, web
search, apps, plugins, environments, workspace roots, and approval requests are
disabled, and its sandbox is read-only. Any side effect comes only through a
tool that Claude Code was already permitted to invoke through AutoGPT's MCP and
security hooks.

The implementation pins `openai-codex` and its bundled runtime together at
`0.144.4`. Do not float either dependency independently. The Claude Agent SDK
and bundled CLI are also pinned by the backend lock. Because `dynamicTools` and
the compatibility surface are version-sensitive, updates require the focused
protocol, Messages conformance, and bundled-CLI tests to pass before rollout.

## Verify an existing local login

To test an existing local Codex login before starting the stack, run this from
`autogpt_platform/backend`:

```sh
poetry run python -m scripts.codex_preview_smoke
```

The smoke test copies, never moves, `~/.codex/auth.json` into an isolated home,
checks account, models, and rate limits, performs one subscription-backed turn,
and fails if the runtime can read a host canary or mutates the source login. It
does consume a small amount of the connected account's Codex usage.

## Deploy a private cloud preview

Codex pairing and supported execution routes do not require availability flags
or frontend build-time overrides. Scope a private preview with the deployment's
ordinary access controls instead of hiding device login with environment
variables. The transport selector is driven by the connected, compatible
subscription providers: it is hidden for zero or one and shown when more than
one is available.

REST, executor, and Copilot executor must share the normal AutoGPT credential
database, encryption configuration, and Redis cluster. Give every process that
may launch App Server an isolated, memory-backed `CODEX_TEMP_ROOT`; never mount
it on a persistent volume.

Set `FRONTEND_BASE_URL`, `NEXT_PUBLIC_FRONTEND_BASE_URL`, and `BETTER_AUTH_URL`
to the exact public origin of the preview, including its `https://` scheme and
without a path. REST deliberately builds the device-login URL from this
configured origin instead of trusting forwarded host headers. Leaving a
production or localhost origin in a cloud preview sends the popup to a host
that does not have the preview's Better Auth cookie.

`NEXT_PUBLIC_FRONTEND_BASE_URL` is compiled into the frontend bundle. Pass it as
a frontend image build argument, then set `BETTER_AUTH_URL` on the frontend at
runtime and `FRONTEND_BASE_URL` on REST at runtime. The development Compose file
forwards all three exported values and keeps `http://localhost:3000` as their
local default.

Device-login status and ownership are Redis-backed, so polling and cancellation
can reach different REST replicas. The active App Server login actor is still
process-local, however: an owner-pod restart interrupts the login and another
replica cannot take it over or resume it. Keep rolling deploys away from active
sign-ins and ask the user to restart an interrupted login. Broad cloud rollout
requires moving the login actor into the dedicated bridge described below.

The containers need outbound HTTPS access to OpenAI's ChatGPT/Codex endpoints.
No callback ingress from OpenAI is required because Codex App Server polls for
device-code completion.

## Preview safety boundary

- The encrypted, USER-scoped `IntegrationCredential` row remains the source of
  truth.
- Raw ChatGPT tokens do not enter Redis, RabbitMQ, frontend responses,
  container-wide environment variables, or AutoPilot session records.
- Each native invocation or shared AutoPilot runtime actor materializes auth
  into an isolated temporary home. An AutoPilot actor keeps that home only
  while one or more overlapping chats are attached, checkpoints Codex-managed
  refresh throughout its lifetime and at shutdown, then cleans the home.
- Code Generation turns expose no dynamic tools or host workspace. AutoPilot
  exposes only the tools registered by its existing Claude SDK/MCP harness.
- Runtime reuse is scoped to the exact user and credential. Credentials are
  never pooled across users or substituted for another account.
- There is no silent platform-key fallback.

The current implementation launches Codex App Server in-process from the
ordinary REST, executor, and Copilot executor containers. Those containers
currently run as root and retain the backend services' broader database, Redis,
filesystem, and network privileges. The temporary-home and fail-closed thread
configuration reduce exposure, but they do not make this a safe boundary for a
broad external cloud rollout.

Treat the current shape as local development or a tightly access-controlled
staff/private preview only. Before enabling external users, move runtime and
device-login supervision into a dedicated unprivileged bridge with a non-root
user, read-only root filesystem, dropped capabilities, memory-only credential
homes, process and concurrency limits, narrowly restricted egress, and a
least-privilege internal credential lease interface. The bridge must also own
login actor takeover or explicit restart semantics. Complete the commercial
and policy review for hosted subscription passthrough before broader rollout.
