# CoPilot Bot

Multi-platform chat bot that bridges AutoGPT to Discord, Slack, Telegram, and
Microsoft Teams (WhatsApp next).

## Running

```bash
# As a standalone service
poetry run copilot-bot

# Or auto-start alongside the rest of the platform
poetry run app   # starts the bot too if AUTOPILOT_BOT_DISCORD_TOKEN is set
```

### Docker (local dev)

The bot's link/unlink flow talks to `platform_linking_manager` over cluster-
internal RPC. The service runs as its own pod in dev/prod but is **opt-in**
locally via the `bot` Docker Compose profile so it doesn't slow regular
`docker compose up`:

```bash
# Start the linking manager alongside your existing stack
docker compose up -d platform_linking_manager

# Or include it via the profile flag (also brings up anything else profile-tagged)
docker compose --profile bot up -d
```

Without it running, `/setup` and other token-issuing flows fail with an
`httpx.ConnectError` from the backend trying to reach a missing host.

## Required environment variables

See `backend/.env.default` for the full list with documentation. Minimum setup:

| Variable | Purpose |
|----------|---------|
| `AUTOPILOT_BOT_DISCORD_TOKEN` | Discord bot token — enables the Discord (socket) adapter |
| `AUTOPILOT_BOT_SLACK_TOKEN` + `AUTOPILOT_BOT_SLACK_SIGNING_SECRET` | Slack bot token + signing secret — set **both** to mount the Slack (webhook / Events API) adapter on the main backend API |
| `AUTOPILOT_BOT_TELEGRAM_TOKEN` + `AUTOPILOT_BOT_TELEGRAM_WEBHOOK_SECRET` | Telegram BotFather token + webhook secret — set **both** to mount the Telegram (webhook / Bot API) adapter, then register the webhook once with `setWebhook` (see `.env.default`). Also disable group privacy mode via BotFather `/setprivacy` or the bot can't see @mentions in groups |
| `MICROSOFT_CLIENT_ID` + `MICROSOFT_CLIENT_SECRET` + `MICROSOFT_TENANT_ID` | Azure Bot registration — set **all three** to mount the Teams (webhook / Bot Connector) adapter, and point the bot's messaging endpoint at `/api/copilot-webhooks/teams/messages`. The tenant id is mandatory: new registrations are single-tenant only since 2025-07-31 |
| `FRONTEND_BASE_URL` | Frontend base URL for link confirmation pages (shared with the rest of the backend) |
| `REDIS_HOST` / `REDIS_PORT` | Session + thread subscription state + copilot stream subscription (inherited from the shared backend config) |
| `PLATFORMLINKINGMANAGER_HOST` | DNS name of the `PlatformLinkingManager` service pod (cluster-internal RPC) |

## Architecture

```
bot/
├── app.py              # CoPilotChatBridge(AppService), adapter factory, outbound @expose RPC
├── config.py           # Shared (platform-agnostic) config
├── handler.py          # Orchestrator: routing, linking, attachment ingestion, batching
├── turn_stream.py      # Streaming one batched turn: chunked sends, artifacts, renames
├── prompt.py           # Prompt + thread-name assembly
├── attachments.py      # Attachment upload + failure notes
├── command_core.py     # Shared /setup + /unlink policy (adapters render it)
├── bot_backend.py     # Thin facade over PlatformLinkingManagerClient + stream_registry
├── text.py             # Text splitting + batch formatting
├── threads.py          # Redis-backed thread subscription tracking
├── webhook_routes.py   # Mounts webhook adapters' inbound routes on the main API
└── adapters/
    ├── base.py         # PlatformAdapter (outbound) + SocketAdapter / WebhookAdapter + MessageContext
    ├── shared.py       # Platform-agnostic adapter helpers (attachments, history budget, bot-loop guard)
    ├── discord/        # SocketAdapter — Discord Gateway
    │   ├── adapter.py  # Gateway connection, events, sends, thread creation
    │   ├── commands.py # Slash commands (/setup, /help, /unlink)
    │   └── config.py   # Discord token + platform limits
    ├── slack/          # WebhookAdapter — Slack Events API
    │   ├── adapter.py       # Inbound event/command routes, sends, mrkdwn, attachments
    │   ├── commands.py      # Slash commands (/setup, /help, /unlink)
    │   ├── config.py        # Slack token + signing secret + platform limits
    │   ├── signing.py       # HMAC-SHA256 request signature verification
    │   ├── text.py          # CommonMark → Slack mrkdwn
    │   └── app-manifest.yaml # Importable Slack app definition (scopes, events, commands)
    ├── telegram/       # WebhookAdapter — Telegram Bot API
    │   ├── adapter.py       # Inbound updates route, sends, chat-model mapping
    │   ├── api_client.py    # Thin httpx Bot API client (JSON + multipart + getFile)
    │   ├── commands.py      # Bot commands (/setup, /help, /unlink)
    │   ├── config.py        # BotFather token + webhook secret + platform limits
    │   └── text.py          # CommonMark → Telegram HTML
    └── teams/          # WebhookAdapter — Bot Framework Connector
        ├── adapter.py       # Inbound activities route, sends, conversation mapping
        ├── api_client.py    # Bot Connector REST client (per-activity serviceUrl)
        ├── auth.py          # Inbound JWT validation + outbound token minting
        ├── commands.py      # Bot commands (/setup, /help, /unlink)
        ├── config.py        # App id + password + tenant id + platform limits
        └── text.py          # CommonMark → Teams-flavoured markdown
```

### Microsoft Teams notes

Teams differs from the other platforms in ways worth knowing before debugging:

- **Auth is a signed JWT, not an HMAC of the body.** The Connector's token
  authenticates the *sender*, not the payload, so a valid token could be
  replayed with any body. The `serviceUrl` claim binding in `auth.py` is what
  closes that — note the wire claim is lowercase `serviceurl`.
- **Replies go to the activity's `serviceUrl`**, not a fixed base URL, because
  Microsoft routes each tenant/region to its own Connector host. That makes it
  attacker-influenced input we attach a bearer token to, hence the host
  allowlist.
- **Single-tenant only.** Microsoft stopped issuing multi-tenant registrations
  after 2025-07-31, so `MICROSOFT_TENANT_ID` is mandatory.
- **No native slash commands.** In a channel the bot only receives messages
  that @mention it — commands included, e.g. `@AutoGPT /setup`. A bare
  `/setup` in a channel never reaches us.
- **Group chats are not supported.** A `groupChat` carries no team identity to
  bill against, so those activities are accepted and ignored (with a log line
  saying so). Only `personal` chats and `channel` messages are handled.
- **Delivery is at-most-once.** The route ACKs 200 and dispatches in a
  background task, and the activity id is claimed in Redis before processing.
  A dispatch failure or a replica recycle mid-turn therefore loses that turn,
  and the Connector's own redelivery is deduped away by the claim. This trades
  the Connector's retry for never double-answering a user; releasing the claim
  on failure is the change to make if the balance should tip the other way.
- **The learned `serviceUrl` map is per process.** The adapter rides the
  N-replica API, so another replica handling the same conversation falls back
  to the default public-cloud host. Fine while replies are reactive (the host
  comes off the inbound activity) and DMs use the default; a proactive channel
  send from a cold replica would 404 for a tenant Microsoft routes elsewhere,
  so persist the mapping in Redis before enabling that.
- **Mention matching is prefix-insensitive.** Teams spells a participant id
  `28:<app-id>` in some places and bare in others; `_bot_identities` accepts
  either, and also matches the activity's own `recipient.id`.

#### Local testing without a tenant

The [M365 Agents Playground](https://learn.microsoft.com/microsoftteams/platform/)
simulates Teams on `localhost:56150` with no tenant, tunnel, or app
registration — useful because sideloading a real app needs an M365 licence.
It sends **no** Bot Framework JWT, so set `APP_ENV=local` and
`AUTOPILOT_BOT_TEAMS_ALLOW_UNVERIFIED=true` to accept unsigned activities.
Both gates are required; neither works on a deployed environment.

When the backend runs in Docker, the Playground's loopback `serviceUrl` is
rewritten to `host.docker.internal` so replies can reach it on the host
(`auth.rewrite_loopback_for_container`) — also local-mode only.

#### App package

`adapters/teams/` holds the sideloadable app package: `manifest.json` plus
`color.png` (192×192) and `outline.png` (32×32, transparent, monochrome).
Replace both `<AZURE_BOT_APP_ID>` placeholders with the Azure Bot's app id,
then zip the three files **flat** (no containing folder):

```bash
cd backend/copilot/bot/adapters/teams
zip ../../../../../AutoGPT-teams.zip manifest.json color.png outline.png
```

Upload via Teams → Apps → Manage your apps → Upload an app. Sideloading needs
an M365 licence and the tenant's app-upload policy enabled.

**Connector taxonomy.** `PlatformAdapter` is the outbound contract the core
handler speaks through. Concrete adapters extend one of two subtypes by how
inbound events arrive:

- **`SocketAdapter`** — owns a long-lived connection (Discord Gateway, Slack
  Socket Mode). Driven by `start`/`stop`; runs in the `copilot-bot` pod.
  Built in `app.py::_build_socket_adapters`.
- **`WebhookAdapter`** — receives inbound HTTPS POSTs (Slack Events API,
  Telegram, Teams, WhatsApp). Stateless; its `register_routes(app)` mounts onto
  the main backend API (via `webhook_routes.register_webhook_adapters`), so it
  rides the existing N-replica deployment — no dedicated pod. Built in
  `webhook_routes._build_webhook_adapters`.

**Locality rule:** anything platform-specific lives under `adapters/<platform>/`.
The only files that name specific platforms are the two factories above, which
decide which adapters to instantiate based on which tokens are set.

## How messaging works

1. User mentions the bot in a channel
2. Adapter's `on_message` handler fires, constructs a `MessageContext`, passes
   it to the shared `MessageHandler`
3. Handler:
   - Checks if the user/server is linked (via `bot_backend`)
   - If not linked → sends a "Link Account" button prompt
   - If linked → creates a thread (for channels) or uses the existing thread/DM
   - Marks the thread as subscribed in Redis (7-day TTL)
   - Streams the AutoPilot response back, chunked at the adapter's
     `chunk_flush_at` boundary
4. Messages that arrive while a stream is running get batched and sent as a
   single follow-up turn once the current stream ends

## Adding a new platform

1. Create `adapters/<platform>/` with `adapter.py`, `commands.py` (if the
   platform has commands), and `config.py`
2. `adapter.py` subclasses **`SocketAdapter`** (long-lived connection) or
   **`WebhookAdapter`** (inbound HTTPS) and implements all the outbound
   `PlatformAdapter` methods — `max_message_length`, `chunk_flush_at`,
   `send_message`, `send_link`, `create_thread`, `send_file`, etc. — plus its
   subtype's inbound method (`start`/`stop` or `register_routes`).
3. `config.py` declares the platform's env vars and any platform-specific
   numbers (message limits, token name, etc.)
4. Register it in the matching factory, gated on its token(s):
   - Socket → `app.py::_build_socket_adapters`
   - Webhook → `webhook_routes.py::_build_webhook_adapters`
   ```python
   if <platform>_config.get_bot_token():
       adapters.append(<Platform>Adapter(api))
   ```

The core handler, text utilities, thread tracking, and platform API all stay
untouched.
