# CoPilot Bot

Multi-platform bot service for AutoGPT CoPilot, built with [Vercel Chat SDK](https://chat-sdk.dev).

Deploys CoPilot to Discord, Telegram, Slack, and more from a single codebase.

## How it works

1. User messages the bot on any platform (Discord, Telegram, Slack)
2. Bot checks if the platform user is linked to an AutoGPT account
3. If not linked → sends a one-time link URL
4. User clicks → logs in to AutoGPT → accounts are linked
5. Future messages are forwarded to CoPilot and responses streamed back

## Setup

```bash
# Install dependencies
npm install

# Copy env template
cp .env.example .env

# Configure at least one platform adapter (e.g. Discord)
# Edit .env with your bot tokens

# Run in development
npm run dev
```

## Architecture

```
src/
├── index.ts           # Standalone entry point
├── config.ts          # Environment-based configuration
├── bot.ts             # Core bot logic (Chat SDK handlers)
├── platform-api.ts    # AutoGPT platform API client
└── api/               # Serverless API routes (Vercel)
    ├── _bot.ts        # Singleton bot instance
    ├── webhooks/      # Platform webhook endpoints
    │   ├── discord.ts
    │   ├── telegram.ts
    │   └── slack.ts
    └── gateway/
        └── discord.ts # Gateway cron for Discord messages
```

## Deployment

### Standalone (Docker/PM2)
```bash
npm run build
npm start
```

### Serverless (Vercel)
Deploy to Vercel. Webhook URLs:
- Discord: `https://your-app.vercel.app/api/webhooks/discord`
- Telegram: `https://your-app.vercel.app/api/webhooks/telegram`
- Slack: `https://your-app.vercel.app/api/webhooks/slack`

For Discord messages (Gateway), add a cron job in `vercel.json`:
```json
{
  "crons": [{ "path": "/api/gateway/discord", "schedule": "*/9 * * * *" }]
}
```

## Dependencies

- [Chat SDK](https://chat-sdk.dev) — Cross-platform bot abstraction
- AutoGPT Platform API — Account linking + CoPilot chat
