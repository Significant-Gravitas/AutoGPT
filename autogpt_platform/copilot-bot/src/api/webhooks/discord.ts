/**
 * Discord webhook endpoint.
 *
 * Receives HTTP Interactions from Discord (button clicks, slash commands,
 * verification pings). For regular messages, the Gateway WebSocket is
 * used via the adapter's built-in connection.
 *
 * Deploy as: POST /api/webhooks/discord
 */

import { getBotInstance } from "../_bot.js";

export async function POST(request: Request) {
  const bot = getBotInstance();
  const handler = bot.webhooks.discord;

  if (!handler) {
    return new Response("Discord adapter not configured", { status: 404 });
  }

  return handler(request);
}
