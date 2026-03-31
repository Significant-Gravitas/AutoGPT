/**
 * Discord webhook endpoint.
 * Deploy as: POST /api/webhooks/discord
 */

import { getBotInstance } from "../_bot.js";

export async function POST(request: Request) {
  const bot = await getBotInstance();
  const handler = bot.webhooks.discord;

  if (!handler) {
    return new Response("Discord adapter not configured", { status: 404 });
  }

  return handler(request);
}
