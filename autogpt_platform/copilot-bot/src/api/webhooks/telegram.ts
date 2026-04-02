/**
 * Telegram webhook endpoint.
 * Deploy as: POST /api/webhooks/telegram
 */

import { getBotInstance } from "../_bot.js";

export async function POST(request: Request) {
  const bot = await getBotInstance();
  const handler = bot.webhooks.telegram;

  if (!handler) {
    return new Response("Telegram adapter not configured", { status: 404 });
  }

  return handler(request);
}
