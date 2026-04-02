/**
 * Slack webhook endpoint.
 * Deploy as: POST /api/webhooks/slack
 */

import { getBotInstance } from "../_bot.js";

export async function POST(request: Request) {
  const bot = await getBotInstance();
  const handler = bot.webhooks.slack;

  if (!handler) {
    return new Response("Slack adapter not configured", { status: 404 });
  }

  return handler(request);
}
