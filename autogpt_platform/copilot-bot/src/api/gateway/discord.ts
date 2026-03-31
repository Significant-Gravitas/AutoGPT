/**
 * Discord Gateway cron endpoint.
 *
 * In serverless environments, Discord's Gateway WebSocket needs a
 * persistent connection to receive messages. This endpoint is called
 * by a cron job every 9 minutes to maintain the connection.
 */

import { getBotInstance } from "../_bot.js";

export async function GET(request: Request) {
  // Verify cron secret in production
  const authHeader = request.headers.get("authorization");
  if (
    process.env.CRON_SECRET &&
    authHeader !== `Bearer ${process.env.CRON_SECRET}`
  ) {
    return new Response("Unauthorized", { status: 401 });
  }

  const bot = await getBotInstance();
  await bot.initialize();

  const discord = bot.getAdapter("discord");
  if (!discord) {
    return new Response("Discord adapter not configured", { status: 404 });
  }

  const baseUrl = process.env.WEBHOOK_BASE_URL ?? "http://localhost:3000";
  const webhookUrl = `${baseUrl}/api/webhooks/discord`;
  const durationMs = 10 * 60 * 1000; // 10 minutes

  return (discord as any).startGatewayListener(
    {},
    durationMs,
    undefined,
    webhookUrl
  );
}
