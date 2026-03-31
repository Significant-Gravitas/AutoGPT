/**
 * CoPilot Bot — Entry point (standalone / long-running).
 *
 * Starts an HTTP server for webhook handling and connects to
 * the Discord Gateway for receiving messages.
 */

import { loadConfig } from "./config.js";
import { createBot } from "./bot.js";

const PORT = parseInt(process.env.PORT ?? "3001", 10);

async function main() {
  console.log("🤖 CoPilot Bot starting...\n");

  const config = loadConfig();

  // Log which adapters are enabled
  const enabled = [
    config.discord && "Discord",
    config.telegram && "Telegram",
    config.slack && "Slack",
  ].filter(Boolean);

  console.log(`📡 Adapters: ${enabled.join(", ") || "none"}`);
  console.log(`🔗 API: ${config.autogptApiUrl}`);
  console.log(`💾 State: ${config.redisUrl ? "Redis" : "In-memory"}`);
  console.log(`🌐 Port: ${PORT}\n`);

  // Create state adapter
  let stateAdapter;
  if (config.redisUrl) {
    const { createRedisState } = await import("@chat-adapter/state-redis");
    stateAdapter = createRedisState({ url: config.redisUrl });
  } else {
    const { createMemoryState } = await import("@chat-adapter/state-memory");
    stateAdapter = createMemoryState();
  }

  // Create the bot
  const bot = await createBot(config, stateAdapter);

  // Start HTTP server for webhooks
  await startNodeServer(bot, PORT);

  // Start Discord Gateway if enabled
  if (config.discord) {
    await bot.initialize();
    const discord = bot.getAdapter("discord") as any;

    if (discord?.startGatewayListener) {
      const webhookUrl = `http://localhost:${PORT}/api/webhooks/discord`;
      console.log(`🔌 Starting Discord Gateway → ${webhookUrl}`);

      // Run gateway in background, restart on disconnect
      const runGateway = async () => {
        while (true) {
          try {
            await discord.startGatewayListener(
              {},
              10 * 60 * 1000, // 10 minutes
              undefined,
              webhookUrl
            );
            console.log("[gateway] Listener ended, restarting...");
          } catch (err) {
            console.error("[gateway] Error, restarting in 5s:", err);
            await new Promise((r) => setTimeout(r, 5000));
          }
        }
      };

      // Don't await — run in background
      runGateway();
    }
  }

  console.log("\n✅ CoPilot Bot ready.\n");

  // Graceful shutdown
  process.on("SIGINT", () => {
    console.log("\n🛑 Shutting down...");
    process.exit(0);
  });
  process.on("SIGTERM", () => {
    console.log("\n🛑 Shutting down...");
    process.exit(0);
  });
}

/**
 * Start a simple HTTP server using Node's built-in http module.
 * Routes webhook requests to the Chat SDK bot.
 */
async function startNodeServer(bot: any, port: number) {
  const { createServer } = await import("http");

  const server = createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", `http://localhost:${port}`);

    // Collect body
    const chunks: Buffer[] = [];
    for await (const chunk of req) {
      chunks.push(chunk as Buffer);
    }
    const body = Buffer.concat(chunks);

    // Build a standard Request object for Chat SDK
    const headers = new Headers();
    for (const [key, value] of Object.entries(req.headers)) {
      if (value) headers.set(key, Array.isArray(value) ? value[0] : value);
    }

    const request = new Request(url.toString(), {
      method: req.method ?? "POST",
      headers,
      body: req.method !== "GET" && req.method !== "HEAD" ? body : undefined,
    });

    // Route to the correct adapter webhook
    // URL: /api/webhooks/{platform}
    const parts = url.pathname.split("/");
    const platform = parts[parts.length - 1];
    const handler = platform ? (bot.webhooks as any)[platform] : undefined;

    if (!handler) {
      res.writeHead(404);
      res.end("Not found");
      return;
    }

    try {
      const response: Response = await handler(request);
      res.writeHead(response.status, Object.fromEntries(response.headers));
      const responseBody = await response.arrayBuffer();
      res.end(Buffer.from(responseBody));
    } catch (err) {
      console.error(`[http] Error handling ${url.pathname}:`, err);
      res.writeHead(500);
      res.end("Internal error");
    }
  });

  server.listen(port, () => {
    console.log(`🌐 HTTP server listening on http://localhost:${port}`);
  });

  return server;
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
