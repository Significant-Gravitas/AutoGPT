/**
 * CoPilot Bot — Entry point.
 *
 * Loads config, creates adapters, starts the bot.
 * For serverless deployment (Vercel), see src/api/ routes.
 * This file is for standalone / long-running deployment.
 */

import { loadConfig } from "./config.js";
import { createBot } from "./bot.js";

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
  console.log(`💾 State: ${config.redisUrl ? "Redis" : "In-memory"}\n`);

  // Create state adapter
  let stateAdapter;
  if (config.redisUrl) {
    const { createRedisState } = await import("@chat-adapter/state-redis");
    stateAdapter = createRedisState({ url: config.redisUrl });
  } else {
    const { createMemoryState } = await import("@chat-adapter/state-memory");
    stateAdapter = createMemoryState();
  }

  // Create and start the bot
  const bot = await createBot(config, stateAdapter);

  console.log("✅ CoPilot Bot ready.\n");

  // Keep the process alive
  process.on("SIGINT", () => {
    console.log("\n🛑 Shutting down...");
    process.exit(0);
  });

  process.on("SIGTERM", () => {
    console.log("\n🛑 Shutting down...");
    process.exit(0);
  });
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
