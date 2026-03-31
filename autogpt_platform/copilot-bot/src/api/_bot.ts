/**
 * Singleton bot instance for serverless environments.
 *
 * In serverless (Vercel), each request may hit a cold or warm instance.
 * We create the bot once per instance and reuse it across requests.
 */

import { loadConfig } from "../config.js";
import { createBot } from "../bot.js";
import type { Chat } from "chat";

let _bot: ReturnType<typeof createBot> | null = null;

export function getBotInstance() {
  if (!_bot) {
    const config = loadConfig();

    // In serverless, always use in-memory state unless Redis is configured
    let stateAdapter;
    if (config.redisUrl) {
      const { createRedisState } = require("@chat-adapter/state-redis");
      stateAdapter = createRedisState({ url: config.redisUrl });
    } else {
      const { createMemoryState } = require("@chat-adapter/state-memory");
      stateAdapter = createMemoryState();
    }

    _bot = createBot(config, stateAdapter);
    console.log("[bot] Instance created (serverless)");
  }

  return _bot;
}
