/**
 * Singleton bot instance for serverless environments.
 *
 * In serverless (Vercel), each request may hit a cold or warm instance.
 * We create the bot once per instance and reuse it across requests.
 */

import { loadConfig } from "../config.js";
import { createBot } from "../bot.js";

let _botPromise: ReturnType<typeof createBot> | null = null;

export async function getBotInstance() {
  if (!_botPromise) {
    const config = loadConfig();

    let stateAdapter;
    if (config.redisUrl) {
      const { createRedisState } = await import("@chat-adapter/state-redis");
      stateAdapter = createRedisState({ url: config.redisUrl });
    } else {
      const { createMemoryState } = await import("@chat-adapter/state-memory");
      stateAdapter = createMemoryState();
    }

    _botPromise = createBot(config, stateAdapter);
    console.log("[bot] Instance created (serverless)");
  }

  return _botPromise;
}
