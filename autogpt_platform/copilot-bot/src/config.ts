import "dotenv/config";

export interface Config {
  /** AutoGPT platform API base URL */
  autogptApiUrl: string;

  /** Whether each adapter is enabled (based on env vars being set) */
  discord: boolean;
  telegram: boolean;
  slack: boolean;

  /** Use Redis for state (production) or in-memory (dev) */
  redisUrl?: string;
}

export function loadConfig(): Config {
  return {
    autogptApiUrl: env("AUTOGPT_API_URL", "http://localhost:8006"),
    discord: !!process.env.DISCORD_BOT_TOKEN,
    telegram: !!process.env.TELEGRAM_BOT_TOKEN,
    slack: !!process.env.SLACK_BOT_TOKEN,
    redisUrl: process.env.REDIS_URL,
  };
}

function env(key: string, fallback: string): string {
  return process.env[key] ?? fallback;
}
