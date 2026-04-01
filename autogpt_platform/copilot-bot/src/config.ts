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
  const isProduction = process.env.NODE_ENV === "production";

  if (isProduction) {
    requireEnv("PLATFORM_BOT_API_KEY");

    const hasAdapter =
      !!process.env.DISCORD_BOT_TOKEN ||
      !!process.env.TELEGRAM_BOT_TOKEN ||
      !!process.env.SLACK_BOT_TOKEN;

    if (!hasAdapter) {
      throw new Error(
        "Production requires at least one adapter token: " +
          "DISCORD_BOT_TOKEN, TELEGRAM_BOT_TOKEN, or SLACK_BOT_TOKEN"
      );
    }
  }

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

function requireEnv(key: string): string {
  const value = process.env[key];
  if (!value) {
    throw new Error(`Missing required environment variable: ${key}`);
  }
  return value;
}
