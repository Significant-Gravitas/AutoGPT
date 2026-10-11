import { defineConfig } from "@playwright/test";
import baseConfig from "./playwright.config";

// Journeys sign up their own users (src/playwright/journeys/journey-user.ts),
// so unlike the smoke suite they skip the seeded-account global setup and run
// against any PLAYWRIGHT_BASE_URL, including a `!deploy` preview.
const baseURL = String(baseConfig.use?.baseURL);
const { hostname } = new URL(baseURL);
const isLocalApp = ["localhost", "127.0.0.1", "[::1]"].includes(hostname);

export default defineConfig({
  ...baseConfig,
  testDir: "./src/playwright/journeys",
  testMatch: /.*-journey\.spec\.ts/,
  globalSetup: undefined,
  timeout: 90_000,
  expect: { timeout: 15_000 },
  webServer: isLocalApp ? baseConfig.webServer : undefined,
});
