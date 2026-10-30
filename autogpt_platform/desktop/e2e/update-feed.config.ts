import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "update-feed",
  testMatch: "real-feed.spec.ts",
  workers: 1,
  retries: 0,
  timeout: 90 * 60_000,
  expect: { timeout: 30_000 },
  outputDir: "update-feed-results",
  reporter: [["list"], ["html", { outputFolder: "update-feed-report", open: "never" }]],
});
