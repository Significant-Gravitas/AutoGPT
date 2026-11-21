import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "update-feed",
  testMatch: "unit.spec.ts",
  workers: 1,
  retries: 0,
  outputDir: "update-feed-unit-results",
});
