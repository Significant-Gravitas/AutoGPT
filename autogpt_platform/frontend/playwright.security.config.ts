import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./src/playwright",
  testMatch: /shared-file-security-happy-path\.spec\.ts/,
  fullyParallel: true,
  forbidOnly: true,
  retries: 0,
  workers: 1,
  reporter: "list",
  use: {
    bypassCSP: true,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], channel: "chromium" },
    },
  ],
});
