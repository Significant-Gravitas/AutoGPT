import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./src/playwright",
  testMatch: "*.browser.spec.ts",
  forbidOnly: true,
  retries: 0,
  timeout: 60000,
  workers: 1,
  reporter: "list",
  use: {
    ...devices["Desktop Chrome"],
    baseURL: "http://127.0.0.1:4175",
    trace: "retain-on-failure",
  },
  webServer: {
    command:
      "node node_modules/vite/bin/vite.js --config vite.tooltip.config.ts --host 127.0.0.1 --port 4175 --strictPort",
    url: "http://127.0.0.1:4175/src/playwright/tooltip/index.html",
    reuseExistingServer: false,
  },
});
