import { defineConfig, devices } from "@playwright/test";

/**
 * Read environment variables from file.
 * https://github.com/motdotla/dotenv
 */
import dotenv from "dotenv";
import path from "path";
dotenv.config({ path: path.resolve(__dirname, ".env") });
dotenv.config({ path: path.resolve(__dirname, "../backend/.env") });
/**
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
  testDir: "./src/tests",
  /* Run tests in files in parallel */
  fullyParallel: true,
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  /* Use more workers in CI for speed, but limit to avoid resource issues */
  workers: process.env.CI ? 2 : undefined,
  /* Reporter to use. See https://playwright.dev/docs/test-re porters */
  reporter: [["html"], ["line"]],
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: "http://localhost:3000/",

    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    bypassCSP: true,
  },
  /* Maximum time one test can run for */
  timeout: 60000,

  /* Configure projects for major browsers */
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },

    // {
    //   name: "firefox",
    //   use: { ...devices["Desktop Firefox"] },
    // },

    {
      name: "webkit",
      use: { ...devices["Desktop Safari"] },
    },

    // /* Test against mobile viewports. */
    // // {
    // //   name: 'Mobile Chrome',
    // //   use: { ...devices['Pixel 5'] },
    // // },
    // // {
    // //   name: 'Mobile Safari',
    // //   use: { ...devices['iPhone 12'] },
    // // },

    // /* Test against branded browsers. */
    // {
    //   name: "Microsoft Edge",
    //   use: { ...devices["Desktop Edge"], channel: "msedge" },
    // },
    // {
    //   name: 'Google Chrome',
    //   use: { ...devices['Desktop Chrome'], channel: 'chrome' },
    // },
  ],

  /* Run your local server before starting the tests */
  webServer: {
    command: "next start -p 3000",
    url: "http://localhost:3000/",
    reuseExistingServer: !process.env.CI, // speeds up local runs
    timeout: 60 * 1000, // increased timeout for production builds
    env: {
      NODE_ENV: "production", // use production mode for stability
      NEXT_PUBLIC_PW_TEST: "true",
    },
  },
});
