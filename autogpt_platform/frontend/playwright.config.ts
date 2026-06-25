import { defineConfig, devices } from "@playwright/test";

/**
 * Read environment variables from file.
 * https://github.com/motdotla/dotenv
 */
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { buildCookieConsentStorageState } from "./src/playwright/credentials/storage-state";
dotenv.config({ path: path.resolve(__dirname, ".env") });
dotenv.config({ path: path.resolve(__dirname, "../backend/.env") });

const frontendRoot = __dirname.replaceAll("\\", "/");
const configuredBaseURL =
  process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000";
const parsedBaseURL = new URL(configuredBaseURL);
const baseURL = parsedBaseURL.toString().replace(/\/$/, "");
const baseOrigin = parsedBaseURL.origin;
const jsonReporterOutputFile = process.env.PLAYWRIGHT_JSON_OUTPUT_FILE;
const configuredWorkers = process.env.PLAYWRIGHT_WORKERS
  ? Number(process.env.PLAYWRIGHT_WORKERS)
  : process.env.CI
    ? 8
    : undefined;

// Directory where CI copies .next/static from the Docker container
const staticCoverageDir = path.resolve(__dirname, ".next-static-coverage");

function normalizeCoverageSourcePath(filePath: string) {
  const normalizedFilePath = filePath.replaceAll("\\", "/");
  const withoutWebpackPrefix = normalizedFilePath.replace(
    /^webpack:\/\/_N_E\//,
    "",
  );

  if (withoutWebpackPrefix.startsWith("./")) {
    return withoutWebpackPrefix.slice(2);
  }

  if (withoutWebpackPrefix.startsWith(frontendRoot)) {
    return path.posix.relative(frontendRoot, withoutWebpackPrefix);
  }

  return withoutWebpackPrefix;
}

// Resolve source maps from the copied .next/static directory.
// Cache parsed results to avoid repeated disk reads during report generation.
const sourceMapCache = new Map<string, object | undefined>();

function resolveSourceMap(sourcePath: string) {
  // sourcePath is the sourceMappingURL, e.g.:
  //   "http://localhost:3000/_next/static/chunks/abc123.js.map"
  const match = sourcePath.match(/_next\/static\/(.+)$/);
  if (!match) return undefined;

  const mapFile = path.join(staticCoverageDir, match[1]);
  if (sourceMapCache.has(mapFile)) return sourceMapCache.get(mapFile);

  try {
    const result = JSON.parse(fs.readFileSync(mapFile, "utf8")) as object;
    sourceMapCache.set(mapFile, result);
    return result;
  } catch {
    sourceMapCache.set(mapFile, undefined);
    return undefined;
  }
}

export default defineConfig({
  testDir: "./src/playwright",
  testMatch: /.*-happy-path\.spec\.ts/,
  /* Global setup file that runs before all tests */
  globalSetup: "./src/playwright/global-setup.ts",
  /* Run tests in files in parallel */
  fullyParallel: true,
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? Number(process.env.PLAYWRIGHT_RETRIES ?? 2) : 0,
  /* Higher worker count keeps PR smoke runtime down without sharing page state. */
  workers: configuredWorkers,
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: [
    ["list"],
    ["html", { open: "never" }],
    [
      "monocart-reporter",
      {
        name: "E2E Coverage Report",
        outputFile: "./coverage/e2e/report.html",
        coverage: {
          reports: ["cobertura"],
          outputDir: "./coverage/e2e",
          entryFilter: (entry: { url: string }) =>
            entry.url.includes("/_next/static/") &&
            !entry.url.includes("node_modules"),
          sourceFilter: (sourcePath: string) =>
            sourcePath.includes("src/") && !sourcePath.includes("node_modules"),
          sourcePath: (filePath: string) =>
            normalizeCoverageSourcePath(filePath),
          sourceMapResolver: (sourcePath: string) =>
            resolveSourceMap(sourcePath),
        },
      },
    ],
    ...(jsonReporterOutputFile
      ? [["json", { outputFile: jsonReporterOutputFile }] as const]
      : []),
  ],
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL,

    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    screenshot: "only-on-failure",
    bypassCSP: true,

    /* Helps debugging failures */
    trace: process.env.CI ? "on-first-retry" : "retain-on-failure",
    video: process.env.CI ? "off" : "retain-on-failure",

    /* Auto-accept cookies in all tests to prevent banner interference */
    storageState: buildCookieConsentStorageState(baseOrigin),
  },
  /* Maximum time one test can run for */
  timeout: 25000,

  /* Configure web server to start automatically (local dev only) */
  webServer: {
    command: "pnpm start",
    url: baseURL,
    reuseExistingServer: true,
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], channel: "chromium" },
    },
  ],
});
