import path from "node:path";
import { fileURLToPath } from "node:url";
import { storybookTest } from "@storybook/addon-vitest/vitest-plugin";
import { playwright } from "@vitest/browser-playwright";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";
import tsconfigPaths from "vite-tsconfig-paths";

const dirname = path.dirname(fileURLToPath(import.meta.url));

// Next's Vite plugin, which the storybook project loads, reads .env into
// process.env for the whole run. Unit tests restore this snapshot so they
// keep testing the defaults (src/tests/integrations/restore-process-env.ts).
const processEnvBeforeStorybook = Object.fromEntries(
  Object.entries(process.env).filter(
    (entry): entry is [string, string] => entry[1] !== undefined,
  ),
);

export default defineConfig({
  test: {
    coverage: {
      provider: "v8",
      reporter: ["text", "cobertura"],
      reportsDirectory: "./coverage",
      include: ["src/**/*.{ts,tsx}"],
      exclude: [
        "src/**/*.test.{ts,tsx}",
        "src/**/*.stories.{ts,tsx}",
        "src/playwright/**",
        "src/tests/**",
      ],
    },
    projects: [
      {
        plugins: [tsconfigPaths(), react()],
        test: {
          name: "unit",
          environment: "happy-dom",
          include: [
            "src/**/*.test.tsx",
            "src/**/*.test.ts",
            "scripts/**/*.test.ts",
          ],
          setupFiles: [
            "./src/tests/integrations/restore-process-env.ts",
            "./src/tests/integrations/vitest.setup.tsx",
          ],
          provide: { processEnvBeforeStorybook },
        },
      },
      {
        // Every story in .storybook/main.ts runs as a test in Chromium: it
        // must render, its play function must pass, and axe must find no
        // violations (stories set `a11y: { test: "error" }`).
        plugins: [
          storybookTest({ configDir: path.join(dirname, ".storybook") }),
        ],
        test: {
          name: "storybook",
          browser: {
            enabled: true,
            headless: true,
            provider: playwright(),
            instances: [{ browser: "chromium" }],
          },
        },
      },
    ],
  },
});
