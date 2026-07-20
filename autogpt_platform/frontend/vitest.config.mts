import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";

// `@autogpt/icons` is an optional (private) dependency that isn't installed in
// CI. next.config.mjs aliases it to a local stub for the webpack/turbopack
// builds, but vitest doesn't read next.config, so mirror that here. Aliasing
// unconditionally keeps tests deterministic (always the Phosphor fallback)
// regardless of whether the package happens to be installed locally.
const autogptIconsStub = fileURLToPath(
  new URL("./src/components/atoms/Icon/agptIconsStub.ts", import.meta.url),
);

export default defineConfig({
  plugins: [tsconfigPaths(), react()],
  resolve: {
    // The /solid entry must come first: Vite alias keys are prefix matches, so
    // the bare key would otherwise rewrite the subpath to "<stub>.ts/solid".
    alias: {
      "@autogpt/icons/solid": autogptIconsStub,
      "@autogpt/icons": autogptIconsStub,
    },
  },
  test: {
    environment: "happy-dom",
    include: ["src/**/*.test.tsx", "src/**/*.test.ts"],
    setupFiles: ["./src/tests/integrations/vitest.setup.tsx"],
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
  },
});
