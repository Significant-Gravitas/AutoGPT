import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";

/**
 * Config for pure unit tests under src/lib that do not need MSW or generated API.
 * Use: npx vitest run --config vitest.unit.mts
 * Or: npm run test:unit:lib
 */
export default defineConfig({
  plugins: [tsconfigPaths(), react()],
  test: {
    environment: "happy-dom",
    include: ["src/lib/**/*.test.tsx", "src/lib/**/*.test.ts"],
    exclude: ["**/node_modules/**", "**/autogpt-server-api/**"],
  },
});
