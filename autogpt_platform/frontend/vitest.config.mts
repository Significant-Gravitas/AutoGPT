import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";

export default defineConfig({
  plugins: [tsconfigPaths(), react()],
  test: {
    environment: "happy-dom",
    include: ["src/**/*.test.tsx"],
    setupFiles: ["./src/tests/integrations/vitest.setup.tsx"],
  },
});
