import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  define: { "process.env": {} },
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
});
