import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  resolve: {
    dedupe: ["react", "react-dom"],
  },
  server: {
    proxy: {
      "/api": "http://localhost:8787",
    },
  },
  test: {
    exclude: ["node_modules/**", "server-dist/**"],
  },
});
