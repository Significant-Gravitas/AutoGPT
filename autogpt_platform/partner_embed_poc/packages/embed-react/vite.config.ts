import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: {
    lib: {
      entry: "src/AutoGPTEmbeddedChat.tsx",
      formats: ["es"],
      fileName: "AutoGPTEmbeddedChat",
    },
    rollupOptions: {
      external: ["react", "react/jsx-runtime", "@ai-sdk/react", "ai"],
      output: {
        assetFileNames: "embedded-chat.css",
      },
    },
  },
  test: {
    environment: "happy-dom",
  },
});
