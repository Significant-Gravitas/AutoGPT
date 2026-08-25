import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: {
    lib: {
      entry: "src/index.tsx",
      formats: ["es"],
      fileName: "AutoGPTEmbeddedChatElement",
      cssFileName: "embedded-chat-element",
    },
    rollupOptions: {
      external: [
        "@ai-sdk/react",
        "ai",
        "react",
        "react/jsx-runtime",
        "react-dom/client",
      ],
    },
  },
  test: {
    environment: "happy-dom",
  },
});
