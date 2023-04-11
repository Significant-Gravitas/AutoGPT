import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import express from "./express-plugin"
import path from "path"

export default defineConfig({
  plugins: [react(), express("src/server")],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src/web"),
    },
  },
})
