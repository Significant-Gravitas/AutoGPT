import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import express from "./express-plugin" //Add this

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), express("src/server")], // Adjust this
})
