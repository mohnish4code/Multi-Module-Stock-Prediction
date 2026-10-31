import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development the UI runs on :5173 and forwards API calls to the FastAPI server.
const api = process.env.API_URL ?? "http://localhost:8000";
const proxied = ["/predict", "/backtest", "/summary", "/tickers", "/quotes", "/health"];

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(proxied.map((path) => [path, { target: api, changeOrigin: true }])),
  },
  build: { outDir: "dist", chunkSizeWarningLimit: 800 },
  test: { environment: "node" },
});
