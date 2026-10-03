// Renders a generated HTML file in headless Chrome with the tokens and fonts, for quick checks.
import { chromium } from "@playwright/test";
import { readFileSync, writeFileSync } from "node:fs";
import { join, dirname, basename } from "node:path";
import { fileURLToPath } from "node:url";
const here = dirname(fileURLToPath(import.meta.url));
const tokens = JSON.parse(readFileSync(join(here, "..", "tokens.json"), "utf8"));
const vars = tokens.map((t) => `${t.name}: ${t.value};`).join("\n");
const files = process.argv.slice(2);
const browser = await chromium.launch({ executablePath: "/usr/bin/google-chrome", args: ["--no-sandbox"] });
for (const f of files) {
  const html = readFileSync(f, "utf8");
  const m = html.match(/width:(\d+)px;height:(\d+)px/); const w = Number(m?.[1] || 1440), h = Number(m?.[2] || 900);
  const page = await browser.newPage({ viewport: { width: w, height: h } });
  await page.setContent(`<!doctype html><html><head><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Geist:wght@300..900&family=Geist+Mono:wght@300..900&display=swap"><style>:root{${vars}} body{margin:0;background:#F9F9F9}</style></head><body>${html}</body></html>`, { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready); await page.waitForTimeout(500);
  const out = f.replace(/\.html$/, ".preview.png"); await page.screenshot({ path: out, fullPage: true }); console.log("wrote", out);
  await page.close();
}
await browser.close();
