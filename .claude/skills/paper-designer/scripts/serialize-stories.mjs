// Renders every Storybook story with headless Chrome and serialises the DOM with
// inlined computed styles, so the real components can be pasted into Paper via
// write_html. This is "Paper Snapshot" done headlessly and reproducibly.
//
//   pnpm build-storybook
//   node design/paper/scripts/serialize-stories.mjs [--only <substring>] [--limit N]
//
// Output: design/paper/kit/<page>/<story-id>.html + .png, and design/paper/kit/manifest.json
import { chromium } from "@playwright/test";
import { spawn } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const staticDir = join(root, "storybook-static");
const outDir = join(root, "design", "paper", "kit");
const PORT = 6007;
const args = process.argv.slice(2);
const only = args.includes("--only") ? args[args.indexOf("--only") + 1] : null;
const limit = args.includes("--limit") ? Number(args[args.indexOf("--limit") + 1]) : Infinity;

import { buildColorMap, serializerFor, inlineLocalImages } from "./serialize-lib.mjs";
const colorByHex = buildColorMap(join(root, "design/paper/tokens.json"));

function pageFor(title) {
  if (title.startsWith("Tokens")) return "00 Tokens";
  if (title.startsWith("Atoms")) return "01 Atoms";
  if (title.startsWith("Molecules")) return "02 Molecules";
  return "02b Organisms";
}

async function main() {
  if (!existsSync(join(staticDir, "index.json"))) throw new Error("storybook-static/index.json missing; run pnpm build-storybook");
  const server = spawn("python3", ["-m", "http.server", String(PORT), "--directory", staticDir], { stdio: "ignore" });
  await new Promise((r) => setTimeout(r, 1500));
  const index = JSON.parse(readFileSync(join(staticDir, "index.json"), "utf8"));
  let stories = Object.values(index.entries).filter((e) => e.type === "story");
  if (only) stories = stories.filter((s) => s.id.includes(only) || s.title.includes(only));
  stories = stories.slice(0, limit);
  console.log(`serialising ${stories.length} stories`);
  const browser = await chromium.launch({ executablePath: "/usr/bin/google-chrome", args: ["--no-sandbox", "--font-render-hinting=none"] });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1, reducedMotion: "reduce" });
  const page = await ctx.newPage();
  const manifest = [];
  const failures = [];
  const script = serializerFor("#storybook-root", colorByHex);
  for (const s of stories) {
    const pageName = pageFor(s.title);
    const dir = join(outDir, pageName);
    mkdirSync(dir, { recursive: true });
    try {
      await page.goto(`http://127.0.0.1:${PORT}/iframe.html?id=${s.id}&viewMode=story`, { waitUntil: "load", timeout: 45000 });
      await page.waitForFunction(() => document.querySelector("#storybook-root")?.children.length > 0, null, { timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(900);
      const res = await page.evaluate(script);
      const html = await inlineLocalImages(res.html);
      const file = join(dir, `${s.id}.html`);
      writeFileSync(file, html);
      const png = join(dir, `${s.id}.png`);
      const vp = page.viewportSize();
      const clip = res.clip; if (clip.x + clip.width > vp.width || clip.y + clip.height > vp.height) await page.setViewportSize({ width: Math.max(vp.width, Math.ceil(clip.x + clip.width)), height: Math.max(vp.height, Math.ceil(clip.y + clip.height)) });
      await page.screenshot({ path: png, clip: { x: clip.x, y: clip.y, width: Math.min(clip.width, 4000), height: Math.min(clip.height, 8000) } });
      if (page.viewportSize().width !== 1280 || page.viewportSize().height !== 900) await page.setViewportSize({ width: 1280, height: 900 });
      const component = s.title.split("/").slice(-1)[0].trim();
      const artboardName = `Kit/${component}/${s.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
      manifest.push({ id: s.id, title: s.title, name: s.name, page: pageName, artboardName, file: `${pageName}/${s.id}.html`, png: `${pageName}/${s.id}.png`, width: res.width, height: res.height, bytes: html.length });
      console.log(`ok  ${s.id} ${res.width}x${res.height} ${html.length}b`);
    } catch (e) {
      failures.push({ id: s.id, error: String(e).slice(0, 200) });
      console.log(`FAIL ${s.id}: ${String(e).slice(0, 120)}`);
    }
  }
  await browser.close();
  server.kill();
  const prev = existsSync(join(outDir, "manifest.json")) ? JSON.parse(readFileSync(join(outDir, "manifest.json"), "utf8")) : { stories: [] };
  const merged = only || limit !== Infinity ? [...prev.stories.filter((p) => !manifest.some((m) => m.id === p.id)), ...manifest] : manifest;
  writeFileSync(join(outDir, "manifest.json"), JSON.stringify({ generatedAt: new Date().toISOString(), stories: merged, failures }, null, 2));
  console.log(`done: ${manifest.length} ok, ${failures.length} failed`);
}
main().catch((e) => { console.error(e); process.exit(1); });
