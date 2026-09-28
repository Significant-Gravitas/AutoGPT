// Captures the OPEN state of overlay components (dialogs, menus, popovers,
// toasts, tooltips, selects) by triggering them in the story, then serialising
// the whole document so portals are included.
import { chromium } from "@playwright/test";
import { spawn } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { buildColorMap, serializerFor, inlineLocalImages } from "./serialize-lib.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const paperDir = join(here, "..");
const root = join(paperDir, "..", "..");
const staticDir = join(root, "storybook-static");
const PORT = 6008;
const OVERLAY = /Dialog|DropdownMenu|Popover|SecondaryMenu|Toast|Tooltip|Select|TimePicker|SearchCommandModal|ExpertAvatarPicker|InstallWorkflowPicker|Collapsible|Accordion|ShowMore/;
const VIEW = { width: 1280, height: 900 };

async function main() {
  const server = spawn("python3", ["-m", "http.server", String(PORT), "--directory", staticDir], { stdio: "ignore" });
  await new Promise((r) => setTimeout(r, 1500));
  const index = JSON.parse(readFileSync(join(staticDir, "index.json"), "utf8"));
  const stories = Object.values(index.entries).filter((e) => e.type === "story" && OVERLAY.test(e.title) && !/copilot/i.test(e.title));
  console.log(`open-state candidates: ${stories.length}`);
  const colorByHex = buildColorMap(join(paperDir, "tokens.json"));
  const script = serializerFor("body", colorByHex);
  const browser = await chromium.launch({ executablePath: "/usr/bin/google-chrome", args: ["--no-sandbox", "--font-render-hinting=none"] });
  const ctx = await browser.newContext({ viewport: VIEW, deviceScaleFactor: 1, reducedMotion: "reduce" });
  const page = await ctx.newPage();
  const manifestPath = join(paperDir, "kit", "manifest.json");
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  let ok = 0;
  for (const s of stories) {
    const id = `${s.id}--open`;
    try {
      await page.goto(`http://127.0.0.1:${PORT}/iframe.html?id=${s.id}&viewMode=story`, { waitUntil: "load", timeout: 45000 });
      await page.waitForFunction(() => document.querySelector("#storybook-root")?.children.length > 0, null, { timeout: 20000 });
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(600);
      const before = await page.evaluate(() => document.body.innerHTML.length);
      const trigger = page.locator("#storybook-root button, #storybook-root [role=combobox], #storybook-root [role=button], #storybook-root input").first();
      if (!(await trigger.count())) { console.log(`skip ${s.id}: no trigger`); continue; }
      if (/Tooltip/.test(s.title)) await trigger.hover(); else await trigger.click({ timeout: 5000 });
      await page.waitForTimeout(900);
      const after = await page.evaluate(() => document.body.innerHTML.length);
      if (after - before < 200) { console.log(`skip ${s.id}: nothing opened`); continue; }
      const res = await page.evaluate(script);
      const html = await inlineLocalImages(res.html);
      const pageName = s.title.startsWith("Atoms") ? "01 Atoms" : s.title.startsWith("Molecules") ? "02 Molecules" : "02b Organisms";
      const dir = join(paperDir, "kit", pageName); mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, `${id}.html`), html);
      await page.screenshot({ path: join(dir, `${id}.png`) });
      const component = s.title.split("/").slice(-1)[0].trim();
      const entry = { id, title: s.title, name: `${s.name} (open)`, page: pageName, artboardName: `Kit/${component}/${s.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}/open`, file: `${pageName}/${id}.html`, png: `${pageName}/${id}.png`, width: VIEW.width, height: VIEW.height, bytes: html.length, exact: true };
      manifest.stories = [...manifest.stories.filter((x) => x.id !== id), entry];
      writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
      ok++; console.log(`ok  ${id} ${html.length}b`);
    } catch (e) { console.log(`FAIL ${id}: ${String(e).slice(0, 120)}`); }
  }
  await browser.close(); server.kill();
  console.log(`OPEN_STATES_DONE ${ok}`);
}
main().catch((e) => { console.error(e); process.exit(1); });
