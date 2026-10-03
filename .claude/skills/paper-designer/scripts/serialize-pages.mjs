// Captures the live app (flags forced on) page by page with headless Chrome and
// serialises each page with inlined computed styles for Paper's write_html.
//
//   node serialize-pages.mjs --base http://localhost:3000 --api http://localhost:8006 \
//        --email <u> --password <p> [--only <substring>] [--widths 1440,390]
//
// Output: design/paper/kit/03 Shells/<slug>-<width>.html (+ .chunks.json, .png)
// and entries merged into design/paper/kit/manifest.json (page "03 Shells").
import { chromium } from "@playwright/test";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { buildColorMap, serializerFor, inlineLocalImages } from "./serialize-lib.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const paperDir = join(here, "..");
const outDir = join(paperDir, "kit", "03 Shells");
const args = process.argv.slice(2);
const opt = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const BASE = opt("--base", "http://localhost:3000");
const API = opt("--api", "http://localhost:8006");
const EMAIL = opt("--email"); const PASSWORD = opt("--password");
const only = opt("--only", null);
const widths = opt("--widths", "1440,390").split(",").map(Number);

const STATIC_ROUTES = [
  ["home", "/home"], ["copilot", "/copilot"], ["library", "/library"], ["library-followups", "/library/followups"],
  ["library-skills", "/library/skills"], ["marketplace", "/marketplace"], ["marketplace-search", "/marketplace/search?searchTerm=email"],
  ["marketplace-skills", "/marketplace/skills"], ["build", "/build"], ["artifacts", "/artifacts"], ["team", "/team"],
  ["team-autopilot", "/team/autopilot"], ["settings", "/settings"], ["settings-account", "/settings/account"],
  ["settings-profile", "/settings/profile"], ["settings-api-keys", "/settings/api-keys"], ["settings-billing", "/settings/billing"],
  ["settings-bots", "/settings/bots"], ["settings-creator-dashboard", "/settings/creator-dashboard"],
  ["settings-integrations", "/settings/integrations"], ["settings-memory", "/settings/memory"], ["settings-oauth-apps", "/settings/oauth-apps"],
  ["profile", "/profile"], ["profile-dashboard", "/profile/dashboard"], ["profile-credits", "/profile/credits"],
  ["profile-integrations", "/profile/integrations"], ["profile-api-keys", "/profile/api-keys"], ["profile-oauth-apps", "/profile/oauth-apps"],
  ["profile-settings", "/profile/settings"], ["raise", "/raise"], ["health", "/health"], ["error", "/error"], ["unauthorized", "/unauthorized"],
  ["admin-dashboard", "/admin/dashboard"], ["admin-users", "/admin/users"], ["admin-marketplace", "/admin/marketplace"], ["admin-spending", "/admin/spending"],
  ["admin-settings", "/admin/settings"], ["admin-bots", "/admin/bots"], ["admin-memory", "/admin/memory"], ["admin-diagnostics", "/admin/diagnostics"],
  ["admin-execution-analytics", "/admin/execution-analytics"], ["admin-impersonation", "/admin/impersonation"], ["admin-platform-costs", "/admin/platform-costs"],
  ["admin-rate-limits", "/admin/rate-limits"], ["admin-test-data", "/admin/test-data"], ["admin-block-cost-estimates", "/admin/block-cost-estimates"],
  ["copilot-styleguide", "/copilot/styleguide"], ["dev-brain-dump-debug", "/dev/brain-dump-debug"],
  ["auth-integrations-setup-wizard", "/auth/integrations/setup-wizard"], ["auth-code-error", "/auth/auth-code-error"],
  ["tour", "/tour"], ["tour-chat", "/tour/chat"],
];
const PUBLIC_ROUTES = [["login", "/login"], ["signup", "/signup"], ["onboarding", "/onboarding"], ["reset-password", "/reset-password"], ["logout-page", "/logout"]];

async function apiGet(path, token) {
  try { const r = await fetch(API + path, { headers: { Authorization: `Bearer ${token}` } }); if (!r.ok) return null; return await r.json(); } catch { return null; }
}

async function dynamicRoutes(token) {
  const out = [];
  const lib = await apiGet("/api/library/agents?page=1&page_size=5", token);
  const agent = lib?.agents?.[0]; if (agent?.id) out.push(["library-agent", `/library/agents/${agent.id}`]);
  const store = await apiGet("/api/store/agents?page=1&page_size=5", token);
  const sa = store?.agents?.[0]; if (sa?.creator && sa?.slug) { out.push(["marketplace-agent", `/marketplace/agent/${sa.creator}/${sa.slug}`]); out.push(["marketplace-creator", `/marketplace/creator/${sa.creator}`]); }
  const skills = await apiGet("/api/store/skills?page=1&page_size=5", token);
  const sk = skills?.skills?.[0] || skills?.listings?.[0] || (Array.isArray(skills) ? skills[0] : null); if (sk?.slug) out.push(["marketplace-skill", `/marketplace/skills/${sk.slug}`]);
  const experts = await apiGet("/api/experts", token);
  const ex = Array.isArray(experts) ? experts[0] : experts?.experts?.[0]; if (ex?.id) { out.push(["team-expert", `/team/${ex.id}`]); out.push(["marketplace-expert", `/marketplace/experts/${ex.id}`]); }
  return out;
}

async function main() {
  mkdirSync(outDir, { recursive: true });
  const colorByHex = buildColorMap(join(paperDir, "tokens.json"));
  const script = serializerFor("body", colorByHex);
  const browser = await chromium.launch({ executablePath: "/usr/bin/google-chrome", args: ["--no-sandbox", "--font-render-hinting=none"] });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1, reducedMotion: "reduce" });
  let token = "";
  if (EMAIL && PASSWORD) {
    const r = await ctx.request.post(`${BASE}/api/auth/sign-in/email`, { data: { email: EMAIL, password: PASSWORD } });
    console.log("sign-in", r.status());
    const t = await ctx.request.get(`${BASE}/api/auth/token`); try { token = (await t.json()).token || ""; } catch {}
  }
  let routes = [...STATIC_ROUTES, ...(token ? await dynamicRoutes(token) : []), ...PUBLIC_ROUTES];
  if (only) routes = routes.filter(([n, p]) => n.includes(only) || p.includes(only));
  console.log(`capturing ${routes.length} routes x ${widths.length} widths`);
  const manifestPath = join(paperDir, "kit", "manifest.json");
  const manifest = existsSync(manifestPath) ? JSON.parse(readFileSync(manifestPath, "utf8")) : { stories: [], failures: [] };
  const page = await ctx.newPage();
  for (const [name, path] of routes) {
    for (const width of widths) {
      const id = `${name}-${width}`;
      try {
        await page.setViewportSize({ width, height: width < 600 ? 844 : 900 });
        await page.goto(BASE + path, { waitUntil: "domcontentloaded", timeout: 60000 });
        await page.waitForLoadState("networkidle", { timeout: 6000 }).catch(() => {});
        await page.evaluate(() => document.fonts.ready);
        await page.waitForTimeout(1800);
        for (const label of ["Accept All", "Accept all", "Got it", "Skip"]) { const b = page.getByRole("button", { name: label }).first(); if (await b.isVisible().catch(() => false)) { await b.click().catch(() => {}); await page.waitForTimeout(400); } }
        const finalUrl = page.url();
        const res = await page.evaluate(script);
        let html = await inlineLocalImages(res.html);
        const png = join(outDir, `${id}.png`);
        await page.screenshot({ path: png, fullPage: true });
        const file = join(outDir, `${id}.html`);
        writeFileSync(file, html);
        const entry = { id, title: `Shells/${name}`, name: String(width), page: "03 Shells", artboardName: `Shell/${name}/${width}`, file: `03 Shells/${id}.html`, png: `03 Shells/${id}.png`, width, height: Math.max(res.height, width < 600 ? 844 : 900), bytes: html.length, exact: true, route: path, finalUrl };
        manifest.stories = [...manifest.stories.filter((s) => s.id !== id), entry];
        writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
        console.log(`ok  ${id} ${res.width}x${res.height} ${html.length}b ${finalUrl.replace(BASE, "")}`);
      } catch (e) {
        console.log(`FAIL ${id}: ${String(e).slice(0, 160)}`);
        manifest.failures = [...(manifest.failures || []), { id, error: String(e).slice(0, 200) }];
      }
    }
  }
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  await browser.close();
}
main().catch((e) => { console.error(e); process.exit(1); });
