/**
 * Before/after screenshots for design-system PRs (SECRT-3099, DS1).
 *
 * Every DS2 and DS5 PR runs this on its base and on its head, then compares
 * the two sets at a pixel threshold of 0. DS tickets run it; they do not
 * edit it.
 *
 *   # 1. Capture. The app must run from a build with NEXT_PUBLIC_PW_TEST=true
 *   #    (mocked feature flags), e.g.
 *   #    NEXT_PUBLIC_PW_TEST=true pnpm build && pnpm start
 *   #    and Storybook from `pnpm build-storybook` served on a port.
 *   SCREENSHOT_EMAIL=… SCREENSHOT_PASSWORD=… \
 *   npx tsx scripts/screenshot-pages.ts capture --out shots/before \
 *     [--base-url http://localhost:3000] [--routes /login,/library] \
 *     [--storybook-url http://localhost:6006] [--stories tokens-,atoms-button--]
 *
 *   # 2. Compare two captures; writes a diff PNG per screenshot.
 *   npx tsx scripts/screenshot-pages.ts compare shots/before shots/after \
 *     --diff shots/diff
 *
 * Determinism: fixed viewports (1440x900, 390x844) at device scale 1,
 * `prefers-reduced-motion: reduce`, CSS animations disabled, the clock pinned
 * with `page.clock.setFixedTime`, UTC and en-US, and `document.fonts.ready`
 * awaited (Poppins loads from Google Fonts with `display=swap`).
 * `--routes`/`--stories` without a value capture nothing of that kind.
 */
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { chromium, type BrowserContext, type Page } from "@playwright/test";

export const VIEWPORTS = [
  { name: "desktop", width: 1440, height: 900 },
  { name: "mobile", width: 390, height: 844 },
] as const;

const FIXED_TIME = new Date("2026-01-15T12:00:00.000Z");

const PUBLIC_ROUTES = ["/login", "/signup", "/reset-password"];

const SIGNED_IN_ROUTES = [
  "/marketplace",
  "/library",
  "/home",
  "/copilot",
  "/team",
  "/build",
  "/settings/profile",
  "/settings/account",
  "/settings/api-keys",
  "/settings/integrations",
];

const DEFAULT_STORIES = ["tokens-", "atoms-button--", "atoms-text--"];

interface PngImage {
  width: number;
  height: number;
  data: Buffer;
}

interface ImageTools {
  readPng(buffer: Buffer): PngImage;
  writePng(image: PngImage): Buffer;
  pixelmatch(
    a: Buffer,
    b: Buffer,
    out: Buffer,
    width: number,
    height: number,
    options: { threshold: number; includeAA: boolean },
  ): number;
}

// pixelmatch and a PNG codec ship inside playwright-core, which
// @playwright/test already installs, so the harness adds no dependency.
function loadImageTools(): ImageTools {
  const fromTest = createRequire(
    createRequire(join(process.cwd(), "package.json")).resolve(
      "@playwright/test/package.json",
    ),
  );
  const fromPlaywright = createRequire(
    fromTest.resolve("playwright/package.json"),
  );
  const coreDir = dirname(
    fromPlaywright.resolve("playwright-core/package.json"),
  );
  const { PNG } = fromPlaywright(join(coreDir, "lib/utilsBundle.js"));
  const pixelmatchModule = fromPlaywright(
    join(coreDir, "lib/third_party/pixelmatch.js"),
  );
  return {
    readPng: (buffer) => PNG.sync.read(buffer),
    writePng: (image) => PNG.sync.write(image),
    pixelmatch: pixelmatchModule.default ?? pixelmatchModule,
  };
}

/** Pixel diff at threshold 0, anti-aliased pixels included. */
export function comparePngs(before: Buffer, after: Buffer) {
  const tools = loadImageTools();
  const a = tools.readPng(before);
  const b = tools.readPng(after);
  if (a.width !== b.width || a.height !== b.height) {
    return {
      diffPixels: Math.max(a.width * a.height, b.width * b.height),
      sizeMismatch: `${a.width}x${a.height} vs ${b.width}x${b.height}`,
      diff: null,
    };
  }
  const out = {
    width: a.width,
    height: a.height,
    data: Buffer.alloc(a.data.length),
  };
  const diffPixels = tools.pixelmatch(
    a.data,
    b.data,
    out.data,
    a.width,
    a.height,
    {
      threshold: 0,
      includeAA: true,
    },
  );
  return { diffPixels, sizeMismatch: null, diff: tools.writePng(out) };
}

function readFlag(argv: string[], name: string) {
  const index = argv.indexOf(name);
  if (index < 0) return undefined;
  const value = argv[index + 1];
  return value === undefined || value.startsWith("--") ? "" : value;
}

function listFlag(argv: string[], name: string, fallback: string[]) {
  const value = readFlag(argv, name);
  if (value === undefined) return fallback;
  return value.split(",").filter(Boolean);
}

function fileNameFor(kind: string, id: string, viewport: string) {
  const slug = id.replace(/^\//, "").replace(/[^a-zA-Z0-9-]+/g, "_") || "root";
  return `${kind}__${slug}__${viewport}.png`;
}

async function settle(page: Page) {
  await page.waitForLoadState("networkidle").catch(() => undefined);
  await page.evaluate(async () => {
    await document.fonts.ready;
    await Promise.all(
      Array.from(document.images)
        .filter((img) => !img.complete)
        .map(
          (img) =>
            new Promise((done) => {
              img.addEventListener("load", done, { once: true });
              img.addEventListener("error", done, { once: true });
            }),
        ),
    );
  });
  // Two frames so layout driven by font metrics has painted.
  await page.evaluate(
    () =>
      new Promise((done) =>
        requestAnimationFrame(() => requestAnimationFrame(done)),
      ),
  );
}

async function newContext(
  browser: Awaited<ReturnType<typeof chromium.launch>>,
  viewport: (typeof VIEWPORTS)[number],
) {
  return browser.newContext({
    viewport: { width: viewport.width, height: viewport.height },
    deviceScaleFactor: 1,
    reducedMotion: "reduce",
    colorScheme: "light",
    locale: "en-US",
    timezoneId: "UTC",
  });
}

async function signIn(context: BrowserContext, baseUrl: string) {
  const email = process.env.SCREENSHOT_EMAIL;
  const password = process.env.SCREENSHOT_PASSWORD;
  if (!email || !password) return false;
  const response = await context.request.post(
    `${baseUrl}/api/auth/sign-in/email`,
    {
      data: { email, password },
      headers: { "Content-Type": "application/json" },
    },
  );
  if (!response.ok()) {
    throw new Error(
      `sign-in failed: ${response.status()} ${await response.text()}`,
    );
  }
  return true;
}

async function shoot(page: Page, url: string, file: string) {
  await page.clock.setFixedTime(FIXED_TIME);
  await page.goto(url, { waitUntil: "load", timeout: 60_000 });
  await settle(page);
  await page.screenshot({
    path: file,
    fullPage: true,
    animations: "disabled",
    caret: "hide",
    scale: "css",
  });
}

async function storyIds(storybookUrl: string, prefixes: string[]) {
  if (prefixes.length === 0) return [];
  const response = await fetch(`${storybookUrl}/index.json`);
  const index = (await response.json()) as {
    entries: Record<string, { id: string; type: string }>;
  };
  return Object.values(index.entries)
    .filter((entry) => entry.type === "story")
    .map((entry) => entry.id)
    .filter((id) => prefixes.some((prefix) => id.startsWith(prefix)))
    .sort();
}

async function capture(argv: string[]) {
  const outDir = resolve(readFlag(argv, "--out") || "screenshots");
  const baseUrl = readFlag(argv, "--base-url") || "http://localhost:3000";
  const storybookUrl = readFlag(argv, "--storybook-url");
  const signedIn = Boolean(
    process.env.SCREENSHOT_EMAIL && process.env.SCREENSHOT_PASSWORD,
  );
  const routes = listFlag(argv, "--routes", [
    ...PUBLIC_ROUTES,
    ...(signedIn ? SIGNED_IN_ROUTES : []),
  ]);
  const stories = storybookUrl
    ? await storyIds(storybookUrl, listFlag(argv, "--stories", DEFAULT_STORIES))
    : [];
  mkdirSync(outDir, { recursive: true });

  const browser = await chromium.launch();
  const written: string[] = [];
  for (const viewport of VIEWPORTS) {
    const publicContext = await newContext(browser, viewport);
    const appContext = await newContext(browser, viewport);
    if (signedIn) await signIn(appContext, baseUrl);
    for (const route of routes) {
      const context = PUBLIC_ROUTES.includes(route)
        ? publicContext
        : appContext;
      const page = await context.newPage();
      const file = join(outDir, fileNameFor("app", route, viewport.name));
      await shoot(page, `${baseUrl}${route}`, file);
      written.push(file);
      await page.close();
    }
    for (const id of stories) {
      const page = await publicContext.newPage();
      const file = join(outDir, fileNameFor("story", id, viewport.name));
      await shoot(
        page,
        `${storybookUrl}/iframe.html?id=${id}&viewMode=story`,
        file,
      );
      written.push(file);
      await page.close();
    }
    await publicContext.close();
    await appContext.close();
  }
  await browser.close();
  console.log(`${written.length} screenshot(s) written to ${outDir}`);
  return 0;
}

function compare(argv: string[]) {
  const [beforeArg, afterArg] = argv.filter((arg) => !arg.startsWith("--"));
  const diffArg = readFlag(argv, "--diff");
  if (!beforeArg || !afterArg) {
    console.error(
      "usage: npx tsx scripts/screenshot-pages.ts compare <before> <after> [--diff <dir>]",
    );
    return 2;
  }
  const beforeDir = resolve(beforeArg);
  const afterDir = resolve(afterArg);
  const diffDir = diffArg ? resolve(diffArg) : null;
  if (diffDir) mkdirSync(diffDir, { recursive: true });

  const names = new Set(
    [...readdirSync(beforeDir), ...readdirSync(afterDir)].filter((name) =>
      name.endsWith(".png"),
    ),
  );
  const rows = [
    "| screenshot | size | changed pixels |",
    "| --- | --- | --- |",
  ];
  let failing = 0;
  for (const name of [...names].sort()) {
    const beforeFile = join(beforeDir, name);
    const afterFile = join(afterDir, name);
    if (!existsSync(beforeFile) || !existsSync(afterFile)) {
      failing++;
      rows.push(
        `| ${name} | - | only in ${existsSync(beforeFile) ? "before" : "after"} |`,
      );
      continue;
    }
    const result = comparePngs(
      readFileSync(beforeFile),
      readFileSync(afterFile),
    );
    if (result.diffPixels > 0) failing++;
    if (diffDir && result.diff) writeFileSync(join(diffDir, name), result.diff);
    const size = result.sizeMismatch ?? sizeOf(readFileSync(afterFile));
    rows.push(`| ${name} | ${size} | ${result.diffPixels} |`);
  }
  console.log(rows.join("\n"));
  console.log(
    `\n${names.size} screenshot(s) compared, ${failing} with differences`,
  );
  return failing === 0 ? 0 : 1;
}

function sizeOf(png: Buffer) {
  const image = loadImageTools().readPng(png);
  return `${image.width}x${image.height}`;
}

async function main(argv: string[]) {
  const [command, ...rest] = argv;
  if (command === "capture") return capture(rest);
  if (command === "compare") return compare(rest);
  console.error(
    "usage: npx tsx scripts/screenshot-pages.ts capture|compare … (see the header of this file)",
  );
  return 2;
}

if (process.argv[1]?.endsWith("screenshot-pages.ts")) {
  main(process.argv.slice(2)).then(
    (code) => {
      process.exitCode = code;
    },
    (error) => {
      console.error(error);
      process.exitCode = 1;
    },
  );
}
