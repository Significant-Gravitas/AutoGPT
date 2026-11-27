/**
 * Is every pixel attached to a token? Loads each page or story, screenshots
 * it, switches it to the canary look (src/app/tokens/themes/canary.css: every
 * colour loud, every corner square, every font wrong), screenshots it again,
 * and reports each visible element whose computed colour, radius or font is
 * not a canary value: that value comes from somewhere other than a token.
 *
 *   # Storybook, from `pnpm build-storybook` served on a port:
 *   npx tsx scripts/ds/canary-check.ts --storybook-url http://localhost:6006 \
 *     [--stories atoms-,molecules-] [--skip atoms-progress--live] \
 *     [--out /tmp/canary] [--concurrency 4]
 *
 *   # The app (a running `pnpm dev` or `pnpm start`):
 *   npx tsx scripts/ds/canary-check.ts --base-url http://localhost:3000 \
 *     [--routes /login,/signup]
 *
 * --stories takes story id prefixes; without it every story is checked.
 * --routes defaults to the public routes, which are checked signed out. Routes
 * behind sign-in need SCREENSHOT_EMAIL and SCREENSHOT_PASSWORD, as for
 * scripts/screenshot-pages.ts; without them the run stops before it starts.
 * --concurrency is a whole number, 1 or more. One 1440x900 viewport.
 *
 * Exits 1 when any target failed to render (the reports are still written)
 * and 2 when the flags or credentials are wrong.
 *
 * Writes <out>/screenshots/<target>__{default,canary}.png, <out>/report.json
 * and <out>/report.txt. <out> defaults to a folder in the OS temp directory.
 *
 * Not checked: images, video, canvas, iframes and embeds; the artwork
 * ds-colour-exempt.json lists, where it renders with a data-slot or class
 * (EXEMPT_ARTWORK below); the TanStack Query devtools; Storybook's own markup. Third-party markup
 * (THIRD_PARTY below) is checked and labelled with its library.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, type Browser, type Page } from "@playwright/test";
import {
  DETERMINISTIC_CHROMIUM_ARGS,
  FIXED_TIME,
  PUBLIC_ROUTES,
  dismissPrompts,
  fileNameFor,
  listFlag,
  newContext,
  readFlag,
  settle,
  signIn,
  storyIds,
  VIEWPORTS,
  DEFAULT_DISMISS,
  DEFAULT_SKIP,
} from "../screenshot-pages";
import { exemptEntries } from "./colour-exempt";
import {
  buildReport,
  formatReport,
  leaksOf,
  readCanary,
  type ElementFacts,
  type Leak,
} from "./canary-report";

const FRONTEND = fileURLToPath(new URL("../..", import.meta.url));
const CANARY_CSS = readFileSync(
  join(FRONTEND, "src/app/tokens/themes/canary.css"),
  "utf8",
);

// ds-colour-exempt.json entries whose artwork renders with a marker the page
// script can find. Each key must still be an entry in that file.
const EXEMPT_ARTWORK: Record<string, string> = {
  "src/components/atoms/AutoGPTLogo/AutoGPTLogo.tsx":
    '[data-slot="autogpt-logo"]',
  "src/components/molecules/GlassOrb/{GlassOrb,GlassSurface}.tsx":
    '[data-slot^="glass-orb"], [data-slot^="glass-surface"]',
  "src/components/molecules/AutopilotAvatar/helpers.ts":
    '[data-slot="autopilot-avatar"]',
  "src/components/atoms/Avatar/Avatar.tsx":
    '[data-slot="avatar-fallback"] > svg',
  "src/app/(platform)/copilot/tools/RenderUI/components/ChartCanvas.tsx":
    ".recharts-wrapper",
  "src/app/(platform)/team/[expertId]/components/ExpertOverview/components/ActivityChart.tsx":
    ".recharts-wrapper",
};

// Elements never checked: raster and embedded content, the TanStack Query
// devtools button (dev builds only) and Storybook's own UI.
const SKIPPED =
  "img, picture, video, canvas, iframe, object, embed, .tsqd-parent-container, .sb-wrapper, .sb-errordisplay, .sb-preparing-story, .sb-preparing-docs, .sb-nopreview, #storybook-docs";

// Markup a dependency renders, labelled so its leaks read apart from ours.
const THIRD_PARTY: [string, string][] = [
  ["xyflow", ".react-flow"],
  ["recharts", ".recharts-wrapper"],
  [
    "shiki",
    ".shiki, pre [style*='color'], [style*='background-color']:has(> pre)",
  ],
  ["highlight.js", ".hljs"],
  ["katex", ".katex"],
  ["mermaid", "[data-streamdown='mermaid'], svg[id^='mermaid']"],
  ["streamdown", "[data-streamdown]"],
  ["sonner", "[data-sonner-toaster]"],
];

// Transitions would catch the colours half way from the default look.
const FREEZE_CSS =
  "*, *::before, *::after { transition: none !important; animation-duration: 0s !important; animation-delay: 0s !important; animation-iteration-count: 1 !important; caret-color: transparent !important; }";

interface CollectConfig {
  skipped: string;
  exempt: string;
  thirdParty: [string, string][];
}

// Runs in the page, so it uses nothing from this module.
function collectFacts(config: CollectConfig): ElementFacts[] {
  const facts: ElementFacts[] = [];
  const SVG_SHAPES = new Set([
    "path",
    "circle",
    "rect",
    "line",
    "polyline",
    "polygon",
    "ellipse",
    "text",
    "tspan",
    "use",
  ]);
  const BORDERS = ["top", "right", "bottom", "left"];
  const exempt = config.exempt ? config.exempt : null;

  function describe(el: Element, pseudo = "") {
    const slot = el.getAttribute("data-slot");
    const classes = (el.getAttribute("class") ?? "")
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 5)
      .join(".");
    const classText =
      classes.length > 70 ? `${classes.slice(0, 70)}…` : classes;
    return `${el.tagName.toLowerCase()}${slot ? `[data-slot=${slot}]` : ""}${classText ? `.${classText}` : ""}${pseudo}`;
  }

  function hasOwnText(el: Element) {
    if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement)
      return el.value !== "" || el.placeholder !== "";
    return [...el.childNodes].some(
      (node) => node.nodeType === Node.TEXT_NODE && node.textContent?.trim(),
    );
  }

  function colours(el: Element, style: CSSStyleDeclaration, text: boolean) {
    const found: { property: string; value: string }[] = [];
    const add = (property: string, value: string) => {
      if (value && value !== "none" && value !== "transparent")
        found.push({ property, value });
    };
    add("background-color", style.backgroundColor);
    for (const side of BORDERS) {
      const width = parseFloat(style.getPropertyValue(`border-${side}-width`));
      const kind = style.getPropertyValue(`border-${side}-style`);
      if (width > 0 && kind !== "none" && kind !== "hidden")
        add(
          `border-${side}-color`,
          style.getPropertyValue(`border-${side}-color`),
        );
    }
    // `auto` is the browser's focus ring, drawn only on :focus-visible.
    if (style.outlineStyle === "auto") {
      if (el.matches(":focus-visible"))
        add("outline-color (browser focus ring)", style.outlineColor);
    } else if (
      style.outlineStyle !== "none" &&
      parseFloat(style.outlineWidth) > 0
    )
      add("outline-color", style.outlineColor);
    if (text) add("color", style.color);
    if (el instanceof SVGElement && SVG_SHAPES.has(el.tagName.toLowerCase())) {
      // An icon drawn in currentColor shows the colour of its text.
      for (const property of ["fill", "stroke"] as const) {
        const value = style.getPropertyValue(property);
        if (!value || value === "none" || value.startsWith("url(")) continue;
        if (property === "stroke" && parseFloat(style.strokeWidth) === 0)
          continue;
        add(value === style.color ? "color" : property, value);
      }
    }
    add("box-shadow", style.boxShadow);
    if (style.backgroundImage.includes("gradient"))
      add("background-image", style.backgroundImage);
    add("text-shadow", style.textShadow);
    if (style.filter.includes("drop-shadow")) add("filter", style.filter);
    return found;
  }

  function radiusOf(style: CSSStyleDeclaration, width: number, height: number) {
    return {
      corners: [
        style.borderTopLeftRadius,
        style.borderTopRightRadius,
        style.borderBottomRightRadius,
        style.borderBottomLeftRadius,
      ],
      width,
      height,
    };
  }

  // A pseudo-element has no rect of its own, so its border box comes from its
  // computed size. An inline one reports `auto`: 0, a size no circle has.
  function pseudoSize(
    style: CSSStyleDeclaration,
    dimension: "width" | "height",
    sides: string[],
  ) {
    const size = parseFloat(style.getPropertyValue(dimension));
    if (Number.isNaN(size)) return 0;
    if (style.boxSizing === "border-box") return size;
    return sides.reduce(
      (total, side) =>
        total +
        (parseFloat(style.getPropertyValue(`padding-${side}`)) || 0) +
        (parseFloat(style.getPropertyValue(`border-${side}-width`)) || 0),
      size,
    );
  }

  function within(el: Element) {
    return (
      el.parentElement?.closest("[data-slot]")?.getAttribute("data-slot") ??
      null
    );
  }

  function origin(el: Element) {
    for (const [name, selector] of config.thirdParty)
      if (el.closest(selector)) return name;
    return "ours";
  }

  for (const el of document.querySelectorAll("body, body *")) {
    if (el.closest(config.skipped)) continue;
    if (exempt && el.closest(exempt)) continue;
    if (!el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true }))
      continue;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) continue;
    const style = getComputedStyle(el);
    // Visually hidden (sr-only, Radix's native <select>): never painted.
    if (
      (rect.width <= 1 && rect.height <= 1) ||
      style.clip === "rect(0px, 0px, 0px, 0px)"
    )
      continue;
    const text = hasOwnText(el);
    const base = { within: within(el), origin: origin(el) };
    // The layout box, not the rect: a transform scales the rect but not the
    // radius, so a scaled rounded square would measure as a circle.
    const box =
      el instanceof HTMLElement
        ? { width: el.offsetWidth, height: el.offsetHeight }
        : rect;
    facts.push({
      ...base,
      element: describe(el),
      colours: colours(el, style, text),
      radius:
        el instanceof SVGElement && !(el instanceof SVGSVGElement)
          ? null
          : radiusOf(style, box.width, box.height),
      font: text ? style.fontFamily : null,
    });
    for (const pseudo of ["::before", "::after"]) {
      const pseudoStyle = getComputedStyle(el, pseudo);
      if (
        pseudoStyle.content === "none" ||
        pseudoStyle.display === "none" ||
        pseudoStyle.visibility === "hidden" ||
        parseFloat(pseudoStyle.opacity) === 0
      )
        continue;
      // `content: ""` draws a box and `content: url(…)` an image, not text.
      const pseudoText =
        pseudoStyle.content !== '""' && !pseudoStyle.content.startsWith("url(");
      facts.push({
        ...base,
        element: describe(el, pseudo),
        colours: colours(el, pseudoStyle, pseudoText),
        radius: radiusOf(
          pseudoStyle,
          pseudoSize(pseudoStyle, "width", ["left", "right"]),
          pseudoSize(pseudoStyle, "height", ["top", "bottom"]),
        ),
        font: pseudoText ? pseudoStyle.fontFamily : null,
      });
    }
    if (
      (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) &&
      el.placeholder &&
      el.value === ""
    ) {
      const placeholder = getComputedStyle(el, "::placeholder");
      facts.push({
        ...base,
        element: describe(el, "::placeholder"),
        colours: [{ property: "color", value: placeholder.color }],
        radius: null,
        font: placeholder.fontFamily,
      });
    }
  }
  return facts;
}

function exemptSelector() {
  const listed = new Set(exemptEntries().map((entry) => entry.files));
  const missing = Object.keys(EXEMPT_ARTWORK).filter(
    (files) => !listed.has(files),
  );
  if (missing.length > 0)
    throw new Error(
      `EXEMPT_ARTWORK names entries ds-colour-exempt.json no longer has: ${missing.join(", ")}`,
    );
  return [...new Set(Object.values(EXEMPT_ARTWORK))].join(", ");
}

interface Target {
  id: string;
  url: string;
  kind: "app" | "story";
}

interface TargetResult {
  id: string;
  url: string;
  screenshots: string[];
  elements: number;
  leaks: Leak[];
  error?: string;
}

// The classes Storybook puts on <body> when a story threw or does not exist,
// and where each one's message is. That markup is skipped, so without this
// the story would count as checked.
const STORY_FAILED: Record<string, string> = {
  "sb-show-errordisplay": "#error-message",
  "sb-show-nopreview": ".sb-nopreview_heading",
};

async function assertStoryRendered(page: Page, id: string) {
  const state = await page.evaluate(
    (failed) => failed.find((name) => document.body.classList.contains(name)),
    Object.keys(STORY_FAILED),
  );
  if (!state) return;
  const message = await page
    .locator(STORY_FAILED[state])
    .first()
    .innerText({ timeout: 1_000 })
    .catch(() => "");
  throw new Error(
    `story ${id} did not render (${state})${message ? `: ${message.trim().split("\n")[0]}` : ""}`,
  );
}

async function checkTarget(
  page: Page,
  target: Target,
  shotsDir: string,
  dismiss: string[],
): Promise<TargetResult> {
  const shot = (look: string) =>
    join(shotsDir, fileNameFor(target.kind, target.id, look));
  assertLoaded(
    await page.goto(target.url, { waitUntil: "load", timeout: 60_000 }),
    target.url,
  );
  await settle(page);
  if (target.kind === "story") await assertStoryRendered(page, target.id);
  if (target.kind === "app" && (await dismissPrompts(page, dismiss))) {
    assertLoaded(
      await page.reload({ waitUntil: "load", timeout: 60_000 }),
      target.url,
    );
    await settle(page);
  }
  await page.addStyleTag({ content: FREEZE_CSS });
  const screenshots = [shot("default"), shot("canary")];
  await page.screenshot({
    path: screenshots[0],
    fullPage: true,
    animations: "disabled",
  });
  await page.addStyleTag({ content: CANARY_CSS });
  await page.evaluate(() => {
    document.documentElement.dataset.look = "canary";
  });
  await settle(page);
  await page.screenshot({
    path: screenshots[1],
    fullPage: true,
    animations: "disabled",
  });
  const facts = await page.evaluate(collectFacts, {
    skipped: SKIPPED,
    exempt: exemptSelector(),
    thirdParty: THIRD_PARTY,
  });
  const canary = readCanary(CANARY_CSS);
  return {
    id: target.id,
    url: target.url,
    screenshots,
    elements: facts.length,
    leaks: facts.flatMap((element) => leaksOf(element, canary)),
  };
}

async function runAll(
  browser: Browser,
  targets: Target[],
  options: {
    shotsDir: string;
    concurrency: number;
    baseUrl: string;
    dismiss: string[];
  },
) {
  const viewport = VIEWPORTS[0];
  // Public routes and stories open signed out, as a visitor sees them: a
  // signed-in visit to /login is sent on to the app.
  const publicContext = await newContext(browser, viewport);
  const appContext = targets.some(isProtected)
    ? await newContext(browser, viewport)
    : null;
  if (appContext && !(await signIn(appContext, options.baseUrl)))
    throw new Error(credentialsMessage(targets));
  const contexts = appContext ? [publicContext, appContext] : [publicContext];
  for (const context of contexts) {
    // tsx keeps function names with a helper the page does not have.
    await context.addInitScript({ content: "globalThis.__name = (fn) => fn;" });
    // The clock belongs to the context: set it once, not from each worker.
    await context.clock.setFixedTime(FIXED_TIME);
  }
  const results: TargetResult[] = [];
  const queue = [...targets];
  async function attempt(target: Target) {
    const context =
      appContext && isProtected(target) ? appContext : publicContext;
    const page = await context.newPage();
    try {
      return await checkTarget(page, target, options.shotsDir, options.dismiss);
    } finally {
      await page.close().catch(() => undefined);
    }
  }
  async function worker() {
    for (;;) {
      const target = queue.shift();
      if (!target) return;
      // A page that crashes or hangs once gets one more try.
      const result = await attempt(target)
        .catch(() => attempt(target))
        .catch((error: unknown) => {
          const message =
            error instanceof Error ? error.message : String(error);
          console.warn(`failed ${target.id}: ${message.split("\n")[0]}`);
          return {
            id: target.id,
            url: target.url,
            screenshots: [],
            elements: 0,
            leaks: [],
            error: message,
          };
        });
      results.push(result);
      if (!result.error) console.log(`checked ${target.id}`);
    }
  }
  await Promise.all(Array.from({ length: options.concurrency }, worker));
  for (const context of contexts) await context.close();
  return results.sort((a, b) => a.id.localeCompare(b.id));
}

function isProtected(target: Target) {
  return target.kind === "app" && !PUBLIC_ROUTES.includes(target.id);
}

function credentialsMessage(targets: Target[]) {
  const routes = targets.filter(isProtected).map((target) => target.id);
  // Without a session these would audit the login redirect under the
  // protected route's name.
  return `set SCREENSHOT_EMAIL and SCREENSHOT_PASSWORD to check ${routes.join(", ")}, or pass only public routes (${PUBLIC_ROUTES.join(", ")})`;
}

/**
 * Fails a navigation that answered with an HTTP error: Playwright resolves
 * those, so an error page would otherwise be checked as the target.
 */
export function assertLoaded(
  response: { ok(): boolean; status(): number } | null,
  url: string,
) {
  if (response && !response.ok())
    throw new Error(`${url} answered HTTP ${response.status()}`);
}

/** A whole number of workers, at least one; null for anything else. */
export function parseConcurrency(value: string | undefined) {
  if (value === undefined) return 4;
  const workers = Number(value);
  return Number.isInteger(workers) && workers >= 1 ? workers : null;
}

async function main(argv: string[]) {
  const outDir = resolve(
    readFlag(argv, "--out") || join(tmpdir(), "ds-canary-check"),
  );
  const shotsDir = join(outDir, "screenshots");
  mkdirSync(shotsDir, { recursive: true });
  const baseUrl = readFlag(argv, "--base-url");
  const storybookUrl = readFlag(argv, "--storybook-url");
  const concurrency = parseConcurrency(readFlag(argv, "--concurrency"));
  if (concurrency === null) {
    console.error("--concurrency takes a whole number of workers, 1 or more");
    return 2;
  }
  const skip = listFlag(argv, "--skip", DEFAULT_SKIP);
  const routes = baseUrl ? listFlag(argv, "--routes", PUBLIC_ROUTES) : [];
  const stories = storybookUrl
    ? (await storyIds(storybookUrl, listFlag(argv, "--stories", [""]))).filter(
        (id) => !skip.includes(id),
      )
    : [];
  const targets: Target[] = [
    ...routes.map((route) => ({
      id: route,
      url: `${baseUrl}${route}`,
      kind: "app" as const,
    })),
    ...stories.map((id) => ({
      id,
      url: `${storybookUrl}/iframe.html?id=${id}&viewMode=story`,
      kind: "story" as const,
    })),
  ];
  if (targets.length === 0) {
    console.error(
      "nothing to check: pass --storybook-url and/or --base-url (see the header of this file)",
    );
    return 2;
  }
  const signedIn = Boolean(
    process.env.SCREENSHOT_EMAIL && process.env.SCREENSHOT_PASSWORD,
  );
  if (!signedIn && targets.some(isProtected)) {
    console.error(credentialsMessage(targets));
    return 2;
  }
  const browser = await chromium.launch({ args: DETERMINISTIC_CHROMIUM_ARGS });
  let results: TargetResult[];
  try {
    results = await runAll(browser, targets, {
      shotsDir,
      concurrency,
      baseUrl: baseUrl ?? "",
      dismiss: listFlag(argv, "--dismiss", DEFAULT_DISMISS),
    });
  } finally {
    await browser.close();
  }
  const report = buildReport(
    results.map(({ id, leaks }) => ({ target: id, leaks })),
  );
  const failed = results.filter((result) => result.error);
  const exempted = {
    skipped: SKIPPED,
    artwork: EXEMPT_ARTWORK,
    thirdPartyLabelled: Object.fromEntries(THIRD_PARTY),
  };
  writeFileSync(
    join(outDir, "report.json"),
    JSON.stringify(
      {
        checked: results.map(
          ({ id, url, screenshots, elements, error, leaks }) => ({
            id,
            url,
            screenshots,
            elements,
            leaks: leaks.length,
            error,
          }),
        ),
        exempted,
        ...report,
      },
      null,
      2,
    ),
  );
  const header = [
    `# Canary check: ${results.length - failed.length} of ${results.length} target(s) rendered`,
    "",
    ...failed.map(
      (result) => `failed: ${result.id}: ${result.error?.split("\n")[0]}`,
    ),
    "",
  ];
  writeFileSync(
    join(outDir, "report.txt"),
    [...header, formatReport(report)].join("\n"),
  );
  console.log(
    `colour ${report.totals.colour.instances}, radius ${report.totals.radius.instances}, font ${report.totals.font.instances} leak(s); report in ${outDir}`,
  );
  if (failed.length > 0) {
    // A target that never rendered was not checked, so the run is not a pass.
    console.error(`${failed.length} target(s) failed to render`);
    return 1;
  }
  return 0;
}

if (process.argv[1]?.endsWith("canary-check.ts")) {
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
