/**
 * Proves every design token resolves to exactly what the class it replaces
 * renders today (SECRT-3099, DS1).
 *
 *   npx tsx scripts/check-token-equivalence.ts [--base <frontend dir>] [--out <dir>]
 *
 * Each old class is compiled with Tailwind against the base checkout's
 * tailwind.config.ts and src/app/globals.css (`--base`, default: this
 * checkout), each new class against this checkout's. Both are then rendered
 * in Chromium at a 16px root and compared on the *computed* value of every
 * property either class emits, so `text-[1.375rem]` and `text-[22px]` are one
 * value and `text-sm` (font-size and line-height) never equals `text-[14px]`.
 * Colours are also compared as pixels at every 5% opacity step, which catches
 * an HSL triplet that rounds differently from its RGB replacement.
 *
 * Prints `old class | new class | resolved value` and exits 1 on any
 * difference. DS2 and DS5 tickets run this script; they do not edit it.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join, resolve } from "node:path";
import { chromium, type Browser } from "@playwright/test";

import { equivalencePairs } from "./codemod-tokens";
import { comparePngs } from "./screenshot-pages";

const localRequire = createRequire(join(process.cwd(), "package.json"));
const postcss = localRequire("postcss");
const tailwindcss = localRequire("tailwindcss");
const loadConfig = localRequire("tailwindcss/loadConfig");

interface Pair {
  group: string;
  oldClass: string;
  newClass: string;
}

// shadcn names keep their names; only their variables moved from HSL to RGB.
// Against a base checkout these prove the values did not move.
const UNCHANGED_NAMES = [
  "bg-background",
  "text-foreground",
  "bg-card",
  "text-card-foreground",
  "bg-popover",
  "text-popover-foreground",
  "bg-primary",
  "text-primary-foreground",
  "bg-secondary",
  "text-secondary-foreground",
  "bg-muted",
  "text-muted-foreground",
  "bg-accent",
  "text-accent-foreground",
  "bg-destructive",
  "text-destructive-foreground",
  "border-border",
  "border-input",
  "ring-ring",
  "bg-sidebar",
  "text-sidebar-foreground",
  "bg-sidebar-primary",
  "text-sidebar-primary-foreground",
  "bg-sidebar-accent",
  "text-sidebar-accent-foreground",
  "border-sidebar-border",
  "ring-sidebar-ring",
  "agpt-card",
  "agpt-border-input",
];

// The body background, and the colour every bordered element gets from
// `* { @apply border-border }` without a colour class of its own.
const SPECIAL_PAIRS: Pair[] = [
  { group: "Page", oldClass: "bg-[#F6F7F8]", newClass: "bg-canvas" },
  { group: "Page", oldClass: "border", newClass: "border" },
  { group: "Opacity", oldClass: "bg-white/50", newClass: "bg-surface/50" },
  { group: "Opacity", oldClass: "bg-stone-500/20", newClass: "bg-overlay/20" },
  {
    group: "Opacity",
    oldClass: "bg-violet-50/50",
    newClass: "bg-accent-surface/50",
  },
  { group: "Opacity", oldClass: "bg-accent/5", newClass: "bg-accent/5" },
  { group: "Opacity", oldClass: "bg-muted/50", newClass: "bg-muted/50" },
  {
    group: "Opacity",
    oldClass: "text-muted-foreground/70",
    newClass: "text-muted-foreground/70",
  },
];

function allPairs(): Pair[] {
  return [
    ...equivalencePairs(),
    ...UNCHANGED_NAMES.map((name) => ({
      group: "shadcn names",
      oldClass: name,
      newClass: name,
    })),
    ...SPECIAL_PAIRS,
  ];
}

const VARIANT_PREFIX = /^placeholder:/;

function baseClass(cls: string) {
  return cls.replace(VARIANT_PREFIX, "");
}

function isRingColor(cls: string) {
  return /^ring-(?!\d|inset|offset)/.test(cls);
}

function isDivide(cls: string) {
  return cls.startsWith("divide-");
}

// Helpers that make a colour-only utility visible without changing it.
function companions(cls: string) {
  if (isRingColor(cls)) return "ring-2";
  if (isDivide(cls)) return "divide-y";
  return "";
}

async function compileCss(dir: string, classes: string[]) {
  const config = loadConfig(join(dir, "tailwind.config.ts"));
  const globals = readFileSync(join(dir, "src/app/globals.css"), "utf8");
  const raw = classes.flatMap((c) => [c, companions(c)]).join(" ");
  const result = await postcss([
    tailwindcss({ ...config, content: [{ raw, extension: "html" }] }),
  ]).process(globals, { from: join(dir, "src/app/globals.css") });
  return result.css as string;
}

function escapeClass(cls: string) {
  return cls.replace(/[^a-zA-Z0-9_-]/g, (ch) => `\\${ch}`);
}

// Non-custom properties a class sets, read from its compiled rules.
function emittedProperties(css: string, cls: string) {
  const props = new Set<string>();
  const root = postcss.parse(css);
  const selector = `.${escapeClass(cls)}`;
  root.walkRules((rule: { selectors: string[]; walkDecls: Function }) => {
    const matches = rule.selectors.some(
      (sel: string) =>
        sel === selector ||
        sel.startsWith(`${selector} `) ||
        sel.startsWith(`${selector}:`) ||
        sel.startsWith(`${selector}>`) ||
        sel.startsWith(`${selector}::`),
    );
    if (!matches) return;
    rule.walkDecls((decl: { prop: string }) => {
      if (!decl.prop.startsWith("--")) props.add(decl.prop);
    });
  });
  if (isRingColor(cls)) props.add("box-shadow");
  // A bare `border` takes its colour from `* { @apply border-border }`.
  if (cls === "border") props.add("border-color");
  return props;
}

function elementHtml(cls: string, index: number) {
  return `<div data-i="${index}" class="${cls} ${companions(cls)}"><span>a</span><span>b</span></div>`;
}

async function computedValues(
  browser: Browser,
  css: string,
  classes: string[],
  propsPerClass: string[][],
) {
  const page = await browser.newPage();
  await page.setContent(
    `<html><head><style>${css}</style></head><body>${classes.map(elementHtml).join("")}</body></html>`,
  );
  const rootSize = await page.evaluate(
    () => getComputedStyle(document.documentElement).fontSize,
  );
  if (rootSize !== "16px") {
    throw new Error(`root font-size is ${rootSize}, expected 16px`);
  }
  const values = await page.evaluate(
    ({ classes, propsPerClass }) =>
      classes.map((cls, i) => {
        const host = document.querySelector(`[data-i="${i}"]`) as HTMLElement;
        const target = cls.startsWith("divide-")
          ? (host.children[1] as HTMLElement)
          : host;
        const style = getComputedStyle(target);
        return Object.fromEntries(
          propsPerClass[i].map((prop) => [prop, style.getPropertyValue(prop)]),
        );
      }),
    { classes, propsPerClass },
  );
  await page.close();
  return values as Record<string, string>[];
}

const SWATCH_HELPERS = ["border-4", "ring-4", "divide-y-4"];
const SWATCH_STEPS = Array.from({ length: 21 }, (_, i) => i * 5);

function swatchClass(cls: string, step: number) {
  return step === 100 ? cls : `${cls}/${step}`;
}

function isColorPair(pair: Pair) {
  return (
    /^(bg|text|border|ring|divide)-/.test(baseClass(pair.oldClass)) &&
    !pair.oldClass.includes("/") &&
    !/^text-\[\d|^text-\d|^text-(h\d|lead|large|body|small|label|eyebrow)$/.test(
      pair.newClass,
    )
  );
}

// Renders each colour at every 5% opacity step on the page background and
// on white, and returns a screenshot of the grid.
async function swatchShot(browser: Browser, css: string, classes: string[]) {
  const page = await browser.newPage({ viewport: { width: 640, height: 200 } });
  const rows = classes
    .map((cls) => {
      const base = baseClass(cls);
      const cells = SWATCH_STEPS.map((step) => {
        const c = swatchClass(base, step);
        const box = "display:inline-block;width:12px;height:12px;margin:4px";
        if (base.startsWith("text-")) {
          return `<span class="${c}" style="${box};font:12px/12px monospace">&#9608;</span>`;
        }
        if (base.startsWith("bg-")) {
          return `<span class="${c}" style="${box}"></span>`;
        }
        if (isRingColor(base)) {
          return `<span class="ring-4 ${c}" style="${box}"></span>`;
        }
        if (isDivide(base)) {
          return `<span class="divide-y-4 ${c}" style="${box}"><i style="display:block;height:4px"></i><i style="display:block;height:4px"></i></span>`;
        }
        return `<span class="border-4 ${c}" style="${box}"></span>`;
      }).join("");
      return `<div style="display:flex;gap:0;background:#fff">${cells}</div>`;
    })
    .join("");
  await page.setContent(
    `<html><head><style>${css}</style></head><body style="margin:0">${rows}</body></html>`,
  );
  const shot = await page.screenshot({
    fullPage: true,
    animations: "disabled",
  });
  await page.close();
  return shot;
}

function formatValue(values: Record<string, string>) {
  return Object.entries(values)
    .map(([prop, value]) => `${prop}: ${value}`)
    .join("; ");
}

async function main(argv: string[]) {
  const headDir = process.cwd();
  const baseIndex = argv.indexOf("--base");
  const baseDir = baseIndex >= 0 ? resolve(argv[baseIndex + 1]) : headDir;
  const outIndex = argv.indexOf("--out");
  const outDir = outIndex >= 0 ? resolve(argv[outIndex + 1]) : null;

  const pairs = allPairs();
  const oldClasses = pairs.map((p) => baseClass(p.oldClass));
  const newClasses = pairs.map((p) => baseClass(p.newClass));
  const swatchOld = pairs.filter(isColorPair).map((p) => p.oldClass);
  const swatchNew = pairs.filter(isColorPair).map((p) => p.newClass);
  const swatchExtras = (list: string[]) =>
    list.flatMap((cls) =>
      SWATCH_STEPS.map((step) => swatchClass(baseClass(cls), step)),
    );

  const oldCss = await compileCss(baseDir, [
    ...oldClasses,
    ...swatchExtras(swatchOld),
    ...SWATCH_HELPERS,
  ]);
  const newCss = await compileCss(headDir, [
    ...newClasses,
    ...swatchExtras(swatchNew),
    ...SWATCH_HELPERS,
  ]);

  const propsPerPair = pairs.map((_, i) => {
    const oldProps = emittedProperties(oldCss, oldClasses[i]);
    const newProps = emittedProperties(newCss, newClasses[i]);
    return { oldProps, newProps };
  });
  const unionProps = propsPerPair.map(({ oldProps, newProps }) =>
    [...new Set([...oldProps, ...newProps])].sort(),
  );

  const browser = await chromium.launch();
  const oldValues = await computedValues(
    browser,
    oldCss,
    oldClasses,
    unionProps,
  );
  const newValues = await computedValues(
    browser,
    newCss,
    newClasses,
    unionProps,
  );
  const oldShot = await swatchShot(browser, oldCss, swatchOld);
  const newShot = await swatchShot(browser, newCss, swatchNew);
  await browser.close();

  const lines = [
    "| group | old class | new class | resolved value (computed, 16px root) | result |",
    "| --- | --- | --- | --- | --- |",
  ];
  let failures = 0;
  pairs.forEach((pair, i) => {
    const { oldProps, newProps } = propsPerPair[i];
    const problems: string[] = [];
    if (oldProps.size === 0) problems.push("old class emits nothing");
    if (newProps.size === 0) problems.push("new class emits nothing");
    const missing = [...oldProps].filter((p) => !newProps.has(p));
    const extra = [...newProps].filter((p) => !oldProps.has(p));
    if (missing.length) problems.push(`new class does not set ${missing}`);
    if (extra.length) problems.push(`new class also sets ${extra}`);
    for (const prop of unionProps[i]) {
      if (oldValues[i][prop] !== newValues[i][prop]) {
        problems.push(
          `${prop}: ${oldValues[i][prop]} != ${newValues[i][prop]}`,
        );
      }
    }
    if (problems.length) failures++;
    lines.push(
      `| ${pair.group} | \`${pair.oldClass}\` | \`${pair.newClass}\` | ${formatValue(newValues[i])} | ${problems.length ? `FAIL: ${problems.join("; ")}` : "identical"} |`,
    );
  });

  const swatch = comparePngs(oldShot, newShot);
  const swatchCells = swatchOld.length * SWATCH_STEPS.length;
  console.log(lines.join("\n"));
  console.log(
    `\nOpacity swatches: ${swatchOld.length} colour pairs x ${SWATCH_STEPS.length} steps (0-100%) = ${swatchCells} swatches, ${swatch.diffPixels} differing pixels`,
  );
  console.log(
    `${pairs.length} pairs checked against ${baseDir === headDir ? "this checkout" : baseDir}: ${failures} failing`,
  );

  if (outDir) {
    mkdirSync(outDir, { recursive: true });
    writeFileSync(join(outDir, "equivalence.md"), `${lines.join("\n")}\n`);
    writeFileSync(join(outDir, "swatches-old.png"), oldShot);
    writeFileSync(join(outDir, "swatches-new.png"), newShot);
    if (swatch.diff)
      writeFileSync(join(outDir, "swatches-diff.png"), swatch.diff);
  }

  return failures === 0 && swatch.diffPixels === 0 ? 0 : 1;
}

main(process.argv.slice(2)).then(
  (code) => {
    process.exitCode = code;
  },
  (error) => {
    console.error(error);
    process.exitCode = 1;
  },
);
