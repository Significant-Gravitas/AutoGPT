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
 * Text sizes are checked as class lists, because a size token bundles a
 * line-height that a raw `text-[13px]` never set: each list the codemod
 * rewrites is rendered inside parents at `leading-[1.5]`, `leading-[30px]`
 * and `leading-normal`, at 1280px and 390px wide, both as written and after
 * cn(). Every Text atom variant is compared as the base checkout's string
 * against this checkout's, bare and through cn() with typical caller
 * overrides, the way <Text> renders it. Font size, line-height, letter
 * spacing, weight, family, transform and colour must match in every case.
 * A few cn() checks prove tailwind-merge reads the size names as font sizes.
 *
 * Prints `old class | new class | resolved value` and exits 1 on any
 * difference. DS2 and DS5 tickets run this script; they do not edit it.
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join, resolve } from "node:path";
import { chromium, type Browser } from "@playwright/test";

import { cn } from "../src/lib/utils";
import {
  buildReplacementMap,
  buildSizeMap,
  rewriteClassList,
  SIZE_TOKENS,
  TOKEN_GROUPS,
} from "./codemod-tokens";
import { comparePngs } from "./screenshot-pages";

const localRequire = createRequire(join(process.cwd(), "package.json"));
const postcss = localRequire("postcss");

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
  // sidebar.tsx's `outline` variant read the deleted HSL variables.
  {
    group: "Sidebar",
    oldClass: "shadow-[0_0_0_1px_hsl(var(--sidebar-border))]",
    newClass: "shadow-[0_0_0_1px_rgb(var(--color-sidebar-border))]",
  },
  {
    group: "Sidebar",
    oldClass: "shadow-[0_0_0_1px_hsl(var(--sidebar-accent))]",
    newClass: "shadow-[0_0_0_1px_rgb(var(--color-sidebar-accent))]",
  },
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
  const oneForOne = TOKEN_GROUPS.flatMap((group) =>
    group.mappings.flatMap((mapping) =>
      mapping.replaces.map((oldClass) => ({
        group: group.group,
        oldClass,
        newClass: mapping.token,
      })),
    ),
  );
  return [
    ...oneForOne,
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

// Each checkout compiles with its own Tailwind, so a base checkout from
// before the Tailwind 4 upgrade still builds the way it did.
async function compileCss(dir: string, classes: string[]) {
  const dirRequire = createRequire(join(dir, "package.json"));
  const from = join(dir, "src/app/globals.css");
  const globals = readFileSync(from, "utf8");
  const raw = classes.flatMap((c) => [c, companions(c)]).join(" ");
  const major = Number(
    dirRequire("tailwindcss/package.json").version.split(".")[0],
  );
  if (major < 4) {
    const loadConfig = dirRequire("tailwindcss/loadConfig");
    const config = loadConfig(join(dir, "tailwind.config.ts"));
    const result = await postcss([
      dirRequire("tailwindcss")({
        ...config,
        content: [{ raw, extension: "html" }],
      }),
    ]).process(globals, { from });
    return result.css as string;
  }
  // Tailwind 4 takes the classes from `@source inline()` and runs the
  // checkout's own postcss.config.mjs. It runs in a plain Node process: under
  // tsx, Tailwind 4 reads tailwind.config.ts's default export as
  // `{ default: config }` and finds no theme.
  const css = `${globals}\n@source inline(${JSON.stringify(raw)});\n`;
  const script = `
    import postcss from "postcss";
    import { createRequire } from "node:module";
    const require = createRequire(${JSON.stringify(join(dir, "package.json"))});
    const { default: config } = await import(${JSON.stringify(join(dir, "postcss.config.mjs"))});
    const plugins = Object.entries(config.plugins).map(([name, options]) => require(name)(options));
    let input = "";
    for await (const chunk of process.stdin) input += chunk;
    const result = await postcss(plugins).process(input, { from: ${JSON.stringify(from)} });
    process.stdout.write(result.css);
  `;
  return execFileSync(process.execPath, ["--input-type=module", "-e", script], {
    cwd: dir,
    input: css,
    maxBuffer: 256 * 1024 * 1024,
    encoding: "utf8",
  });
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
    !pair.oldClass.includes("/")
  );
}

// Class lists the codemod produces for every raw size. Lists the codemod
// skips are not swaps, so they are not checked.
function codemodTextPairs(): Pair[] {
  const map = buildReplacementMap();
  const sizeMap = buildSizeMap();
  const samples = SIZE_TOKENS.flatMap((mapping) =>
    mapping.replaces.flatMap((raw) => [
      raw,
      `${raw} leading-5`,
      `md:${raw}`,
      `${raw} md:leading-6`,
    ]),
  );
  return samples
    .map((oldClass) => ({
      group: "Font size (codemod)",
      oldClass,
      newClass: rewriteClassList(oldClass, map, sizeMap).output,
    }))
    .filter((pair) => pair.oldClass !== pair.newClass);
}

// What callers pass to <Text className> on top of a variant.
const TEXT_OVERRIDES = [
  "text-sm",
  "text-[13px]",
  "leading-6",
  "tracking-tight",
  "font-normal",
  "text-zinc-500",
];

// Every Text variant, base string against head string, merged with cn() as
// <Text> does: bare, and under each caller override.
async function textAtomPairs(baseDir: string, headDir: string) {
  const load = async (dir: string) =>
    (
      (await import(join(dir, "src/components/atoms/Text/helpers.ts"))) as {
        variants: Record<string, string>;
      }
    ).variants;
  const oldVariants = await load(baseDir);
  const newVariants = await load(headDir);
  const pairs: Pair[] = [];
  for (const variant of Object.keys(oldVariants)) {
    for (const override of ["", ...TEXT_OVERRIDES]) {
      pairs.push({
        group: override
          ? `Text \`${variant}\` + \`${override}\``
          : `Text \`${variant}\``,
        oldClass: cn(oldVariants[variant], override),
        newClass: cn(newVariants[variant], override),
      });
    }
  }
  return pairs;
}

const TEXT_PROPS = [
  "font-size",
  "line-height",
  "letter-spacing",
  "font-weight",
  "font-family",
  "text-transform",
  "color",
] as const;
const PARENT_LINE_HEIGHTS = [
  "leading-[1.5]",
  "leading-[30px]",
  "leading-normal",
];
const TEXT_VIEWPORTS = [1280, 390];

// Every (viewport, parent line-height, as written | after cn()) value of the
// text properties, for each class list.
async function textValues(browser: Browser, css: string, lists: string[]) {
  const results: Record<string, string>[][] = lists.map(() => []);
  for (const width of TEXT_VIEWPORTS) {
    const page = await browser.newPage({ viewport: { width, height: 800 } });
    const cells = lists.flatMap((list, i) =>
      PARENT_LINE_HEIGHTS.flatMap((parent) =>
        [list, cn(list)].map(
          (cls, mode) =>
            `<div class="${parent}" style="font-size:16px"><span data-i="${i}" data-m="${mode}" class="${cls}" style="display:block">Ag</span></div>`,
        ),
      ),
    );
    await page.setContent(
      `<html><head><style>${css}</style></head><body>${cells.join("")}</body></html>`,
    );
    const values = await page.evaluate(
      (props) =>
        Array.from(document.querySelectorAll("span[data-i]")).map((el) => {
          const style = getComputedStyle(el);
          return {
            i: Number((el as HTMLElement).dataset.i),
            values: Object.fromEntries(
              props.map((prop) => [prop, style.getPropertyValue(prop)]),
            ),
          };
        }),
      [...TEXT_PROPS],
    );
    for (const { i, values: v } of values) results[i].push(v);
    await page.close();
  }
  return results;
}

// tailwind-merge must read the size names as font sizes: a colour next to a
// size survives, two sizes collapse to the last, and a later leading wins.
const CN_CHECKS: [string, string][] = [
  ["text-body text-zinc-500", "text-body text-zinc-500"],
  [
    "text-body-sm text-foreground-tertiary",
    "text-body-sm text-foreground-tertiary",
  ],
  ["text-body text-body-sm", "text-body-sm"],
  ["text-caption leading-inherit", "text-caption leading-inherit"],
  ["text-body-sm leading-inherit leading-5", "text-body-sm leading-5"],
  ["tracking-caps tracking-eyebrow", "tracking-eyebrow"],
];

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
  const codemodTexts = codemodTextPairs();
  const atomTexts = await textAtomPairs(baseDir, headDir);
  const texts = [...codemodTexts, ...atomTexts];
  const textClasses = (side: "oldClass" | "newClass") =>
    texts.flatMap((pair) =>
      [pair[side], cn(pair[side])].join(" ").split(/\s+/),
    );
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
    ...textClasses("oldClass"),
    ...PARENT_LINE_HEIGHTS,
  ]);
  const newCss = await compileCss(headDir, [
    ...newClasses,
    ...swatchExtras(swatchNew),
    ...SWATCH_HELPERS,
    ...textClasses("newClass"),
    ...PARENT_LINE_HEIGHTS,
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
  const oldText = await textValues(
    browser,
    oldCss,
    texts.map((p) => p.oldClass),
  );
  const newText = await textValues(
    browser,
    newCss,
    texts.map((p) => p.newClass),
  );
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

  const textLines = [
    "| group | old classes | new classes | font-size | line-height in a parent at leading-[1.5] / leading-[30px] / leading-normal | result |",
    "| --- | --- | --- | --- | --- | --- |",
  ];
  let textFailures = 0;
  const cases = TEXT_VIEWPORTS.length * PARENT_LINE_HEIGHTS.length * 2;
  texts.forEach((pair, i) => {
    const problems = new Set<string>();
    oldText[i].forEach((oldValue, c) => {
      for (const prop of TEXT_PROPS) {
        if (oldValue[prop] !== newText[i][c][prop]) {
          problems.add(`${prop}: ${oldValue[prop]} != ${newText[i][c][prop]}`);
        }
      }
    });
    if (problems.size) textFailures++;
    // Desktop, as written: one value per parent line-height.
    const shown = PARENT_LINE_HEIGHTS.map((_, p) => newText[i][p * 2]);
    textLines.push(
      `| ${pair.group} | \`${pair.oldClass}\` | \`${pair.newClass}\` | ${shown[0]["font-size"]} | ${shown.map((v) => v["line-height"]).join(" / ")} | ${problems.size ? `FAIL: ${[...problems].join("; ")}` : `identical (${cases} cases)`} |`,
    );
  });
  const cnFailures = CN_CHECKS.filter(([input, expected]) => {
    const merged = cn(input);
    if (merged !== expected) {
      console.error(`cn("${input}") = "${merged}", expected "${expected}"`);
    }
    return merged !== expected;
  }).length;

  const swatch = comparePngs(oldShot, newShot);
  const swatchCells = swatchOld.length * SWATCH_STEPS.length;
  console.log(lines.join("\n"));
  console.log(`\n${textLines.join("\n")}`);
  console.log(
    `\nOpacity swatches: ${swatchOld.length} colour pairs x ${SWATCH_STEPS.length} steps (0-100%) = ${swatchCells} swatches, ${swatch.diffPixels} differing pixels`,
  );
  console.log(
    `${pairs.length} pairs checked against ${baseDir === headDir ? "this checkout" : baseDir}: ${failures} failing`,
  );
  console.log(
    `${texts.length} text class lists x ${cases} cases (viewport, parent line-height, as written | cn()): ${textFailures} failing`,
  );
  console.log(`${CN_CHECKS.length} cn() checks: ${cnFailures} failing`);

  if (outDir) {
    mkdirSync(outDir, { recursive: true });
    writeFileSync(
      join(outDir, "equivalence.md"),
      `${lines.join("\n")}\n\n${textLines.join("\n")}\n`,
    );
    writeFileSync(join(outDir, "swatches-old.png"), oldShot);
    writeFileSync(join(outDir, "swatches-new.png"), newShot);
    if (swatch.diff)
      writeFileSync(join(outDir, "swatches-diff.png"), swatch.diff);
  }

  const clean =
    failures === 0 &&
    textFailures === 0 &&
    cnFailures === 0 &&
    swatch.diffPixels === 0;
  return clean ? 0 : 1;
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
