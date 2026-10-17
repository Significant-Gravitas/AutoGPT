#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const FRONTEND = path.resolve(import.meta.dirname, "..");
const SRC = path.join(FRONTEND, "src");

const SKIP_DIRS = new Set([
  "node_modules",
  ".next",
  "__generated__",
  "coverage",
  "dist",
]);

const CUSTOM_RAMPS = new Set([
  "slate",
  "zinc",
  "red",
  "orange",
  "yellow",
  "green",
  "purple",
  "pink",
  "customGray",
]);
const CUSTOM_FLATS = new Set([
  "white",
  "black",
  "textGrey",
  "textBlack",
  "bgLightGrey",
]);
const DEFAULT_PALETTE = new Set([
  "gray",
  "neutral",
  "stone",
  "amber",
  "lime",
  "emerald",
  "teal",
  "cyan",
  "sky",
  "blue",
  "indigo",
  "violet",
  "fuchsia",
  "rose",
]);
const SHADE = /^(?:25|50|100|150|200|300|400|500|600|700|800|900|950)$/;

const SEMANTIC = [
  "sidebar-primary-foreground",
  "sidebar-accent-foreground",
  "sidebar-primary",
  "sidebar-foreground",
  "sidebar-accent",
  "sidebar-border",
  "sidebar-ring",
  "sidebar",
  "primary-foreground",
  "secondary-foreground",
  "destructive-foreground",
  "muted-foreground",
  "accent-foreground",
  "popover-foreground",
  "card-foreground",
  "primary",
  "secondary",
  "destructive",
  "muted",
  "accent",
  "popover",
  "card",
  "background",
  "foreground",
  "border",
  "input",
  "ring",
].sort((a, b) => b.length - a.length);

const PROP =
  "bg|text|border(?:-[trblxy])?|ring(?:-offset)?|fill|stroke|from|to|via|outline|divide|placeholder|caret|decoration|accent";

const CLASS_RE = new RegExp(
  `(?<![\\w-])((?:(?:[a-z0-9]+|\\[[^\\]]+\\]):)*)(?:${PROP})-(\\[(?:[^\\]]+)\\]|[a-zA-Z][\\w-]*)(?:\\/\\d+)?`,
  "g",
);

const ARBITRARY_LENGTH_RE =
  /(?<![\\w-])(?:[a-z0-9-]+:)*(?:p|px|py|pt|pr|pb|pl|m|mx|my|mt|mr|mb|ml|gap|gap-x|gap-y|w|h|size|min-w|max-w|min-h|max-h|top|right|bottom|left|inset|inset-x|inset-y|basis|space-x|space-y|leading|tracking|rounded|z)-\[\d/g;

const ARBITRARY_TYPE_RE = /(?<![\\w-])(?:[a-z0-9-]+:)*text-\[\d/g;

const ARBITRARY_FONT_RE = /(?<![\\w-])(?:[a-z0-9-]+:)*font-\[\d/g;

const HEX_RE =
  /#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3})\b/g;
const FUNC_COLOR_RE = /\b(?:rgba?|hsla?|oklch|oklab|color-mix)\(/g;
const DARK_RE = /(?<![\\w-])dark:/g;
const AGPT_RE = /(?<![\\w-])agpt-[a-z0-9-]+/g;
const LEGACY_IMPORT_RE = /@\/components\/__legacy__/g;

const RADIUS_NAMED_RE =
  /(?<![\\w-])rounded-(?:xsmall|small|medium|large|xlarge|2xlarge)\b/g;
const RADIUS_TAILWIND_RE =
  /(?<![\\w-])rounded(?:-(?:none|sm|md|lg|xl|2xl|3xl|full))?\b/g;

const IMPORT_TARGETS = [
  ["atomsButton", /@\/components\/atoms\/Button/g],
  ["uiButton", /@\/components\/ui\/button/g],
  ["legacyButton", /@\/components\/__legacy__\/ui\/button/g],
  ["atomsInput", /@\/components\/atoms\/Input/g],
  ["uiInput", /@\/components\/ui\/input/g],
  ["legacyInput", /@\/components\/__legacy__\/ui\/input/g],
  ["atomsText", /@\/components\/atoms\/Text/g],
];

const SHADCN_ZINC_LIGHT = {
  background: "0 0% 100%",
  foreground: "240 10% 3.9%",
  card: "0 0% 100%",
  "card-foreground": "240 10% 3.9%",
  popover: "0 0% 100%",
  "popover-foreground": "240 10% 3.9%",
  primary: "240 5.9% 10%",
  "primary-foreground": "0 0% 98%",
  secondary: "240 4.8% 95.9%",
  "secondary-foreground": "240 5.9% 10%",
  muted: "240 4.8% 95.9%",
  "muted-foreground": "240 3.8% 46.1%",
  accent: "240 4.8% 95.9%",
  "accent-foreground": "240 5.9% 10%",
  destructive: "0 84.2% 60.2%",
  "destructive-foreground": "0 0% 98%",
  border: "240 5.9% 90%",
  input: "240 5.9% 90%",
  ring: "240 10% 3.9%",
  radius: "0.5rem",
  "chart-1": "12 76% 61%",
  "chart-2": "173 58% 39%",
  "chart-3": "197 37% 24%",
  "chart-4": "43 74% 66%",
  "chart-5": "27 87% 67%",
};

const SHADCN_ZINC_DARK = {
  background: "240 10% 3.9%",
  foreground: "0 0% 98%",
  card: "240 10% 3.9%",
  "card-foreground": "0 0% 98%",
  popover: "240 10% 3.9%",
  "popover-foreground": "0 0% 98%",
  primary: "0 0% 98%",
  "primary-foreground": "240 5.9% 10%",
  secondary: "240 3.7% 15.9%",
  "secondary-foreground": "0 0% 98%",
  muted: "240 3.7% 15.9%",
  "muted-foreground": "240 5% 64.9%",
  accent: "240 3.7% 15.9%",
  "accent-foreground": "0 0% 98%",
  destructive: "0 62.8% 30.6%",
  "destructive-foreground": "0 0% 98%",
  border: "240 3.7% 15.9%",
  input: "240 3.7% 15.9%",
  ring: "240 4.9% 83.9%",
  "chart-1": "220 70% 50%",
  "chart-2": "160 60% 45%",
  "chart-3": "30 80% 55%",
  "chart-4": "280 65% 60%",
  "chart-5": "340 75% 55%",
};

const SHADCN_SIDEBAR_LIGHT = {
  "sidebar-background": "0 0% 98%",
  "sidebar-foreground": "240 5.3% 26.1%",
  "sidebar-primary": "240 5.9% 10%",
  "sidebar-primary-foreground": "0 0% 98%",
  "sidebar-accent": "240 4.8% 95.9%",
  "sidebar-accent-foreground": "240 5.9% 10%",
  "sidebar-border": "220 13% 91%",
  "sidebar-ring": "217.2 91.2% 59.8%",
};

const SHADCN_SIDEBAR_DARK = {
  "sidebar-background": "240 5.9% 10%",
  "sidebar-foreground": "240 4.8% 95.9%",
  "sidebar-primary": "224.3 76.3% 48%",
  "sidebar-primary-foreground": "0 0% 100%",
  "sidebar-accent": "240 3.7% 15.9%",
  "sidebar-accent-foreground": "240 4.8% 95.9%",
  "sidebar-border": "240 3.7% 15.9%",
  "sidebar-ring": "217.2 91.2% 59.8%",
};

function emptyCounts() {
  return {
    semantic: 0,
    customRamp: 0,
    customFlat: 0,
    defaultPalette: 0,
    arbitraryColor: 0,
    arbitraryLength: 0,
    arbitraryType: 0,
    arbitraryFont: 0,
    hexLiteral: 0,
    functionalColor: 0,
    darkVariant: 0,
    darkPrefix: 0,
    agptClass: 0,
    legacyImport: 0,
    radiusNamed: 0,
    radiusTailwind: 0,
  };
}

function areaOf(rel) {
  if (rel.includes("__legacy__")) return "legacy";
  if (
    rel.includes("/components/atoms/") ||
    rel.includes("/components/molecules/") ||
    rel.includes("/components/organisms/") ||
    rel.includes("/components/styles/") ||
    rel.includes("/components/tokens/") ||
    rel.includes("/components/ui/")
  ) {
    return "designSystem";
  }
  if (rel.startsWith("src/app/")) return "app";
  return "other";
}

function isArtwork(rel) {
  return /Illustration|AutoGPTLogo|\.generated\.ts$|NotionAvatar|FolderIcon\.tsx|FileIllustration|\/icons\.tsx$/.test(
    rel,
  );
}

function bucketOf(rel) {
  if (
    rel.includes("__tests__") ||
    rel.includes(".test.") ||
    rel.includes(".spec.") ||
    rel.includes(".stories.")
  ) {
    return "testsAndStories";
  }
  return "product";
}

function stripComments(source) {
  let out = "";
  let i = 0;
  while (i < source.length) {
    const two = source.slice(i, i + 2);
    if (two === "//") {
      const end = source.indexOf("\n", i);
      i = end === -1 ? source.length : end;
      continue;
    }
    if (two === "/*") {
      const end = source.indexOf("*/", i + 2);
      i = end === -1 ? source.length : end + 2;
      continue;
    }
    out += source[i];
    i += 1;
  }
  return out;
}

function classifyColor(token) {
  if (token.startsWith("[")) {
    const inner = token.slice(1, -1);
    if (/^(?:#|rgb|hsl|oklch|oklab|color-mix)/.test(inner))
      return "arbitraryColor";
    return null;
  }
  for (const name of SEMANTIC) {
    if (token === name) return "semantic";
  }
  for (const name of CUSTOM_FLATS) {
    if (token === name) return "customFlat";
  }
  const dash = token.indexOf("-");
  const family = dash === -1 ? token : token.slice(0, dash);
  const shade = dash === -1 ? "" : token.slice(dash + 1);
  if (CUSTOM_RAMPS.has(family) && SHADE.test(shade)) return "customRamp";
  if (DEFAULT_PALETTE.has(family) && SHADE.test(shade)) return "defaultPalette";
  if (family === "customGray" && SHADE.test(shade)) return "customRamp";
  return null;
}

function bump(map, key, n = 1) {
  map[key] = (map[key] ?? 0) + n;
}

function walk(dir, files) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP_DIRS.has(entry.name)) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(full, files);
      continue;
    }
    if (!/\.(tsx|ts|css)$/.test(entry.name)) continue;
    files.push(full);
  }
}

function parseThemeBlock(css, selector) {
  const start = css.indexOf(selector);
  if (start === -1) return {};
  const open = css.indexOf("{", start);
  const close = css.indexOf("}", open);
  const body = css.slice(open + 1, close);
  const vars = {};
  for (const match of body.matchAll(/--([\w-]+)\s*:\s*([^;]+);/g)) {
    vars[match[1]] = match[2].replace(/\/\*.*\*\//g, "").trim();
  }
  return vars;
}

function compareTheme(actual, expected) {
  const same = [];
  const different = [];
  const missing = [];
  for (const [key, value] of Object.entries(expected)) {
    if (!(key in actual)) {
      missing.push(key);
      continue;
    }
    if (actual[key] === value) same.push(key);
    else different.push({ token: key, actual: actual[key], shadcn: value });
  }
  return {
    same: same.length,
    different,
    missing,
    compared: Object.keys(expected).length,
  };
}

function scanFile(source) {
  const counts = emptyCounts();
  const stripped = stripComments(source);
  for (const match of stripped.matchAll(CLASS_RE)) {
    const variants = match[1] ?? "";
    const kind = classifyColor(match[2]);
    if (!kind) continue;
    counts[kind] += 1;
    if (variants.split(":").includes("dark")) counts.darkVariant += 1;
  }
  counts.arbitraryLength = [...stripped.matchAll(ARBITRARY_LENGTH_RE)].length;
  counts.arbitraryType = [...stripped.matchAll(ARBITRARY_TYPE_RE)].length;
  counts.arbitraryFont = [...stripped.matchAll(ARBITRARY_FONT_RE)].length;
  counts.darkPrefix = [...stripped.matchAll(DARK_RE)].length;
  for (const match of stripped.matchAll(HEX_RE)) {
    const before = stripped[match.index - 1];
    if (before === "[") continue;
    counts.hexLiteral += 1;
  }
  counts.functionalColor = [...stripped.matchAll(FUNC_COLOR_RE)].length;
  counts.agptClass = [...stripped.matchAll(AGPT_RE)].length;
  counts.legacyImport = [...stripped.matchAll(LEGACY_IMPORT_RE)].length;
  counts.radiusNamed = [...stripped.matchAll(RADIUS_NAMED_RE)].length;
  const tailwindRadius = [...stripped.matchAll(RADIUS_TAILWIND_RE)].length;
  counts.radiusTailwind = Math.max(0, tailwindRadius - counts.radiusNamed);
  return counts;
}

function addCounts(target, extra) {
  for (const key of Object.keys(target)) target[key] += extra[key];
}

function bypassOf(counts) {
  return (
    counts.customRamp +
    counts.customFlat +
    counts.defaultPalette +
    counts.arbitraryColor +
    counts.hexLiteral +
    counts.functionalColor
  );
}

function outsideBothOf(counts) {
  return (
    counts.defaultPalette +
    counts.arbitraryColor +
    counts.hexLiteral +
    counts.functionalColor
  );
}

function tallyFamilies(source) {
  const families = {};
  const stripped = stripComments(source);
  for (const match of stripped.matchAll(CLASS_RE)) {
    const token = match[2];
    if (token.startsWith("[")) continue;
    const dash = token.indexOf("-");
    const family = dash === -1 ? token : token.slice(0, dash);
    const shade = dash === -1 ? "" : token.slice(dash + 1);
    if (!SHADE.test(shade)) continue;
    if (!DEFAULT_PALETTE.has(family) && !CUSTOM_RAMPS.has(family)) continue;
    const key = `${DEFAULT_PALETTE.has(family) ? "default" : "custom"}:${family}`;
    families[key] = (families[key] ?? 0) + 1;
  }
  return families;
}

function topFiles(rows, key, limit) {
  return rows
    .filter((row) => row.counts[key] > 0)
    .sort((a, b) => b.counts[key] - a.counts[key])
    .slice(0, limit)
    .map((row) => ({ file: row.rel, count: row.counts[key], area: row.area }));
}

function selfTest() {
  const sample = [
    'className="hover:bg-zinc-100 text-muted-foreground border-gray-200 bg-[#fff] text-[13px] dark:bg-white"',
    "const paint = '#112233'; style={{ backgroundColor: 'rgb(1, 2, 3)' }}",
    'import { Button } from "@/components/__legacy__/ui/button";',
    "rounded-small rounded-lg agpt-card",
    "font-[600] p-[13px]",
  ].join("\n");
  const counts = scanFile(sample);
  const expect = {
    customRamp: 1,
    semantic: 1,
    defaultPalette: 1,
    arbitraryColor: 1,
    customFlat: 1,
    darkVariant: 1,
    arbitraryType: 1,
    hexLiteral: 1,
    darkPrefix: 1,
    functionalColor: 1,
    legacyImport: 1,
    radiusNamed: 1,
    agptClass: 1,
    arbitraryFont: 1,
    arbitraryLength: 1,
  };
  const failures = [];
  for (const [key, value] of Object.entries(expect)) {
    if (counts[key] !== value) {
      failures.push(`${key} expected ${value} got ${counts[key]}`);
    }
  }
  if (failures.length) {
    console.error(failures.join("\n"));
    process.exit(1);
  }
  console.log("self-test ok");
}

function main() {
  if (process.argv.includes("--self-test")) {
    selfTest();
    return;
  }

  const files = [];
  walk(SRC, files);
  const rows = [];
  const product = emptyCounts();
  const productChrome = emptyCounts();
  const artwork = emptyCounts();
  const testsAndStories = emptyCounts();
  const byArea = {
    designSystem: emptyCounts(),
    legacy: emptyCounts(),
    app: emptyCounts(),
    other: emptyCounts(),
  };
  const imports = Object.fromEntries(IMPORT_TARGETS.map(([name]) => [name, 0]));
  let productFiles = 0;
  let chromeFiles = 0;
  let chromeFilesWithColorBypass = 0;
  let chromeFilesWithSemantic = 0;
  let productFilesWithSemantic = 0;
  let artworkFiles = 0;
  let chromeFilesOutsideBoth = 0;
  const chromeFamilies = {};
  let paletteHexDefinitions = 0;

  for (const full of files) {
    const rel = path.relative(FRONTEND, full).split(path.sep).join("/");
    const source = fs.readFileSync(full, "utf8");
    const counts = scanFile(source);
    if (rel === "src/components/styles/colors.ts") {
      paletteHexDefinitions = counts.hexLiteral;
      counts.hexLiteral = 0;
    }
    const area = areaOf(rel);
    const bucket = bucketOf(rel);
    rows.push({ rel, area, bucket, counts });
    addCounts(bucket === "product" ? product : testsAndStories, counts);
    if (bucket === "product") {
      addCounts(byArea[area], counts);
      productFiles += 1;
      if (counts.semantic > 0) productFilesWithSemantic += 1;
      if (isArtwork(rel)) {
        addCounts(artwork, counts);
        artworkFiles += 1;
      } else {
        addCounts(productChrome, counts);
        chromeFiles += 1;
        if (bypassOf(counts) > 0) chromeFilesWithColorBypass += 1;
        if (outsideBothOf(counts) > 0) chromeFilesOutsideBoth += 1;
        if (counts.semantic > 0) chromeFilesWithSemantic += 1;
        for (const [key, n] of Object.entries(tallyFamilies(source))) {
          chromeFamilies[key] = (chromeFamilies[key] ?? 0) + n;
        }
      }
    }
    if (bucket === "product") {
      for (const [name, re] of IMPORT_TARGETS) {
        imports[name] += [...source.matchAll(re)].length;
      }
    }
  }

  const globals = fs.readFileSync(path.join(SRC, "app", "globals.css"), "utf8");
  const light = parseThemeBlock(globals, ":root");
  const dark = parseThemeBlock(globals, ".dark {");
  const themeOverlap = {
    source:
      "https://github.com/shadcn-ui/ui/commit/d5920cc3c1c12ed7d54c384e86fd207b648b6e42",
    light: compareTheme(light, SHADCN_ZINC_LIGHT),
    dark: compareTheme(dark, SHADCN_ZINC_DARK),
    sidebarLight: compareTheme(light, SHADCN_SIDEBAR_LIGHT),
    sidebarDark: compareTheme(dark, SHADCN_SIDEBAR_DARK),
  };

  const productRows = rows.filter((row) => row.bucket === "product");
  const chromeRows = productRows.filter((row) => !isArtwork(row.rel));
  const report = {
    generatedBy: "scripts/audit-design-tokens.mjs",
    unit: "source occurrence after comment stripping",
    filesScanned: files.length,
    paletteHexDefinitions,
    productFiles,
    productFilesWithSemantic,
    chromeFiles,
    chromeFilesWithColorBypass,
    chromeFilesWithSemantic,
    chromeFilesOutsideBoth,
    chromeFamilies,
    artworkFiles,
    artworkNote:
      "Artwork files are illustrations, logos, generated avatar parts, and icon sheets. Their hex is paint, not UI chrome. productChrome excludes them. Color picker data such as ColorStep/helpers.ts stays in chrome because those classes style product UI.",
    exclusions: [...SKIP_DIRS],
    notMeasured: [
      "class names built by string concatenation or template variables",
      "the rendered DOM",
      "node_modules and src/app/api/__generated__",
    ],
    limiter:
      "Each file is read once. A match is one regex hit. The count cannot include a second pass over compiled CSS. It undercounts classes assembled at runtime.",
    product,
    productChrome,
    artwork,
    testsAndStories,
    byArea,
    imports,
    themeOverlap,
    topChromeBypassFiles: topFiles(
      chromeRows.map((row) => ({
        ...row,
        counts: { ...row.counts, bypass: bypassOf(row.counts) },
      })),
      "bypass",
      15,
    ),
    topBypassFiles: topFiles(
      productRows.map((row) => ({
        ...row,
        counts: { ...row.counts, bypass: bypassOf(row.counts) },
      })),
      "bypass",
      20,
    ),
    topDefaultPaletteFiles: topFiles(productRows, "defaultPalette", 15),
    topHexFiles: topFiles(productRows, "hexLiteral", 15),
    topLegacyImportFiles: topFiles(productRows, "legacyImport", 15),
  };

  const outPath = path.join(
    FRONTEND,
    "scripts",
    "audit-design-tokens.report.json",
  );
  if (!process.argv.includes("--no-write")) {
    fs.writeFileSync(outPath, JSON.stringify(report, null, 2) + "\n");
  }
  console.log(
    JSON.stringify(
      {
        filesScanned: report.filesScanned,
        paletteHexDefinitions: report.paletteHexDefinitions,
        productFiles: report.productFiles,
        chromeFiles: report.chromeFiles,
        chromeFilesWithColorBypass: report.chromeFilesWithColorBypass,
        chromeFilesOutsideBoth: report.chromeFilesOutsideBoth,
        chromeOutsideBoth: outsideBothOf(report.productChrome),
        chromeFamilies: report.chromeFamilies,
        artworkFiles: report.artworkFiles,
        productChrome: report.productChrome,
        artwork: {
          hexLiteral: report.artwork.hexLiteral,
          customRamp: report.artwork.customRamp,
          defaultPalette: report.artwork.defaultPalette,
        },
        product,
        byArea: Object.fromEntries(
          Object.entries(byArea).map(([name, counts]) => [
            name,
            {
              semantic: counts.semantic,
              customRamp: counts.customRamp,
              customFlat: counts.customFlat,
              defaultPalette: counts.defaultPalette,
              arbitraryColor: counts.arbitraryColor,
              hexLiteral: counts.hexLiteral,
              legacyImport: counts.legacyImport,
            },
          ]),
        ),
        imports,
        themeOverlap: {
          lightSame: themeOverlap.light.same,
          lightDifferent: themeOverlap.light.different,
          darkSame: themeOverlap.dark.same,
          darkDifferent: themeOverlap.dark.different,
          sidebarLightSame: themeOverlap.sidebarLight.same,
          sidebarDarkSame: themeOverlap.sidebarDark.same,
        },
        wrote: process.argv.includes("--no-write") ? null : outPath,
      },
      null,
      2,
    ),
  );
}

main();
