import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { variants as textVariants } from "@/components/atoms/Text/helpers";
import { COLOR_ROLES } from "@/components/styles/color-roles";
import { tones } from "@/components/styles/tones";
import { FRONTEND_ROOT, readCss } from "./token-css";

// DESIGN.md names every colour role, tone, shadow step, layer, duration,
// easing, Text variant and Button variant, size, shape and quiet tone. This
// keeps each list equal to the source, both ways.

const read = (file: string) =>
  readFileSync(path.join(FRONTEND_ROOT, file), "utf8");
const md = read("DESIGN.md");
const ticks = (text: string) =>
  [...text.matchAll(/`([^`]+)`/g)].map(([, name]) => name);

function section(heading: string) {
  const start = md.indexOf(`\n## ${heading}\n`);
  expect(start, `DESIGN.md has "## ${heading}"`).toBeGreaterThan(-1);
  return md.slice(start, md.indexOf("\n## ", start + 1));
}

// The names in the last column of the section's table, header row left out.
const tableNames = (heading: string) =>
  section(heading)
    .split("\n")
    .filter((line) => line.startsWith("|") && !line.startsWith("| -"))
    .slice(1)
    .flatMap((row) => ticks(row.split("|").at(-2) ?? ""));

// The names on the section's lines that start with one of `labels`.
const lineNames = (heading: string, ...labels: string[]) =>
  section(heading)
    .split("\n")
    .flatMap((line) => {
      const label = labels.find((start) => line.startsWith(start));
      return label ? ticks(line.slice(label.length)) : [];
    });

// Every `<prefix><name>` written alone in backticks anywhere on the page.
const classNames = (prefix: string, name: string) =>
  [...md.matchAll(new RegExp(`\`${prefix}(${name})\``, "g"))].map(([, n]) => n);

function cssNames(file: string, pattern: RegExp, after = "") {
  const css = readCss(file);
  return [...css.slice(css.indexOf(after)).matchAll(pattern)].map(([, n]) => n);
}

// The keys of the object literal that starts at `opener`: its lines one
// level in that begin with `name:`.
function objectKeys(file: string, opener: string) {
  const text = read(file);
  expect(text.indexOf(opener), `${file} has "${opener}"`).toBeGreaterThan(-1);
  const keys: string[] = [];
  let depth = 0;
  for (const line of text.slice(text.indexOf(opener)).split("\n")) {
    const key = depth === 1 && line.match(/^\s*"?([\w-]+)"?:/);
    if (key) keys.push(key[1]);
    depth += (line.match(/{/g) ?? []).length - (line.match(/}/g) ?? []).length;
    if (depth <= 0) break;
  }
  return keys;
}

const BUTTON = "src/components/atoms/Button/helpers.ts";
const COMPACT = "src/components/atoms/Button/compact-variants.ts";
const SHAPE = "src/app/tokens/shape.css";

describe("DESIGN.md names what the source defines", () => {
  it.each([
    [
      "colour roles",
      tableNames("Colour: pick the role by what the thing is"),
      [...COLOR_ROLES],
    ],
    ["tones", tableNames("Tones"), Object.keys(tones)],
    [
      "shadow steps",
      classNames("shadow-", "[a-z0-9-]+"),
      cssNames(SHAPE, /--shadow-([a-z0-9-]+):/g, "@theme"),
    ],
    ["layers", classNames("z-", "[a-z-]+"), cssNames(SHAPE, /--z-([a-z-]+):/g)],
    [
      "durations",
      classNames("duration-", "[a-z]+"),
      cssNames("src/app/tokens/motion.css", /--duration-([a-z]+):/g),
    ],
    [
      "easings",
      classNames("ease-", "[a-z-]+"),
      cssNames(
        "src/app/globals.css",
        /--ease-([a-z-]+):/g,
        "@theme inline reference",
      ),
    ],
    [
      "Text variants",
      lineNames("Text", "- **Variants:**"),
      Object.keys(textVariants),
    ],
    [
      "Button variants",
      lineNames("Buttons", "- **`variant`**", "- Compact variants:"),
      [
        ...objectKeys(BUTTON, "variant: {"),
        ...objectKeys(COMPACT, "variant: {"),
      ],
    ],
    [
      "Button sizes",
      lineNames("Buttons", "- **`size`:**", "- Compact sizes:"),
      [...objectKeys(BUTTON, "size: {"), ...objectKeys(COMPACT, "size: {")],
    ],
    [
      "Button shapes",
      lineNames("Buttons", "- **`shape`**"),
      objectKeys(BUTTON, "const BUTTON_SHAPES = {"),
    ],
    [
      "Button quiet tones",
      lineNames("Buttons", "- **`tone`:**"),
      objectKeys(BUTTON, "const BUTTON_TONES = {"),
    ],
  ])("%s", (_name, listed, source) => {
    expect(source.length).toBeGreaterThan(0);
    expect([...new Set(listed)].sort()).toEqual([...new Set(source)].sort());
  });
});
