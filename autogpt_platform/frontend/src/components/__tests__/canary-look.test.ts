// @vitest-environment node
import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

// The canary look (src/app/tokens/themes/canary.css) proves that every pixel
// follows the tokens, so it must give every token a value of its own: a token
// it leaves out keeps its default and hides the elements that use it. See
// docs/engineering/design-system.md.

const FRONTEND_ROOT = path.resolve(__dirname, "../../..");
const read = (file: string) =>
  readFileSync(path.join(FRONTEND_ROOT, file), "utf8").replace(
    /\/\*[\s\S]*?\*\//g,
    "",
  );

const DEFAULT_FILES = [
  "src/app/tokens/theme.css",
  "src/app/tokens/typography.css",
  "src/app/tokens/shape.css",
  "src/app/tokens/motion.css",
  "src/components/styles/fonts.css",
];
const CANARY_FILE = "src/app/tokens/themes/canary.css";

function declarations(css: string, selector: RegExp) {
  return new Map(
    [...css.matchAll(selector)].flatMap(([, body]) =>
      [...body.matchAll(/(--[a-z0-9-]+):\s*([^;]+);/g)].map(
        ([, name, value]) => [name, value.replace(/\s+/g, " ").trim()] as const,
      ),
    ),
  );
}

const defaults = new Map(
  DEFAULT_FILES.flatMap((file) => [
    ...declarations(read(file), /^:root\s*\{([^}]*)\}/gm),
  ]),
);
const canary = declarations(
  read(CANARY_FILE),
  /^html\[data-look="canary"\]\s*\{([^}]*)\}/gm,
);

const TRIPLET = /^\d{1,3} \d{1,3} \d{1,3}$/;

function required() {
  return [...defaults.keys()].filter(
    (name) =>
      /^--color-/.test(name) ||
      (/^--radius(?:-|$)/.test(name) && name !== "--radius-full") ||
      name === "--shadow-color" ||
      /^--text-/.test(name),
  );
}

describe("the canary look", () => {
  it("redefines every colour role, radius, the shadow colour and every text size", () => {
    expect(required().length).toBeGreaterThan(90);
    const missing = required().filter((name) => !canary.has(name));
    expect(missing).toEqual([]);
  });

  it("redefines the sans and mono fonts", () => {
    for (const name of ["--font-geist-sans", "--font-geist-mono"])
      expect(canary.get(name), name).toBeTruthy();
  });

  it("declares no name the defaults lack, and leaves --radius-full alone", () => {
    const unknown = [...canary.keys()].filter((name) => !defaults.has(name));
    expect(unknown).toEqual([]);
    expect(canary.has("--radius-full")).toBe(false);
  });

  it("gives every colour a triplet that no default colour has", () => {
    const defaultColours = new Set(
      [...defaults]
        .filter(([name]) => /^--color-/.test(name))
        .map(([, value]) => value),
    );
    for (const [name, value] of canary) {
      if (!/^--(?:color-|shadow-color$)/.test(name)) continue;
      expect(value, name).toMatch(TRIPLET);
      expect(defaultColours.has(value), `${name}: ${value}`).toBe(false);
    }
  });

  it("squares every radius and scales every text size", () => {
    for (const [name, value] of canary) {
      if (/^--radius/.test(name)) expect(value, name).toBe("0px");
      if (/^--text-/.test(name) && defaults.get(name) !== "1")
        expect(value, name).toBe(`calc(${defaults.get(name)} * 1.25)`);
    }
  });

  it("is not imported by the app", () => {
    const importers: string[] = [];
    function walk(dir: string) {
      for (const entry of readdirSync(dir)) {
        if (entry === "node_modules" || entry === "__generated__") continue;
        const full = path.join(dir, entry);
        if (statSync(full).isDirectory()) walk(full);
        else if (
          /\.(css|tsx?)$/.test(entry) &&
          !full.endsWith(".test.ts") &&
          /themes\/canary\.css/.test(readFileSync(full, "utf8"))
        )
          importers.push(path.relative(FRONTEND_ROOT, full));
      }
    }
    walk(path.join(FRONTEND_ROOT, "src"));
    expect(importers).toEqual([]);
  });
});
