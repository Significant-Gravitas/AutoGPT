// @vitest-environment node
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import postcss, { type AcceptedPlugin, type Root } from "postcss";
import { describe, expect, it } from "vitest";
import tailwindConfig from "../../../../tailwind.config";
import { COLOR_ROLES } from "../color-roles";

// One colour system: the roles. Each has a value in tokens/theme.css's one
// `:root` block and an `@theme inline` entry in globals.css that turns it into
// classes; Tailwind's own palette is reset, so nothing else is a colour.

const FRONTEND_ROOT = path.resolve(__dirname, "../../../..");
const read = (file: string) =>
  readFileSync(path.join(FRONTEND_ROOT, file), "utf8");
const themeCss = read("src/app/tokens/theme.css");
const globalsCss = read("src/app/globals.css");

interface PostcssConfig {
  plugins: Record<string, Record<string, unknown>>;
}

// The app's stylesheet through its own PostCSS steps, with one background
// class per role and a few palette classes added to what Tailwind scans.
async function compile(classes: string[]) {
  const require = createRequire(path.join(FRONTEND_ROOT, "package.json"));
  const { default: config }: { default: PostcssConfig } = require(
    path.join(FRONTEND_ROOT, "postcss.config.mjs"),
  );
  const plugins = Object.entries(config.plugins).map(
    ([name, options]) => require(name)(options) as AcceptedPlugin,
  );
  const from = path.join(FRONTEND_ROOT, "src/app/globals.css");
  const css = `${globalsCss}\n@source inline(${JSON.stringify(classes.join(" "))});\n`;
  return (await postcss(plugins).process(css, { from })).root;
}

function declarationsOf(root: Root, selector: string) {
  const found: Record<string, string> = {};
  root.walkRules((rule) => {
    if (!rule.selectors.includes(selector)) return;
    rule.walkDecls((decl) => {
      found[decl.prop] = decl.value;
    });
  });
  return found;
}

// Each role's alpha in globals.css's `@theme inline`: `1`, a fixed opacity
// (the inbox's black washes) or `var(--alpha-<role>)`, which a scoped palette
// such as `palette-classic` can swap.
function registeredAlphas() {
  const theme = [
    ...globalsCss.matchAll(/^@theme inline[^{]*\{([^}]*)\}/gm),
  ].join("\n");
  const alphas = new Map<string, string>();
  for (const [, name, read, alpha] of theme.matchAll(
    /--color-([a-z-]+):\s*rgb\(\s*var\(--color-([a-z-]+)\)\s*\/\s*(1|0?\.\d+|var\(--alpha-[a-z-]+\))\s*\);/g,
  )) {
    if (name === read) alphas.set(name, alpha);
  }
  return alphas;
}

function rootRoleBlocks() {
  return [...themeCss.matchAll(/^:root\s*\{([^}]*)\}/gm)].filter(([, body]) =>
    /--color-[a-z-]+:\s*\d+ \d+ \d+;/.test(body),
  );
}

describe("colour roles", () => {
  it("keeps every role's value in one :root block of tokens/theme.css", () => {
    const blocks = rootRoleBlocks();
    expect(blocks).toHaveLength(1);
    const declared = [
      ...blocks[0][1].matchAll(/--color-([a-z-]+):\s*\d+ \d+ \d+;/g),
    ].map(([, name]) => name);
    expect([...declared].sort()).toEqual([...COLOR_ROLES].sort());
  });

  it("maps every role into Tailwind in globals.css's @theme inline", () => {
    const alphas = registeredAlphas();
    for (const role of COLOR_ROLES) {
      const alpha = alphas.get(role);
      expect(alpha, role).toBeDefined();
      if (alpha?.startsWith("var("))
        expect(alpha, role).toBe(`var(--alpha-${role})`);
    }
  });

  it("resets Tailwind's palette before any colour is registered", () => {
    const reset = themeCss.search(/@theme\s*\{\s*--color-\*:\s*initial;\s*\}/);
    expect(reset).toBeGreaterThanOrEqual(0);
    expect(themeCss.search(/^@theme inline/m)).toBeGreaterThan(reset);
    expect(globalsCss.indexOf('@import "./tokens/theme.css"')).toBeLessThan(
      globalsCss.search(/^@theme/m),
    );
    expect(existsSync(path.join(FRONTEND_ROOT, "tailwind.v3-preset.ts"))).toBe(
      false,
    );
    expect(tailwindConfig).not.toHaveProperty("presets");
    expect(tailwindConfig.theme).not.toHaveProperty("colors");
    expect(tailwindConfig.theme.extend).not.toHaveProperty("colors");
  });

  it("generates a class for every role, and none for the palette", async () => {
    const palette = ["bg-red-500", "bg-zinc-100", "bg-white", "bg-black"];
    const root = await compile([
      ...COLOR_ROLES.map((role) => `bg-${role}`),
      ...palette,
    ]);
    // A long entry wraps over lines in globals.css; compare without spaces.
    const squash = (value: string) => value.replace(/\s+/g, "");
    const alphas = registeredAlphas();
    for (const role of COLOR_ROLES) {
      const found = declarationsOf(root, `.bg-${role}`);
      expect(Object.keys(found), role).toEqual(["background-color"]);
      expect(squash(found["background-color"]), role).toBe(
        squash(`rgb(var(--color-${role}) / ${alphas.get(role)})`),
      );
    }
    for (const name of palette) {
      expect(declarationsOf(root, `.${name}`), name).toEqual({});
    }
  });
});
