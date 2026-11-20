// @vitest-environment node
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import postcss, { type AcceptedPlugin, type Root } from "postcss";
import { beforeAll, describe, expect, it } from "vitest";

interface PostcssConfig {
  plugins: Record<string, Record<string, unknown>>;
}

const frontend = join(__dirname, "..", "..");
const require = createRequire(join(frontend, "package.json"));
const { default: postcssConfig }: { default: PostcssConfig } = require(
  join(frontend, "postcss.config.mjs"),
);
const plugins = Object.entries(postcssConfig.plugins).map(
  ([name, options]) => require(name)(options) as AcceptedPlugin,
);

// The shadcn names streamdown's prebuilt markup uses, and the role each one
// renders (src/app/tokens/theme.css). The app itself names roles only.
const VENDOR_ALIASES: Record<string, string> = {
  background: "canvas",
  foreground: "foreground-base",
  muted: "surface-hover",
  "muted-foreground": "foreground-tertiary",
  border: "border-base",
  primary: "action",
  "primary-foreground": "action-foreground",
};

let root: Root;

beforeAll(async () => {
  const from = join(frontend, "src/app/globals.css");
  const classes = [
    "dark:bg-surface bg-muted/40 text-card",
    ...Object.keys(VENDOR_ALIASES).map((name) => `bg-${name}`),
  ].join(" ");
  const css = `${readFileSync(from, "utf8")}\n@source inline(${JSON.stringify(classes)});\n`;
  root = (await postcss(plugins).process(css, { from })).root;
});

function declarations(selector: string) {
  const found: Record<string, string> = {};
  root.walkRules((rule) => {
    if (rule.selector !== selector) return;
    rule.walkDecls((decl) => {
      found[decl.prop] = decl.value;
    });
  });
  return found;
}

function rootVariables() {
  const vars: Record<string, string> = {};
  root.walkRules(":root", (rule) => {
    rule.walkDecls(/^--color-/, (decl) => {
      vars[decl.prop] = decl.value;
    });
  });
  return vars;
}

describe("theme", () => {
  it("keeps only streamdown's shadcn names, each rendering its role", () => {
    const vars = rootVariables();
    expect(
      Object.keys(vars).filter((name) => vars[name].startsWith("var(")),
    ).toEqual([]);
    for (const [name, role] of Object.entries(VENDOR_ALIASES)) {
      expect(vars[`--color-${role}`]).toMatch(/^\d+ \d+ \d+$/);
      expect(declarations(`.bg-${name}`)).toEqual({
        "background-color": `rgb(var(--color-${role}) / 1)`,
      });
    }
    expect(declarations(".bg-muted\\/40")).toEqual({
      "background-color": "rgb(var(--color-surface-hover) / 0.4)",
    });
  });

  it("generates nothing for the shadcn names it dropped", () => {
    expect(declarations(".text-card")).toEqual({});
  });

  it("has no .dark overrides of the colour roles", () => {
    root.walkRules((rule) => {
      if (!rule.selectors.includes(".dark")) return;
      rule.walkDecls((decl) => {
        expect(decl.prop.startsWith("--color-")).toBe(false);
      });
    });
  });

  it("applies dark: under a .dark ancestor, not the OS setting", () => {
    let found = false;
    root.walkRules((rule) => {
      if (!rule.selector.includes("dark\\:bg-surface")) return;
      found = true;
      expect(rule.selector).toContain(".dark");
    });
    expect(found).toBe(true);
    root.walkAtRules("media", (at) => {
      expect(at.params).not.toContain("prefers-color-scheme");
    });
  });
});
