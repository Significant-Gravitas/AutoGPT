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

const SHADCN_ALIASES: Record<string, string> = {
  background: "canvas",
  foreground: "foreground-base",
  card: "surface",
  "card-foreground": "foreground-base",
  popover: "surface",
  "popover-foreground": "foreground-base",
  primary: "action",
  "primary-foreground": "action-foreground",
  secondary: "surface-hover",
  "secondary-foreground": "foreground-strong",
  muted: "surface-hover",
  "muted-foreground": "foreground-tertiary",
  accent: "brand",
  "accent-foreground": "action-foreground",
  destructive: "error",
  "destructive-foreground": "action-foreground",
  border: "border-base",
  input: "border-strong",
  ring: "focus",
  sidebar: "surface-muted",
  "sidebar-foreground": "foreground-emphasis",
  "sidebar-primary": "action",
  "sidebar-primary-foreground": "action-foreground",
  "sidebar-accent": "surface-hover",
  "sidebar-accent-foreground": "foreground-strong",
  "sidebar-border": "border-base",
  "sidebar-ring": "info",
};

let root: Root;

beforeAll(async () => {
  const from = join(frontend, "src/app/globals.css");
  const classes = "dark:bg-surface bg-primary";
  const css = `${readFileSync(from, "utf8")}\n@source inline(${JSON.stringify(classes)});\n`;
  root = (await postcss(plugins).process(css, { from })).root;
});

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
  it("aliases every shadcn name to its role", () => {
    const vars = rootVariables();
    for (const [name, role] of Object.entries(SHADCN_ALIASES)) {
      expect(vars[`--color-${name}`]).toBe(`var(--color-${role})`);
      expect(vars[`--color-${role}`]).toMatch(/^\d+ \d+ \d+$/);
    }
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
