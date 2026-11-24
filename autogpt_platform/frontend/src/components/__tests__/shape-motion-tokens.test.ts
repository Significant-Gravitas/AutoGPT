// @vitest-environment node
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import postcss, { type AcceptedPlugin, type Root } from "postcss";
import { describe, expect, it } from "vitest";
import tailwindConfig from "../../../tailwind.config";

// The radius, shadow, layer and motion tokens: their values live in
// tokens/shape.css and tokens/motion.css, globals.css registers each one, and
// tailwind.config.ts and globals.css's `@theme` hold none of them. See
// docs/engineering/design-system.md.

const FRONTEND_ROOT = path.resolve(__dirname, "../../..");
const read = (file: string) =>
  readFileSync(path.join(FRONTEND_ROOT, file), "utf8").replace(
    /\/\*[\s\S]*?\*\//g,
    "",
  );
const globalsCss = read("src/app/globals.css");
const shapeCss = read("src/app/tokens/shape.css");
const motionCss = read("src/app/tokens/motion.css");
const colourRoles = new Set(
  [...read("src/app/tokens/theme.css").matchAll(/(--color-[a-z0-9-]+):/g)].map(
    ([, name]) => name,
  ),
);

function rootVariables(css: string) {
  return [...css.matchAll(/^:root\s*\{([^}]*)\}/gm)].flatMap(([, body]) =>
    [...body.matchAll(/(--[a-z0-9-]+):\s*([^;]+);/g)].map(([, name]) => name),
  );
}

function themeEntries(css: string) {
  return [...css.matchAll(/^@theme[^{]*\{([^}]*)\}/gm)].flatMap(([, body]) =>
    [...body.matchAll(/(--[a-z0-9*-]+):\s*([^;]+);/g)].map(
      ([, name, value]) => [name, value.replace(/\s+/g, " ").trim()] as const,
    ),
  );
}

// The `@theme` entry that turns a token into classes, and a class that must
// read it.
function registration(name: string) {
  const radius = /^--radius(?:-(.+))?$/.exec(name);
  if (radius)
    return {
      entry: `${name}: var(${name})`,
      className: radius[1] ? `rounded-${radius[1]}` : "rounded",
    };
  const step = /^--(z|duration|ease)-(.+)$/.exec(name);
  if (!step) return null;
  const [, kind, value] = step;
  const namespace = {
    z: "z-index",
    duration: "transition-duration",
    ease: "ease",
  }[kind];
  return {
    entry: `--${namespace}-${value}: var(${name})`,
    className: `${kind}-${value}`,
  };
}

// Variables on :root that are not a class of their own: the shadow colour
// (read by the shadow steps), the smooth-shadow-ring-sm ring colour (read by
// that utility) and the app shell's sizes (read by nothing yet).
const NOT_CLASSES = /^--(?:shadow-color|smooth-ring-color|shell-.+)$/;

const TOKENS = [...rootVariables(shapeCss), ...rootVariables(motionCss)];
const CLASS_TOKENS = TOKENS.filter((name) => !NOT_CLASSES.test(name));
const THEME_FAMILIES =
  /^--(?:radius|shadow|transition-duration|ease|z-index|default-transition)/;

interface PostcssConfig {
  plugins: Record<string, Record<string, unknown>>;
}

async function compile(classes: string[]) {
  const require = createRequire(path.join(FRONTEND_ROOT, "package.json"));
  const { default: config }: { default: PostcssConfig } = require(
    path.join(FRONTEND_ROOT, "postcss.config.mjs"),
  );
  const plugins = Object.entries(config.plugins).map(
    ([name, options]) => require(name)(options) as AcceptedPlugin,
  );
  const from = path.join(FRONTEND_ROOT, "src/app/globals.css");
  const source = readFileSync(from, "utf8");
  const css = `${source}\n@source inline(${JSON.stringify(classes.join(" "))});\n`;
  return (await postcss(plugins).process(css, { from })).root;
}

function declarationsOf(root: Root, selector: string) {
  const found: Record<string, string> = {};
  root.walkRules((rule) => {
    if (!rule.selectors.includes(selector)) return;
    rule.walkDecls((decl) => {
      found[decl.prop] = decl.value.replace(/\s+/g, " ");
    });
  });
  return found;
}

describe("shape and motion tokens", () => {
  it("are imported by globals.css beside the other token files", () => {
    expect(globalsCss).toContain('@import "./tokens/shape.css";');
    expect(globalsCss).toContain('@import "./tokens/motion.css";');
  });

  it("register every variable on :root in globals.css's @theme", () => {
    expect(CLASS_TOKENS.length).toBeGreaterThan(0);
    const entries = new Set(
      themeEntries(globalsCss).map(([name, value]) => `${name}: ${value}`),
    );
    for (const name of CLASS_TOKENS) {
      const expected = registration(name);
      expect(expected, `${name} is no known token family`).not.toBeNull();
      expect(entries.has(expected!.entry), expected!.entry).toBe(true);
    }
  });

  it("read the variables that are not a class of their own", () => {
    expect(TOKENS).toEqual(
      expect.arrayContaining(["--shadow-color", "--smooth-ring-color"]),
    );
    for (const [name, value] of themeEntries(shapeCss))
      expect(value, name).toMatch(/var\(--(?:shadow-color|color-[a-z0-9-]+)\)/);
    expect(globalsCss).toContain("var(--smooth-ring-color)");
  });

  it("leave no radius, shadow, duration, easing or z-index value in globals.css's @theme", () => {
    const literal = themeEntries(globalsCss).filter(
      ([name, value]) =>
        THEME_FAMILIES.test(name) && !/^var\(--[a-z0-9-]+\)$/.test(value),
    );
    expect(literal).toEqual([]);
  });

  it("leave none in tailwind.config.ts", () => {
    for (const key of [
      "borderRadius",
      "boxShadow",
      "transitionDuration",
      "transitionTimingFunction",
      "zIndex",
    ]) {
      expect(tailwindConfig.theme, key).not.toHaveProperty(key);
      expect(tailwindConfig.theme.extend, key).not.toHaveProperty(key);
    }
  });

  it("keep the shadow steps in shape.css, coloured by --shadow-color or a role", () => {
    const shadows = themeEntries(shapeCss);
    expect(shadows.map(([name]) => name).sort()).toEqual(
      [
        "--shadow",
        "--shadow-sm",
        "--shadow-md",
        "--shadow-lg",
        "--shadow-xl",
        "--shadow-2xl",
        "--shadow-inner",
        "--shadow-subtle",
        "--shadow-message-table",
        "--shadow-lift",
        "--shadow-popover-raised",
        "--shadow-composer",
        "--shadow-drop",
        "--shadow-scroll-edge",
        "--shadow-keycap",
        "--shadow-gloss",
        "--shadow-sheen",
        "--shadow-outline",
        "--shadow-halo",
        "--shadow-spotlight",
        "--shadow-glow",
        "--shadow-status-glow",
        "--shadow-beacon",
        "--shadow-chip-glow-brand",
        "--shadow-chip-glow-warning",
        "--shadow-chip-glow-sky",
        "--shadow-chip-glow-neutral",
      ].sort(),
    );
    expect(colourRoles.size).toBeGreaterThan(0);
    for (const [name, value] of shadows) {
      const colours = value.match(
        /rgba?\([^()]*(?:\([^()]*\)[^()]*)*\)|#[0-9a-f]{3,8}/gi,
      );
      expect(colours?.length, name).toBeGreaterThan(0);
      for (const colour of colours ?? []) {
        const read = /^rgb\(var\((--[a-z0-9-]+)\) \/ [\d.]+\)$/.exec(colour);
        expect(read, `${name}: ${colour}`).not.toBeNull();
        const variable = read![1];
        expect(
          variable === "--shadow-color" || colourRoles.has(variable),
          `${name} reads ${variable}, which is neither --shadow-color nor a colour role`,
        ).toBe(true);
      }
    }
  });

  it("compile every shadow step to a class a shadow colour can recolour", async () => {
    const steps = themeEntries(shapeCss).map(([name]) =>
      name.replace(/^--shadow-?/, ""),
    );
    const classes = steps.map((step) => (step ? `shadow-${step}` : "shadow"));
    const root = await compile(classes);
    for (const className of classes) {
      const shadow = declarationsOf(root, `.${className}`)["--tw-shadow"];
      expect(shadow, className).toMatch(/var\(--tw-shadow-color, rgb\(var\(/);
    }
  });

  it("compile to classes that read them at runtime", async () => {
    const classes = CLASS_TOKENS.map((name) => registration(name)!.className);
    const root = await compile([
      ...classes,
      "z-10",
      "duration-200",
      "shadow-md",
      "shadow-scrim/10",
    ]);
    for (const [index, name] of CLASS_TOKENS.entries()) {
      const values = Object.values(declarationsOf(root, `.${classes[index]}`));
      expect(values.length, classes[index]).toBeGreaterThan(0);
      for (const value of values)
        expect(value, classes[index]).toBe(`var(${name})`);
    }
    expect(declarationsOf(root, ".z-10")).toEqual({ "z-index": "10" });
    expect(declarationsOf(root, ".duration-200")).toMatchObject({
      "transition-duration": "200ms",
    });
    // A coloured shadow still swaps the colour in: the value stays in the class.
    expect(declarationsOf(root, ".shadow-md")["--tw-shadow"]).toBe(
      "0 4px 6px -1px var(--tw-shadow-color, rgb(var(--shadow-color) / 0.1)), 0 2px 4px -2px var(--tw-shadow-color, rgb(var(--shadow-color) / 0.1))",
    );
    // The selector is built from the class name: Tailwind scans src, and the
    // escaped selector written out would add a class to the app's stylesheet.
    const coloured = `.${"shadow-scrim/10".replace("/", "\\/")}`;
    expect(declarationsOf(root, coloured)).toEqual({
      "--tw-shadow-color": "rgb(var(--color-scrim) / 0.1)",
    });
  });
});
