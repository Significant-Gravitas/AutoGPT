// @vitest-environment node
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import postcss, { type AcceptedPlugin, type Root } from "postcss";
import { describe, expect, it } from "vitest";
import { RAW_TEXT_SIZES, TYPE_SCALE } from "../ds/text-size-scale";

interface PostcssConfig {
  plugins: Record<string, Record<string, unknown>>;
}

// The app's real stylesheet and PostCSS steps, compiled for every step of the
// type scale and every Tailwind size it replaced (SECRT-3152).
const frontend = join(__dirname, "..", "..");
const require = createRequire(join(frontend, "package.json"));
const { default: postcssConfig }: { default: PostcssConfig } = require(
  join(frontend, "postcss.config.mjs"),
);
const plugins = Object.entries(postcssConfig.plugins).map(
  ([name, options]) => require(name)(options) as AcceptedPlugin,
);

const SCALE_CLASSES = TYPE_SCALE.map((step) => `text-${step.name}`);
const RAW_CLASSES = Object.keys(RAW_TEXT_SIZES);

let compiled: Promise<Root> | undefined;
function compile() {
  compiled ??= (async () => {
    const from = join(frontend, "src/app/globals.css");
    const classes = [...SCALE_CLASSES, ...RAW_CLASSES].join(" ");
    const css = `${readFileSync(from, "utf8")}\n@source inline(${JSON.stringify(classes)});\n`;
    return (await postcss(plugins).process(css, { from })).root;
  })();
  return compiled;
}

function declarations(root: Root, selector: string) {
  const out: Record<string, string> = {};
  root.walkRules((rule) => {
    if (!rule.selectors.includes(selector)) return;
    rule.walkDecls((decl) => {
      out[decl.prop] = decl.value;
    });
  });
  return out;
}

function rootValues() {
  const css = readFileSync(
    join(frontend, "src/app/tokens/typography.css"),
    "utf8",
  );
  const values = new Map<string, string>();
  for (const [, name, value] of css.matchAll(/(--text-[\w-]+):\s*([^;]+);/g))
    values.set(name, value.trim());
  return values;
}

describe("the type scale", () => {
  it("gives every step a utility that reads its token", async () => {
    const root = await compile();
    for (const step of TYPE_SCALE) {
      expect(declarations(root, `.text-${step.name}`)).toEqual({
        "font-size": `var(--text-${step.name})`,
        "line-height": `var(--text-${step.name}--line-height)`,
      });
    }
  });

  it("holds each step's size and line height in tokens/typography.css", () => {
    const values = rootValues();
    for (const step of TYPE_SCALE) {
      expect(values.get(`--text-${step.name}`)).toBe(step.size);
      expect(values.get(`--text-${step.name}--line-height`)).toBe(
        step.lineHeight,
      );
    }
    expect(values.size).toBe(TYPE_SCALE.length * 2);
  });

  it("generates nothing for Tailwind's own sizes", async () => {
    const root = await compile();
    // Plain names (`text-sm` ... `text-9xl`): nothing to escape in a selector.
    for (const raw of RAW_CLASSES) {
      expect(declarations(root, `.${raw}`)).toEqual({});
    }
  });

  it("sizes Streamdown's own Tailwind classes from the same steps", async () => {
    const root = await compile();
    for (const raw of [
      "text-xs",
      "text-sm",
      "text-base",
      "text-lg",
      "text-2xl",
    ]) {
      const step = RAW_TEXT_SIZES[raw].token.slice("text-".length);
      expect(
        declarations(
          root,
          `:where([data-streamdown], [data-streamdown] *).${raw}`,
        ),
      ).toEqual({
        "font-size": `var(--text-${step})`,
        "line-height": `var(--text-${step}--line-height)`,
      });
    }
  });

  it("maps every replaced size to a step of the scale", () => {
    const steps = new Set(SCALE_CLASSES);
    for (const target of Object.values(RAW_TEXT_SIZES))
      expect(steps.has(target.token)).toBe(true);
  });
});
