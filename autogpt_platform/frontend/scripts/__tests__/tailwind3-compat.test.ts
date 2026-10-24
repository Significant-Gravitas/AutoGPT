// @vitest-environment node
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import postcss, { type AcceptedPlugin, type Root, type Rule } from "postcss";
import { describe, expect, it } from "vitest";

interface PostcssConfig {
  plugins: Record<string, Record<string, unknown>>;
}

// The app's real stylesheet and PostCSS steps (postcss.config.mjs), compiled
// for a fixed set of classes: what each one renders must be what Tailwind 3
// rendered.
const frontend = join(__dirname, "..", "..");
const require = createRequire(join(frontend, "package.json"));
const { default: postcssConfig }: { default: PostcssConfig } = require(
  join(frontend, "postcss.config.mjs"),
);
const plugins = Object.entries(postcssConfig.plugins).map(
  ([name, options]) => require(name)(options) as AcceptedPlugin,
);

const CLASSES = [
  "space-y-4",
  "-space-x-1.5",
  "sm:space-x-4",
  "divide-y",
  "divide-zinc-100",
  "outline-none",
  "focus-visible:outline-none",
  "outline",
  "outline-2",
  "-translate-x-1/2",
  "rotate-180",
  "scale-95",
  "transform",
  "bg-gradient-to-r",
  "shadow-sm",
  "shadow",
  "rounded",
  "rounded-sm",
  "blur-sm",
  "ring",
  "md:flex",
  "bg-gray-400",
  "w-(--sidebar-width)",
  "bg-surface/50",
  "bg-zinc-500/50",
  "shadow-black/10",
  "transition",
  "transition-none",
  "justify-self-start",
  "sr-only",
  "border",
  "file:border-0",
  "active:scale-[0.92]",
  "scrollbar-thin",
  "scrollbar-thumb-border-base",
  "hover:scrollbar-thumb-zinc-300",
  "hover:bg-primary/90",
  "hover:text-zinc-800",
  "aria-pressed:text-zinc-900",
  "focus-visible:outline-none",
  "focus-visible:-outline-offset-2",
  "text-lg",
  "leading-relaxed",
  // Tailwind 3 generated nothing for these.
  "size-13",
  "shadow-xs",
  "z-12",
  "disabled:opacity-1",
];

async function compile() {
  const from = join(frontend, "src/app/globals.css");
  const css = `${readFileSync(from, "utf8")}\n@source inline(${JSON.stringify(CLASSES.join(" "))});\n`;
  const result = await postcss(plugins).process(css, { from });
  return result.root;
}

function rulesFor(root: Root, selector: string) {
  const rules: Rule[] = [];
  root.walkRules((rule) => {
    if (rule.selectors.includes(selector)) rules.push(rule);
  });
  return rules;
}

function declarations(root: Root, selector: string) {
  const out: Record<string, string> = {};
  for (const rule of rulesFor(root, selector)) {
    rule.walkDecls((decl) => {
      out[decl.prop] = decl.value;
    });
  }
  return out;
}

const TRANSFORM =
  "translate(var(--tw-translate-x), var(--tw-translate-y)) rotate(var(--tw-rotate)) skewX(var(--tw-skew-x, 0)) skewY(var(--tw-skew-y, 0)) scaleX(var(--tw-scale-x)) scaleY(var(--tw-scale-y))";

describe("Tailwind 4 renders the app's classes as Tailwind 3 did", () => {
  it("spaces and divides children with Tailwind 3's selectors", async () => {
    const root = await compile();
    expect(
      declarations(root, ".space-y-4 > :not([hidden]) ~ :not([hidden])"),
    ).toEqual({
      "--tw-space-y-reverse": "0",
      "margin-top": "calc(1rem * calc(1 - var(--tw-space-y-reverse)))",
      "margin-bottom": "calc(1rem * var(--tw-space-y-reverse))",
    });
    expect(
      declarations(root, ".divide-y > :not([hidden]) ~ :not([hidden])"),
    ).toEqual({
      "--tw-divide-y-reverse": "0",
      "border-top-width": "calc(1px * calc(1 - var(--tw-divide-y-reverse)))",
      "border-bottom-width": "calc(1px * var(--tw-divide-y-reverse))",
    });
    expect(
      declarations(root, ".divide-zinc-100 > :not([hidden]) ~ :not([hidden])"),
    ).toEqual({ "border-color": "#EFEFF0" });
    expect(
      rulesFor(root, ".sm\\:space-x-4 > :not([hidden]) ~ :not([hidden])"),
    ).toHaveLength(1);
    expect(root.toString()).not.toMatch(
      /:where\([^)]* > :not\(:last-child\)\)/,
    );
  });

  it("keeps Tailwind 3's outline classes", async () => {
    const root = await compile();
    expect(declarations(root, ".outline-none")).toEqual({
      outline: "2px solid transparent",
      "outline-offset": "2px",
    });
    expect(
      declarations(root, ".focus-visible\\:outline-none:focus-visible"),
    ).toEqual({ outline: "2px solid transparent", "outline-offset": "2px" });
    expect(declarations(root, ".outline")).toEqual({
      "outline-style": "solid",
    });
    expect(declarations(root, ".outline-2")).toEqual({
      "outline-width": "2px",
    });
  });

  it("moves elements with one `transform`, as Tailwind 3 did", async () => {
    const root = await compile();
    expect(declarations(root, ".-translate-x-1\\/2")).toEqual({
      "--tw-translate-x": "calc(50% * -1)",
      transform: TRANSFORM,
    });
    expect(declarations(root, ".rotate-180")).toEqual({
      "--tw-rotate": "180deg",
      transform: TRANSFORM,
    });
    expect(declarations(root, ".scale-95")).toEqual({
      "--tw-scale-x": ".95",
      "--tw-scale-y": ".95",
      transform: TRANSFORM,
    });
    expect(declarations(root, ".transform")).toEqual({ transform: TRANSFORM });
  });

  it("blends `bg-gradient-to-*` in sRGB", async () => {
    const root = await compile();
    expect(declarations(root, ".bg-gradient-to-r")).toMatchObject({
      "--tw-gradient-position": "to right",
    });
  });

  it("keeps Tailwind 3's theme values and breakpoints", async () => {
    const root = await compile();
    expect(declarations(root, ".shadow-sm")["--tw-shadow"]).toBe(
      "0 1px 2px 0 var(--tw-shadow-color, rgb(0 0 0 / 0.05))",
    );
    expect(declarations(root, ".rounded")).toEqual({
      "border-radius": "0.25rem",
    });
    expect(declarations(root, ".rounded-sm")).toEqual({
      "border-radius": "calc(var(--radius) - 4px)",
    });
    expect(declarations(root, ".blur-sm")["--tw-blur"]).toBe("blur(4px)");
    expect(declarations(root, ".bg-gray-400")).toEqual({
      "background-color": "#9ca3af",
    });
    expect(declarations(root, ".ring")).toMatchObject({
      "--tw-ring-offset-shadow":
        "var(--tw-ring-inset,) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color)",
      "box-shadow":
        "var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow)",
    });
    expect(declarations(root, ".shadow-sm")["box-shadow"]).toBe(
      "var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow)",
    );
    expect(declarations(root, ".ring")["--tw-ring-shadow"]).toContain(
      "calc(3px + var(--tw-ring-offset-width)) var(--tw-ring-color, rgb(59 130 246 / 0.5))",
    );
    expect(declarations(root, ".w-\\(--sidebar-width\\)")).toEqual({
      width: "var(--sidebar-width)",
    });
    let mdMedia = "";
    root.walkAtRules("media", (atRule) => {
      if (atRule.toString().includes(".md\\:flex")) mdMedia = atRule.params;
    });
    expect(mdMedia).toBe("(width >= 768px)");
  });

  it("writes colours with an alpha, not an oklab mix", async () => {
    const root = await compile();
    expect(declarations(root, ".bg-surface\\/50")).toEqual({
      "background-color": "rgb(var(--color-surface) / 0.5)",
    });
    expect(declarations(root, ".bg-zinc-500\\/50")).toEqual({
      "background-color": "rgb(131 131 140 / 0.5)",
    });
    expect(declarations(root, ".shadow-black\\/10")).toEqual({
      "--tw-shadow-color": "rgb(20 20 20 / 0.1)",
    });
    expect(root.toString()).not.toContain("in oklab");
  });

  it("keeps Tailwind 3's transitions, alignment and hidden text", async () => {
    const root = await compile();
    expect(declarations(root, ".transition")["transition-property"]).toBe(
      "color, background-color, border-color, text-decoration-color, fill, stroke, opacity, box-shadow, transform, filter, backdrop-filter",
    );
    expect(declarations(root, ".transition-none")).toEqual({
      "transition-property": "none",
    });
    expect(declarations(root, ".justify-self-start")).toEqual({
      "justify-self": "start",
    });
    expect(declarations(root, ".sr-only").clip).toBe("rect(0, 0, 0, 0)");
    expect(declarations(root, ".sr-only")["clip-path"]).toBeUndefined();
    expect(declarations(root, ".border")).toEqual({ "border-width": "1px" });
    expect(
      declarations(root, ".file\\:border-0::file-selector-button"),
    ).toEqual({ "border-width": "0px" });
    expect(declarations(root, ".active\\:scale-\\[0\\.92\\]:active")).toEqual({
      "--tw-scale-x": "0.92",
      "--tw-scale-y": "0.92",
      transform: TRANSFORM,
    });
  });

  it("styles scrollbars as tailwind-scrollbar 3 did", async () => {
    const root = await compile();
    expect(declarations(root, ".scrollbar-thin")).toMatchObject({
      "scrollbar-width": "thin",
      "scrollbar-color":
        "var(--scrollbar-thumb, initial) var(--scrollbar-track, initial)",
    });
    expect(declarations(root, ".scrollbar-thumb-border-base")).toEqual({
      "--scrollbar-thumb": "rgb(var(--color-border-base) / 1) !important",
    });
    expect(declarations(root, ".hover\\:scrollbar-thumb-zinc-300")).toEqual({
      "--scrollbar-thumb-hover": "#C5C5C9 !important",
    });
    expect(
      rulesFor(root, ".scrollbar-thin::-webkit-scrollbar-thumb:hover"),
    ).toHaveLength(1);
    expect(root.toString()).not.toContain("var(--tw-scrollbar-thumb)");
  });

  it("orders rules as Tailwind 3 did, so the same class wins a conflict", async () => {
    const root = await compile();
    const order: string[] = [];
    root.walkRules((rule) => {
      order.push(rule.selector);
    });
    const at = (selector: string) => order.indexOf(selector);
    expect(at(".hover\\:text-zinc-800:hover")).toBeGreaterThan(
      at('.aria-pressed\\:text-zinc-900[aria-pressed="true"]'),
    );
    expect(at(".focus-visible\\:outline-none:focus-visible")).toBeLessThan(
      at(".focus-visible\\:-outline-offset-2:focus-visible"),
    );
    expect(declarations(root, ".text-lg")["line-height"]).toBe("1.75rem");
  });

  it("generates nothing for classes Tailwind 3 had no value for", async () => {
    const root = await compile();
    for (const selector of [
      ".size-13",
      ".shadow-xs",
      ".z-12",
      ".disabled\\:opacity-1:disabled",
    ]) {
      expect(rulesFor(root, selector)).toEqual([]);
    }
  });
});
