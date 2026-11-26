import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { cn } from "@/lib/utils";

// cn() must keep the classes tailwind-merge 2 kept: they still render as in
// Tailwind 3, so dropping one would change what an element looks like.
describe("cn() merges classes as tailwind-merge 2 did", () => {
  it.each([
    ["bg-surface-hover bg-gradient-to-b", "bg-surface-hover bg-gradient-to-b"],
    ["border-y border-2", "border-y border-2"],
    ["max-h-40 max-h-none", "max-h-40 max-h-none"],
    ["outline outline-2", "outline outline-2"],
    ["font-['Geist'] font-medium", "font-['Geist'] font-medium"],
    ["font-sans font-['Geist']", "font-['Geist']"],
    [
      "scrollbar-thumb-muted-foreground scrollbar-thumb-rounded-lg",
      "scrollbar-thumb-muted-foreground scrollbar-thumb-rounded-lg",
    ],
    ["scrollbar-thin scrollbar-none", "scrollbar-thin scrollbar-none"],
    [
      "text-body text-foreground-tertiary",
      "text-body text-foreground-tertiary",
    ],
    ["text-caption text-body", "text-body"],
    ["text-heading-md text-hero", "text-hero"],
    ["rounded-xl rounded-2xlarge", "rounded-2xlarge"],
    ["leading-6 leading-inherit", "leading-inherit"],
    ["tracking-tight tracking-eyebrow", "tracking-eyebrow"],
    ["p-2 p-4", "p-4"],
  ])("cn(%j) is %j", (input, expected) => {
    expect(cn(input)).toBe(expected);
  });
});

// Tailwind's own shadow steps, which tailwind-merge knows already, and the
// two it has always read as colours (see tw-merge.ts).
const UNREGISTERED_SHADOWS = new Set([
  "",
  "sm",
  "md",
  "lg",
  "xl",
  "2xl",
  "inner",
  "subtle",
  "message-table",
]);

function shapeShadowSteps() {
  const css = readFileSync(
    path.resolve(__dirname, "../../app/tokens/shape.css"),
    "utf8",
  ).replace(/\/\*[\s\S]*?\*\//g, "");
  const theme = /@theme[^{]*\{([^}]*)\}/.exec(css)?.[1] ?? "";
  return [...theme.matchAll(/--shadow(?:-([a-z0-9-]+))?:/g)].map(
    ([, step]) => step ?? "",
  );
}

describe("cn() knows the named shadow steps", () => {
  const named = shapeShadowSteps().filter(
    (step) => !UNREGISTERED_SHADOWS.has(step),
  );

  it("has steps to check", () => {
    expect(named).toContain("lift");
  });

  it.each(named)("shadow-%s replaces another shadow step", (step) => {
    expect(cn("shadow-sm", `shadow-${step}`)).toBe(`shadow-${step}`);
    expect(cn(`shadow-${step}`, "shadow-md")).toBe("shadow-md");
  });

  it.each(named)("shadow-%s survives next to a shadow colour", (step) => {
    expect(cn(`shadow-${step}`, "shadow-brand/45")).toBe(
      `shadow-${step} shadow-brand/45`,
    );
  });
});

function shapeLayers() {
  const css = readFileSync(
    path.resolve(__dirname, "../../app/tokens/shape.css"),
    "utf8",
  ).replace(/\/\*[\s\S]*?\*\//g, "");
  return [...css.matchAll(/--z-([a-z-]+):/g)].map(([, name]) => name);
}

describe("cn() knows the named layers", () => {
  const layers = shapeLayers();

  it("has layers to check", () => {
    expect(layers).toEqual(expect.arrayContaining(["raised", "overlay"]));
  });

  it("keeps the last of two layers", () => {
    expect(cn("z-raised", "z-overlay")).toBe("z-overlay");
    expect(cn("z-overlay", "z-raised")).toBe("z-raised");
  });

  it("lets a named layer and a number replace each other", () => {
    expect(cn("z-10", "z-overlay")).toBe("z-overlay");
    expect(cn("z-overlay", "z-10")).toBe("z-10");
    expect(cn("hover:z-raised", "hover:z-sticky")).toBe("hover:z-sticky");
  });

  it("keeps layers under different variants", () => {
    expect(cn("z-raised", "focus-visible:z-sticky")).toBe(
      "z-raised focus-visible:z-sticky",
    );
  });

  it.each(layers)("z-%s replaces another layer", (name) => {
    expect(cn("z-auto", `z-${name}`)).toBe(`z-${name}`);
  });
});
