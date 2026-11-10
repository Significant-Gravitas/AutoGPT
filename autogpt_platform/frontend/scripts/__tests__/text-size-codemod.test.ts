// @vitest-environment node
import { describe, expect, it } from "vitest";
import { twMerge } from "../../src/lib/tw-merge";
import { rewriteSizes, rewriteSource, TYPE_SCALE } from "../ds/text-size-scale";

describe("the text-size codemod", () => {
  it("keeps variant prefixes and !important", () => {
    expect(
      rewriteSizes("text-sm md:text-2xl !text-xs [&_svg]:text-lg").output,
    ).toBe("text-body md:text-heading-md !text-caption [&_svg]:text-lead");
  });

  it("leaves arbitrary sizes, scale steps and look-alikes alone", () => {
    const list =
      "text-[13px] text-body text-base-content context-sm text-[var(--c)]";
    expect(rewriteSizes(list)).toEqual({ output: list, changes: [] });
  });

  it("rewrites strings and template text, never comments", () => {
    const source = [
      "// text-sm stays in a comment",
      'const a = "text-sm font-medium";',
      "const b = cn(`md:text-lg ${extra}`, 'text-3xl');",
    ].join("\n");
    expect(rewriteSource(source, "file.tsx").output).toBe(
      [
        "// text-sm stays in a comment",
        'const a = "text-body font-medium";',
        "const b = cn(`md:text-lead ${extra}`, 'text-heading');",
      ].join("\n"),
    );
  });
});

describe("tw-merge and the type scale", () => {
  // tw-merge.ts lists the steps by hand; a step missing there would survive
  // next to another size in cn() and leave the winner to stylesheet order.
  it("reads every step as a font size, so two steps collapse to the last", () => {
    for (const step of TYPE_SCALE) {
      expect(twMerge(`text-caption text-${step.name}`)).toBe(
        `text-${step.name}`,
      );
      expect(twMerge(`text-${step.name} text-foreground-secondary`)).toBe(
        `text-${step.name} text-foreground-secondary`,
      );
    }
  });
});
