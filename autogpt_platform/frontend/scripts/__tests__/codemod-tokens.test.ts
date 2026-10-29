// @vitest-environment node
import { describe, expect, it } from "vitest";

import {
  buildReplacementMap,
  buildSizeMap,
  rewriteClassList,
} from "../codemod-tokens";

function rewrite(list: string, options = { snap: true, flagged: false }) {
  return rewriteClassList(
    list,
    buildReplacementMap(options),
    buildSizeMap(options),
  );
}

describe("codemod-tokens snap tables", () => {
  it("leaves snapped rows out of the exact map", () => {
    const exact = { snap: false, flagged: false };
    expect(rewrite("text-gray-500 text-[9px]", exact).output).toBe(
      "text-gray-500 text-[9px]",
    );
  });

  it("snaps a grey to the role for how the class uses it", () => {
    expect(
      rewrite("text-gray-400 border-gray-400 bg-gray-100 border-t-gray-200")
        .output,
    ).toBe(
      "text-foreground-subtle border-border-strong bg-surface-hover border-t-border-base",
    );
  });

  it("keeps variants, important and opacity", () => {
    expect(
      rewrite(
        "hover:text-neutral-600/70 md:!bg-red-600 placeholder:text-gray-500",
      ).output,
    ).toBe(
      "hover:text-foreground-secondary/70 md:!bg-error-hover placeholder:text-placeholder",
    );
  });

  it("leaves a use the row has no role for", () => {
    expect(rewrite("bg-gray-500 text-gray-200").output).toBe(
      "bg-gray-500 text-gray-200",
    );
  });

  it("skips an opacity on a token that carries its own", () => {
    const { output, changes } = rewrite("bg-red-200 bg-red-200/50");
    expect(output).toBe("bg-error/20 bg-red-200/50");
    expect(changes[1].skipped).toMatch(/own opacity/);
  });

  it("runs the flagged rows only when asked", () => {
    expect(rewrite("text-yellow-700 ring-green-200").output).toBe(
      "text-yellow-700 ring-green-200",
    );
    expect(
      rewrite("text-yellow-700 ring-green-200", { snap: true, flagged: true })
        .output,
    ).toBe("text-warning-foreground ring-success/40");
  });

  it("snaps raw sizes and keeps the parent's line height", () => {
    expect(rewrite("text-[9px] font-medium").output).toBe(
      "text-tiny leading-inherit font-medium",
    );
    expect(rewrite("text-[36px] leading-[50px]").output).toBe(
      "text-title leading-12",
    );
  });

  it("snaps tracking and leading to the nearest step", () => {
    expect(rewrite("tracking-[0.14em] leading-[1.35]").output).toBe(
      "tracking-caps leading-snug",
    );
  });
});
