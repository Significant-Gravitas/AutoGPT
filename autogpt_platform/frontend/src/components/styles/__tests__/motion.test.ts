// @vitest-environment node
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { duration, ease, spring } from "../motion";

// motion.ts mirrors src/app/tokens/motion.css for Framer Motion, which cannot
// read CSS variables: the same names, in seconds and as cubic-bezier numbers.

const motionCss = readFileSync(
  path.resolve(__dirname, "../../../app/tokens/motion.css"),
  "utf8",
).replace(/\/\*[\s\S]*?\*\//g, "");

function camel(name: string) {
  return name.replace(/-([a-z])/g, (_, letter: string) => letter.toUpperCase());
}

function declared(prefix: string) {
  return Object.fromEntries(
    [
      ...motionCss.matchAll(
        new RegExp(`--${prefix}-([a-z-]+):\\s*([^;]+);`, "g"),
      ),
    ].map(([, name, value]) => [camel(name), value.trim()]),
  );
}

describe("motion.ts", () => {
  it("has every duration of motion.css, in seconds", () => {
    const css = declared("duration");
    expect(Object.keys(css).length).toBeGreaterThan(0);
    expect(Object.keys(duration).sort()).toEqual(Object.keys(css).sort());
    for (const [name, value] of Object.entries(css)) {
      const ms = /^(\d+)ms$/.exec(value);
      expect(ms, `--duration-${name}: ${value}`).not.toBeNull();
      expect(duration[name as keyof typeof duration], name).toBe(
        Number(ms![1]) / 1000,
      );
    }
  });

  it("has every easing of motion.css, as its cubic-bezier numbers", () => {
    const css = declared("ease");
    expect(Object.keys(css).length).toBeGreaterThan(0);
    expect(Object.keys(ease).sort()).toEqual(Object.keys(css).sort());
    for (const [name, value] of Object.entries(css)) {
      const curve = /^cubic-bezier\(([^)]+)\)$/.exec(value);
      expect(curve, `--ease-${name}: ${value}`).not.toBeNull();
      expect(ease[name as keyof typeof ease], name).toEqual(
        curve![1].split(",").map(Number),
      );
    }
  });

  it("steps the durations from instant to slowest", () => {
    expect(duration).toEqual({
      instant: 0.1,
      fast: 0.15,
      base: 0.2,
      slow: 0.3,
      slower: 0.4,
      slowest: 0.5,
    });
  });

  it("has the spring presets, which CSS has no equivalent of", () => {
    expect(spring).toEqual({
      snappy: { stiffness: 520, damping: 30 },
      gentle: { stiffness: 300, damping: 25 },
      bouncy: { stiffness: 200, damping: 15 },
    });
  });
});
