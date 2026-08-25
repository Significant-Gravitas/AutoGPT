import { describe, expect, it } from "vitest";
import { asymptoticProgress, finishProgress } from "../useCompactionProgress";

describe("asymptoticProgress", () => {
  it("starts at zero and rises quickly at first", () => {
    expect(asymptoticProgress(0)).toBe(0);
    expect(asymptoticProgress(5_000)).toBeGreaterThan(0.2);
  });

  it("keeps slowing and never exceeds its cap", () => {
    const atOneMinute = asymptoticProgress(60_000);
    expect(atOneMinute).toBeLessThan(0.92);
    // Float saturation: far out the curve rounds to the cap, never past it.
    expect(asymptoticProgress(600_000)).toBeLessThanOrEqual(0.92);
  });

  it("is monotonic", () => {
    let prev = -1;
    for (let ms = 0; ms <= 120_000; ms += 1_000) {
      const value = asymptoticProgress(ms);
      expect(value).toBeGreaterThan(prev);
      prev = value;
    }
  });
});

describe("finishProgress", () => {
  it("sprints from the current value to exactly 1", () => {
    expect(finishProgress(0.5, 0)).toBeCloseTo(0.5, 5);
    expect(finishProgress(0.5, 200)).toBeGreaterThan(0.9);
    expect(finishProgress(0.5, 1_000)).toBe(1);
  });

  it("snaps to 1 near the end instead of crawling forever", () => {
    expect(finishProgress(0.99, 100)).toBe(1);
  });
});
