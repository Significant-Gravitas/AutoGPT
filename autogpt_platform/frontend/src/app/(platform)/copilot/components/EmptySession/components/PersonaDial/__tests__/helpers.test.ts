import { describe, expect, it } from "vitest";
import {
  angleFromCentre,
  clampRotation,
  DIAL_STEP,
  indexFromRotation,
  nearestVirtual,
  rotationForVirtual,
  shouldWrap,
  snapRotation,
  stepsFromBottom,
  virtualFromRotation,
  wrapIndex,
} from "../helpers";

describe("wrapIndex", () => {
  it("maps virtual slots onto the roster in both directions", () => {
    expect(wrapIndex(0, 5)).toBe(0);
    expect(wrapIndex(7, 5)).toBe(2);
    expect(wrapIndex(-1, 5)).toBe(4);
    expect(wrapIndex(-6, 5)).toBe(4);
  });
});

describe("virtualFromRotation / rotationForVirtual", () => {
  it("round-trips a slot through its rotation", () => {
    for (const virtual of [-7, -1, 0, 3, 12]) {
      expect(virtualFromRotation(rotationForVirtual(virtual))).toBe(virtual);
    }
  });

  it("rounds to the nearest slot mid-drag", () => {
    expect(virtualFromRotation(-DIAL_STEP * 1.4)).toBe(1);
    expect(virtualFromRotation(-DIAL_STEP * 1.6)).toBe(2);
  });
});

describe("indexFromRotation", () => {
  it("wraps the bottom slot onto the roster", () => {
    expect(indexFromRotation(rotationForVirtual(0), 10)).toBe(0);
    expect(indexFromRotation(rotationForVirtual(13), 10)).toBe(3);
    expect(indexFromRotation(rotationForVirtual(-2), 10)).toBe(8);
  });
});

describe("nearestVirtual", () => {
  it("picks the equivalent slot closest to the current rotation", () => {
    // Sitting at virtual 9 of 10, index 0 is nearer as virtual 10 than 0.
    expect(nearestVirtual(0, 10, rotationForVirtual(9))).toBe(10);
    expect(nearestVirtual(8, 10, rotationForVirtual(9))).toBe(8);
    expect(nearestVirtual(0, 10, rotationForVirtual(1))).toBe(0);
  });
});

describe("snapRotation", () => {
  it("snaps to the nearest slot boundary", () => {
    expect(snapRotation(-DIAL_STEP * 2.4)).toBe(-DIAL_STEP * 2);
    expect(snapRotation(-DIAL_STEP * 2.6)).toBe(-DIAL_STEP * 3);
    expect(snapRotation(0)).toBe(0);
  });
});

describe("clampRotation", () => {
  it("bounds rotation to the arc's first and last slots", () => {
    expect(clampRotation(50, 4)).toBe(0);
    expect(clampRotation(-DIAL_STEP * 99, 4)).toBe(-DIAL_STEP * 3);
    expect(clampRotation(-DIAL_STEP, 4)).toBe(-DIAL_STEP);
  });
});

describe("shouldWrap", () => {
  it("wraps only when the roster exceeds the render window", () => {
    expect(shouldWrap(11)).toBe(true);
    expect(shouldWrap(10)).toBe(false);
    expect(shouldWrap(1)).toBe(false);
  });
});

describe("stepsFromBottom", () => {
  it("measures slot distance from the selection point", () => {
    expect(stepsFromBottom(3, rotationForVirtual(3))).toBe(0);
    expect(stepsFromBottom(5, rotationForVirtual(3))).toBe(2);
    expect(stepsFromBottom(1, rotationForVirtual(3))).toBe(2);
  });
});

describe("angleFromCentre", () => {
  it("returns the pointer's angle around the centre in degrees", () => {
    const centre = { x: 0, y: 0 };
    expect(angleFromCentre(centre, { x: 10, y: 0 })).toBe(0);
    expect(angleFromCentre(centre, { x: 0, y: 10 })).toBe(90);
    expect(angleFromCentre(centre, { x: -10, y: 0 })).toBe(180);
  });
});
