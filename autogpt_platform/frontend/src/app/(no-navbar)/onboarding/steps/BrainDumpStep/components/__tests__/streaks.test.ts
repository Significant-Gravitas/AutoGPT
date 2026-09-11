import { afterEach, describe, expect, it, vi } from "vitest";
import {
  AUTOPILOT_AVATAR,
  SHAPES,
} from "@/components/molecules/BotAvatar/helpers";
import {
  advanceStreaks,
  bodyRadiusAt,
  createStreakField,
  createStreakStore,
  envelope,
  headAngle,
  hexToRgb,
  makeStreakPair,
  MAX_STREAKS,
  measureBodyProfile,
  orbitPoint,
  project,
  smoothProfile,
  spawnRate,
  tailFor,
  tailSpan,
} from "../streaks";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("streak geometry", () => {
  it.each(SHAPES)(
    "fits a finite orbit around $id with a canvas-free fallback",
    (shape) => {
      vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
      const field = createStreakField(
        { ...AUTOPILOT_AVATAR, shape: shape.id },
        160,
      );
      expect(field.box).toBe(240);
      expect(field.profile).toHaveLength(72);
      expect(
        field.profile.every((radius) => Number.isFinite(radius) && radius > 0),
      ).toBe(true);
      const [first, second] = makeStreakPair(field, 7, 0.8, 100, () => 0.25);
      expect(first.id).toBe(7);
      expect(second.id).toBe(8);
      expect(first.sweep).toBe(-second.sweep);
      expect(first.tilt).toBe(-second.tilt);
      expect(first.bornAt).toBe(second.bornAt);
      for (const streak of [first, second]) {
        for (const angle of [0, Math.PI / 2, Math.PI, -Math.PI / 2]) {
          const point = orbitPoint(streak, angle);
          expect(Math.hypot(point.x, point.y, point.z)).toBeCloseTo(1);
          const projected = project(field, streak, point);
          expect(projected.scale).toBeGreaterThanOrEqual(0.75);
          expect(projected.scale).toBeLessThanOrEqual(1.25);
          expect(Number.isFinite(projected.x + projected.y)).toBe(true);
        }
      }
      expect(bodyRadiusAt(field, 0)).toBeCloseTo(
        bodyRadiusAt(field, Math.PI * 2),
      );
      expect(bodyRadiusAt(field, -0.2)).toBeCloseTo(
        bodyRadiusAt(field, Math.PI * 2 - 0.2),
      );
    },
  );

  it("measures a path against its canvas outline", () => {
    vi.stubGlobal("Path2D", class {});
    const isPointInPath = vi.fn(
      (_path: unknown, x: number, y: number) =>
        Math.hypot(x - 60, y - 60) <= 30,
    );
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      isPointInPath,
    } as unknown as CanvasRenderingContext2D);
    const profile = measureBodyProfile(
      "M0 0",
      { x: 60, y: 60 },
      { width: 80, top: 20, bottom: 100 },
    );
    expect(isPointInPath).toHaveBeenCalled();
    for (const radius of profile) expect(radius).toBeCloseTo(30, 3);
    expect(smoothProfile(Array(72).fill(30))).toEqual(Array(72).fill(30));
  });

  it("grows and fades the tail smoothly over a complete lap", () => {
    const field = createStreakField(AUTOPILOT_AVATAR, 160);
    const [streak] = makeStreakPair(field, 0, 1, 0, () => 0.75);
    expect(headAngle(streak, 0)).toBe(streak.start);
    expect(headAngle(streak, 1)).toBeCloseTo(streak.start + streak.sweep);
    expect(headAngle(streak, 0.5)).toBeCloseTo(streak.start + streak.sweep / 2);
    expect(tailSpan(streak, 0)).toBe(0);
    expect(tailSpan(streak, 1)).toBe(0);
    expect(tailSpan(streak, 0.09)).toBeCloseTo(streak.tailAngle / 2);
    expect(tailSpan(streak, 0.5)).toBe(streak.tailAngle);
    expect(tailSpan(streak, 0.9)).toBeCloseTo(streak.tailAngle / 2);
    expect([-0.1, 0, 1, 1.1].map(envelope)).toEqual([0, 0, 0, 0]);
    expect(envelope(0.04)).toBeCloseTo(0.5);
    expect(envelope(0.5)).toBe(1);
    expect(envelope(0.96)).toBeCloseTo(0.5);
    expect(tailFor(0)).toBeLessThan(tailFor(0.5));
    expect(tailFor(2)).toBe(tailFor(1));
    expect(spawnRate(0)).toBeLessThan(spawnRate(0.5));
    expect(spawnRate(2)).toBe(spawnRate(1));
    expect(hexToRgb("#12abef")).toEqual([18, 171, 239]);
  });
});

describe("streak spawning", () => {
  it("bounds work after a long background pause and reclaims expired slots first", () => {
    const field = createStreakField(AUTOPILOT_AVATAR, 160);
    const store = createStreakStore();
    for (let id = 0; id < MAX_STREAKS; id += 2)
      store.streaks.push(...makeStreakPair(field, id, 1, 1));
    store.nextId = MAX_STREAKS;
    store.lastTick = 1;
    advanceStreaks(store, field, 1, 100000000);
    expect(store.streaks).toHaveLength(2);
    expect(store.streaks.every((streak) => streak.bornAt === 100000000)).toBe(
      true,
    );
    expect(store.nextId).toBe(MAX_STREAKS + 2);
    expect(store.budget).toBeLessThan(1);
  });

  it("drops excess demand when full, never exceeds capacity, and handles backward time", () => {
    const field = createStreakField(AUTOPILOT_AVATAR, 160);
    const store = createStreakStore();
    store.budget = 1e12;
    advanceStreaks(store, field, 1, 1000);
    expect(store.streaks).toHaveLength(MAX_STREAKS);
    const nextId = store.nextId;
    for (let now = 1010; now <= 2000; now += 10)
      advanceStreaks(store, field, 1, now);
    expect(store.nextId).toBe(nextId);
    expect(store.budget).toBeLessThan(1);
    advanceStreaks(store, field, 1, 1900);
    expect(store.budget).toBeGreaterThanOrEqual(0);
    expect(store.streaks).toHaveLength(MAX_STREAKS);
  });
});
