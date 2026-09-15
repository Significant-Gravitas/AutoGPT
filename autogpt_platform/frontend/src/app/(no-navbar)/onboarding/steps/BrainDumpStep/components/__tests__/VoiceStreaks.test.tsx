import { act, render } from "@testing-library/react";
import { motionValue } from "framer-motion";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createStreakStore, type StreakField } from "../streaks";
import { VoiceStreaks } from "../VoiceStreaks";
import { VoiceAura } from "../VoiceAura";
import { AUTOPILOT_AVATAR } from "@/components/molecules/BotAvatar/helpers";

const field: StreakField = {
  size: 160,
  pad: 40,
  box: 240,
  centre: { x: 80, y: 80 },
  profile: Array(72).fill(56),
  clearanceMin: 12,
  clearanceMax: 16,
  colors: ["#112233", "#445566"],
};
let frames: Map<number, FrameRequestCallback>;
let nextFrame: number;
let context: CanvasRenderingContext2D;

function tick(now: number) {
  act(() => {
    const pending = [...frames];
    frames.clear();
    for (const [, callback] of pending) callback(now);
  });
}

beforeEach(() => {
  frames = new Map();
  nextFrame = 0;
  context = {
    setTransform: vi.fn(),
    clearRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
  } as unknown as CanvasRenderingContext2D;
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context);
  vi.spyOn(performance, "now").mockReturnValue(100);
  vi.stubGlobal(
    "requestAnimationFrame",
    vi.fn((callback: FrameRequestCallback) => {
      frames.set(++nextFrame, callback);
      return nextFrame;
    }),
  );
  vi.stubGlobal(
    "cancelAnimationFrame",
    vi.fn((id: number) => frames.delete(id)),
  );
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("VoiceStreaks lifecycle", () => {
  it("keeps its current comets when a rerender supplies a new levels array", () => {
    const store = createStreakStore();
    const props = { field, store, layer: "front" as const, isActive: true };
    const { rerender, unmount } = render(
      <VoiceStreaks {...props} levels={[motionValue(0)]} />,
    );
    const initialPair = store.streaks;
    const loud = motionValue(1);
    const readLoud = vi.spyOn(loud, "get");
    rerender(<VoiceStreaks {...props} levels={[loud]} />);
    expect(store.streaks).toBe(initialPair);
    expect(store.nextId).toBe(2);
    expect(cancelAnimationFrame).not.toHaveBeenCalled();
    tick(200);
    tick(300);
    expect(readLoud).toHaveBeenCalled();
    expect(store.streaks.length).toBeGreaterThan(2);
    unmount();
    expect(frames.size).toBe(0);
  });

  it("draws both depth layers, sizes the canvas and clears it on deactivation", () => {
    const store = createStreakStore();
    const levels = [motionValue(0)];
    function Layers({ active }: { active: boolean }) {
      return (
        <>
          <VoiceStreaks
            field={field}
            levels={levels}
            store={store}
            layer="behind"
            isActive={active}
          />
          <VoiceStreaks
            field={field}
            levels={levels}
            store={store}
            layer="front"
            isActive={active}
          />
        </>
      );
    }
    const { container, rerender } = render(<Layers active />);
    expect(container.querySelector("canvas")?.width).toBe(
      240 * (window.devicePixelRatio || 1),
    );
    tick(100);
    tick(400);
    tick(800);
    tick(1200);
    expect(context.stroke).toHaveBeenCalled();
    expect(context.moveTo).toHaveBeenCalled();
    rerender(<Layers active={false} />);
    expect(store.streaks).toEqual([]);
    expect(frames.size).toBe(0);
    expect(context.clearRect).toHaveBeenCalledWith(0, 0, 240, 240);
  });

  it("accepts an empty level list and discards stale spawn budget on restart", () => {
    const store = createStreakStore();
    store.budget = 10000;
    const { rerender } = render(
      <VoiceStreaks
        field={field}
        levels={[]}
        store={store}
        layer="front"
        isActive
      />,
    );
    expect(store.budget).toBe(0);
    tick(200);
    tick(100000000);
    expect(Number.isFinite(store.budget)).toBe(true);
    rerender(
      <VoiceStreaks
        field={field}
        levels={[]}
        store={store}
        layer="front"
        isActive={false}
      />,
    );
    rerender(
      <VoiceStreaks
        field={field}
        levels={[]}
        store={store}
        layer="front"
        isActive
      />,
    );
    expect(store.streaks).toHaveLength(2);
  });

  it("does not schedule work when a canvas context is unavailable", () => {
    vi.mocked(HTMLCanvasElement.prototype.getContext).mockReturnValue(null);
    render(
      <VoiceStreaks
        field={field}
        levels={[]}
        store={createStreakStore()}
        layer="front"
        isActive
      />,
    );
    expect(frames.size).toBe(0);
  });
});

it("keeps the voice aura layers around their child and resizes with the avatar", () => {
  const levels = [motionValue(0.5)];
  const { container, rerender } = render(
    <VoiceAura config={AUTOPILOT_AVATAR} size={160} levels={levels} isActive>
      <span>Avatar</span>
    </VoiceAura>,
  );
  expect(container.querySelectorAll("canvas")).toHaveLength(2);
  rerender(
    <VoiceAura
      config={AUTOPILOT_AVATAR}
      size={160}
      levels={[...levels]}
      isActive
    >
      <span>Avatar</span>
    </VoiceAura>,
  );
  expect(container.textContent).toBe("Avatar");
  rerender(
    <VoiceAura
      config={{ ...AUTOPILOT_AVATAR, color: "coral" }}
      size={320}
      levels={levels}
      isActive
    >
      <span>Avatar</span>
    </VoiceAura>,
  );
  expect(container.querySelector("canvas")?.style.width).toBe("480px");
  tick(400);
});
