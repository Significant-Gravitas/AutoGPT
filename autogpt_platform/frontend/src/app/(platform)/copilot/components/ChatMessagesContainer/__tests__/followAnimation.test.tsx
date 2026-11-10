import { act, cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  FOLLOW_SPRING,
  LANDING_GRACE_MS,
  REFLOW_HOLD_MS,
  useFollowAnimation,
} from "../useFollowAnimation";

const motion = vi.hoisted(() => ({ reduce: false }));
vi.mock("framer-motion", async (importActual) => ({
  ...(await importActual<typeof import("framer-motion")>()),
  useReducedMotion: () => motion.reduce,
}));

let resizeWidth: (width: number) => void = () => {};

class FakeResizeObserver {
  constructor(private callback: ResizeObserverCallback) {}
  observe() {
    resizeWidth = (width) =>
      this.callback(
        [{ contentRect: { width } } as ResizeObserverEntry],
        this as unknown as ResizeObserver,
      );
  }
  unobserve() {}
  disconnect() {}
}

const renders: ReturnType<typeof useFollowAnimation>["resize"][] = [];

function Probe(props: { settled: boolean; sessionID: string }) {
  const { resize, widthRef } = useFollowAnimation(props);
  renders.push(resize);
  return <div ref={widthRef} />;
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);
  motion.reduce = false;
  renders.length = 0;
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("how the thread follows its growth", () => {
  it("lands instantly, then follows with a spring once the thread has landed", () => {
    const { rerender } = render(<Probe settled={false} sessionID="a" />);
    expect(renders.at(-1)).toBe("instant");
    rerender(<Probe settled sessionID="a" />);
    expect(renders.at(-1)).toBe("instant");
    act(() => vi.advanceTimersByTime(LANDING_GRACE_MS));
    expect(renders.at(-1)).toEqual(FOLLOW_SPRING);
  });

  it("lands a thread it switches to instantly again", () => {
    const { rerender } = render(<Probe settled sessionID="a" />);
    act(() => vi.advanceTimersByTime(LANDING_GRACE_MS));
    expect(renders.at(-1)).toEqual(FOLLOW_SPRING);
    rerender(<Probe settled={false} sessionID="b" />);
    expect(renders.at(-1)).toBe("instant");
    rerender(<Probe settled sessionID="b" />);
    act(() => vi.advanceTimersByTime(LANDING_GRACE_MS - 1));
    expect(renders.at(-1)).toBe("instant");
    act(() => vi.advanceTimersByTime(1));
    expect(renders.at(-1)).toEqual(FOLLOW_SPRING);
  });

  it("never animates for a reader who prefers reduced motion", () => {
    motion.reduce = true;
    render(<Probe settled sessionID="a" />);
    act(() => vi.advanceTimersByTime(LANDING_GRACE_MS * 5));
    expect(renders.at(-1)).toBe("instant");
  });

  it("stays pinned while the thread reflows to a new width", () => {
    render(<Probe settled sessionID="a" />);
    act(() => vi.advanceTimersByTime(LANDING_GRACE_MS));
    act(() => resizeWidth(800));
    expect(renders.at(-1)).toEqual(FOLLOW_SPRING);
    act(() => resizeWidth(600));
    expect(renders.at(-1)).toBe("instant");
    act(() => vi.advanceTimersByTime(REFLOW_HOLD_MS / 2));
    act(() => resizeWidth(560));
    act(() => vi.advanceTimersByTime(REFLOW_HOLD_MS - 1));
    expect(renders.at(-1)).toBe("instant");
    act(() => vi.advanceTimersByTime(1));
    expect(renders.at(-1)).toEqual(FOLLOW_SPRING);
  });
});
