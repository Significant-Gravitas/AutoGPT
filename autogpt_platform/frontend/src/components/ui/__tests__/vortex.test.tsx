import { render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { Vortex } from "../vortex";

type ObserverCallback = (entries: Array<{ isIntersecting: boolean }>) => void;

const rafCallbacks = new Map<number, FrameRequestCallback>();
let rafId = 0;
const intersectionCallbacks: ObserverCallback[] = [];
const resizeCallbacks: Array<() => void> = [];
const disconnectSpy = vi.fn();
let mockSize = { width: 800, height: 600 };
let mockCtx: ReturnType<typeof createMockContext>;

function createMockContext() {
  return {
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    fillStyle: "",
    strokeStyle: "",
    lineCap: "",
    lineWidth: 0,
    globalCompositeOperation: "",
  };
}

function flushFrames(count = 1) {
  for (let i = 0; i < count; i++) {
    const pending = Array.from(rafCallbacks.values());
    rafCallbacks.clear();
    for (const callback of pending) callback(performance.now());
  }
}

function setReducedMotion(matches: boolean) {
  vi.stubGlobal("matchMedia", vi.fn().mockReturnValue({ matches }));
}

beforeEach(() => {
  rafCallbacks.clear();
  rafId = 0;
  intersectionCallbacks.length = 0;
  resizeCallbacks.length = 0;
  disconnectSpy.mockClear();
  mockSize = { width: 800, height: 600 };
  mockCtx = createMockContext();

  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    rafCallbacks.set(++rafId, cb);
    return rafId;
  });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => {
    rafCallbacks.delete(id);
  });
  vi.stubGlobal(
    "IntersectionObserver",
    class {
      constructor(cb: ObserverCallback) {
        intersectionCallbacks.push(cb);
      }
      observe() {}
      disconnect = disconnectSpy;
    },
  );
  vi.stubGlobal(
    "ResizeObserver",
    class {
      constructor(cb: () => void) {
        resizeCallbacks.push(cb);
      }
      observe() {
        for (const cb of resizeCallbacks) cb();
      }
      disconnect = disconnectSpy;
    },
  );
  setReducedMotion(false);

  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
    mockCtx as unknown as CanvasRenderingContext2D,
  );
  Object.defineProperty(HTMLElement.prototype, "clientWidth", {
    configurable: true,
    get: () => mockSize.width,
  });
  Object.defineProperty(HTMLElement.prototype, "clientHeight", {
    configurable: true,
    get: () => mockSize.height,
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  Reflect.deleteProperty(HTMLElement.prototype, "clientWidth");
  Reflect.deleteProperty(HTMLElement.prototype, "clientHeight");
});

describe("Vortex", () => {
  test("animates while visible and stops when hidden", () => {
    render(<Vortex particleCount={10} />);

    const [onIntersection] = intersectionCallbacks;
    onIntersection([{ isIntersecting: true }]);
    flushFrames(2);
    expect(mockCtx.stroke).toHaveBeenCalled();

    onIntersection([{ isIntersecting: false }]);
    expect(rafCallbacks.size).toBe(0);
  });

  test("acts on the last entry when intersection records are batched", () => {
    render(<Vortex particleCount={10} />);

    intersectionCallbacks[0]([
      { isIntersecting: true },
      { isIntersecting: false },
    ]);
    expect(rafCallbacks.size).toBe(0);
    expect(mockCtx.stroke).not.toHaveBeenCalled();
  });

  test("renders a single static frame under prefers-reduced-motion", () => {
    setReducedMotion(true);
    render(<Vortex particleCount={10} />);

    expect(intersectionCallbacks.length).toBe(0);
    flushFrames(1);
    expect(mockCtx.stroke).toHaveBeenCalled();
    expect(rafCallbacks.size).toBe(0);
  });

  test("sizes the canvas to its container and follows resizes", () => {
    mockSize = { width: 0, height: 0 };
    const { container } = render(<Vortex particleCount={10} />);
    const canvas = container.querySelector("canvas");

    expect(canvas?.width).toBe(0);

    mockSize = { width: 640, height: 480 };
    for (const cb of resizeCallbacks) cb();
    expect(canvas?.width).toBe(640);
    expect(canvas?.height).toBe(480);
  });

  test("disconnects observers and cancels frames on unmount", () => {
    const { unmount } = render(<Vortex particleCount={10} />);
    intersectionCallbacks[0]([{ isIntersecting: true }]);

    unmount();
    expect(disconnectSpy).toHaveBeenCalledTimes(2);
    expect(rafCallbacks.size).toBe(0);
  });
});
