import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useAudioBars } from "../useAudioBars";

const { reducedMotion } = vi.hoisted(() => ({
  reducedMotion: { value: false },
}));

vi.mock("framer-motion", async () => {
  const actual =
    await vi.importActual<typeof import("framer-motion")>("framer-motion");
  return {
    ...actual,
    useReducedMotion: () => reducedMotion.value,
  };
});

let activeBand = 0;
let nextFrameId = 1;
const animationFrames = new Map<number, FrameRequestCallback>();
const disconnect = vi.fn();
const closeAudioContext = vi.fn(() => Promise.resolve());
const cancelAnimationFrame = vi.fn((frameId: number) => {
  animationFrames.delete(frameId);
});

class FakeAudioContext {
  state: AudioContextState = "running";
  sampleRate = 48_000;

  createAnalyser() {
    return {
      fftSize: 512,
      frequencyBinCount: 256,
      smoothingTimeConstant: 0,
      getByteFrequencyData(samples: Uint8Array) {
        samples.fill(0);
        const binWidth = 48_000 / 512;
        const ranges = [
          [80, 250],
          [250, 500],
          [500, 900],
          [900, 1600],
          [1600, 3000],
        ];
        const [minimum, maximum] = ranges[activeBand];
        const start = Math.max(1, Math.ceil(minimum / binWidth));
        const end = Math.min(samples.length, Math.ceil(maximum / binWidth));

        for (let index = start; index < end; index += 1) {
          samples[index] = 255;
        }
      },
    } as unknown as AnalyserNode;
  }

  createMediaStreamSource() {
    return {
      connect() {},
      disconnect,
    } as unknown as MediaStreamAudioSourceNode;
  }

  resume() {
    return Promise.resolve();
  }

  close() {
    return closeAudioContext();
  }
}

beforeEach(() => {
  activeBand = 0;
  nextFrameId = 1;
  animationFrames.clear();
  reducedMotion.value = false;
  disconnect.mockClear();
  closeAudioContext.mockClear();
  cancelAnimationFrame.mockClear();
  vi.stubGlobal("AudioContext", FakeAudioContext);
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    const frameId = nextFrameId;
    nextFrameId += 1;
    animationFrames.set(frameId, callback);
    return frameId;
  });
  vi.stubGlobal("cancelAnimationFrame", cancelAnimationFrame);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useAudioBars", () => {
  it("updates each bar from its own voice-frequency band", () => {
    const { result, unmount } = renderHook(() =>
      useAudioBars({} as MediaStream),
    );

    flushAnimationFrame(performance.now() + 100);

    expect(result.current[0].get()).toBeGreaterThan(0);
    expect(result.current.slice(1).every((level) => level.get() === 0)).toBe(
      true,
    );

    activeBand = 4;
    flushAnimationFrame(performance.now() + 200);

    expect(result.current[4].get()).toBeGreaterThan(0);
    expect(result.current[1].get()).toBe(0);
    expect(result.current[2].get()).toBe(0);
    expect(result.current[3].get()).toBe(0);

    unmount();
    expect(cancelAnimationFrame).toHaveBeenCalledWith(3);
    expect(animationFrames.size).toBe(0);
    expect(disconnect).toHaveBeenCalledOnce();
    expect(closeAudioContext).toHaveBeenCalledOnce();
  });

  it("does not start an analyser when reduced motion is preferred", () => {
    reducedMotion.value = true;

    const { result } = renderHook(() => useAudioBars({} as MediaStream));

    expect(animationFrames.size).toBe(0);
    expect(result.current.every((level) => level.get() === 0)).toBe(true);
  });

  it("falls back to static bars when audio analysis cannot start", () => {
    vi.stubGlobal(
      "AudioContext",
      class {
        constructor() {
          throw new Error("audio unavailable");
        }
      },
    );

    const { result } = renderHook(() => useAudioBars({} as MediaStream));

    expect(animationFrames.size).toBe(0);
    expect(result.current.every((level) => level.get() === 0)).toBe(true);
  });

  it("closes a context when analyser initialization fails", () => {
    const closePartiallyStartedContext = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal(
      "AudioContext",
      class {
        state: AudioContextState = "running";
        close = closePartiallyStartedContext;
        createAnalyser() {
          throw new Error("analyser unavailable");
        }
      },
    );

    const { result } = renderHook(() => useAudioBars({} as MediaStream));

    expect(closePartiallyStartedContext).toHaveBeenCalledOnce();
    expect(animationFrames.size).toBe(0);
    expect(result.current.every((level) => level.get() === 0)).toBe(true);
  });
});

function flushAnimationFrame(now: number) {
  const frame = animationFrames.entries().next().value;
  if (!frame) return;
  const [frameId, callback] = frame;
  animationFrames.delete(frameId);
  act(() => callback(now));
}
