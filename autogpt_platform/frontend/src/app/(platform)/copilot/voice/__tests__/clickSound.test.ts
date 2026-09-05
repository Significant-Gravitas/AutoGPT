import { afterEach, describe, expect, it, vi } from "vitest";

import {
  playClickSound,
  primeClickSound,
  resetClickSound,
} from "../clickSound";

describe("clickSound", () => {
  afterEach(() => {
    resetClickSound();
    vi.unstubAllGlobals();
  });

  it("plays a short tone through the shared context", () => {
    const { oscillator, contexts } = stubAudio("running");

    primeClickSound();
    playClickSound();
    playClickSound();

    expect(contexts).toHaveLength(1);
    expect(oscillator.start).toHaveBeenCalledTimes(2);
    expect(oscillator.stop).toHaveBeenCalledTimes(2);
    // Long enough to hear, short enough not to sit on top of the user's
    // next word.
    const [[startAt], [stopAt]] = oscillator.stop.mock.calls.length
      ? [oscillator.start.mock.calls[0], oscillator.stop.mock.calls[0]]
      : [[0], [0]];
    expect(stopAt - startAt).toBeLessThanOrEqual(0.2);
  });

  it("resumes a context the browser suspended", () => {
    const { contexts } = stubAudio("suspended");

    primeClickSound();

    expect(contexts[0].resume).toHaveBeenCalled();
  });

  it("stays silent rather than throwing when Web Audio is missing", () => {
    vi.stubGlobal("AudioContext", undefined);

    expect(() => primeClickSound()).not.toThrow();
    expect(() => playClickSound()).not.toThrow();
  });

  it("stays silent rather than throwing when the context refuses to start", () => {
    vi.stubGlobal(
      "AudioContext",
      class {
        constructor() {
          throw new Error("not allowed");
        }
      },
    );

    expect(() => playClickSound()).not.toThrow();
  });
});

function stubAudio(state: "running" | "suspended") {
  const oscillator = {
    type: "",
    frequency: {
      setValueAtTime: vi.fn(),
      exponentialRampToValueAtTime: vi.fn(),
    },
    connect: vi.fn(() => gain),
    start: vi.fn(),
    stop: vi.fn(),
  };
  const gain = {
    gain: {
      setValueAtTime: vi.fn(),
      linearRampToValueAtTime: vi.fn(),
      exponentialRampToValueAtTime: vi.fn(),
    },
    connect: vi.fn(),
  };
  const contexts: {
    resume: ReturnType<typeof vi.fn>;
    close: ReturnType<typeof vi.fn>;
  }[] = [];

  vi.stubGlobal(
    "AudioContext",
    class {
      state = state;
      currentTime = 0;
      destination = {};
      resume = vi.fn(() => {
        this.state = "running";
        return Promise.resolve();
      });
      close = vi.fn(() => Promise.resolve());
      createOscillator = () => oscillator;
      createGain = () => gain;
      constructor() {
        contexts.push(this);
      }
    },
  );

  return { oscillator, gain, contexts };
}
