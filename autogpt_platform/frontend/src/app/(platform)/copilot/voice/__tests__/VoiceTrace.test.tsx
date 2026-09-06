import { act, StrictMode } from "react";
import { describe, expect, it, vi } from "vitest";

import { render } from "@/tests/integrations/test-utils";

const takeMicLevel = vi.fn(() => 0.05);
vi.mock("../micLevel", () => ({
  takeMicLevel: () => takeMicLevel(),
  reportMicLevel: vi.fn(),
}));
vi.mock("../speechLevel", () => ({ readSpeechLevel: () => null }));

import { TICK_MS, VoiceTrace } from "../components/VoiceTrace";

describe("VoiceTrace", () => {
  it("consumes one level per tick, even under StrictMode", () => {
    // Taking a level empties it. Reading inside the state updater meant
    // React's second invocation saw nothing, so the strip drew a flat line
    // with a spike only when a frame landed between the two calls.
    vi.useFakeTimers();
    takeMicLevel.mockClear();

    render(
      <StrictMode>
        <VoiceTrace source="mic" color="bg-emerald-500" />
      </StrictMode>,
    );
    takeMicLevel.mockClear();
    act(() => void vi.advanceTimersByTime(TICK_MS * 3));

    expect(takeMicLevel).toHaveBeenCalledTimes(3);
    vi.useRealTimers();
  });

  it("draws the level it consumed, not the emptied second read", () => {
    vi.useFakeTimers();
    takeMicLevel.mockImplementation(() => 0.001);

    const { container } = render(
      <StrictMode>
        <VoiceTrace source="mic" color="bg-emerald-500" />
      </StrictMode>,
    );
    act(() => void vi.advanceTimersByTime(TICK_MS * 20));
    expect(tallest(container)).toBeLessThan(10);

    takeMicLevel.mockImplementation(() => 0.05);
    act(() => void vi.advanceTimersByTime(TICK_MS * 5));

    // Read inside the updater, every one of these ticks would have drawn the
    // emptied read instead.
    expect(tallest(container)).toBeGreaterThan(40);
    takeMicLevel.mockImplementation(() => 0.05);
    vi.useRealTimers();
  });
});

function tallest(container: HTMLElement): number {
  const columns = container.querySelectorAll<HTMLElement>("span[style]");
  return Math.max(...[...columns].map((c) => parseFloat(c.style.height)));
}
