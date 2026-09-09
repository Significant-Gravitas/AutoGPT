import { act, StrictMode } from "react";
import { describe, expect, it, vi } from "vitest";

import { render } from "@/tests/integrations/test-utils";

const takeMicLevel = vi.fn(() => 0.05);
vi.mock("../micLevel", () => ({
  takeMicLevel: () => takeMicLevel(),
  reportMicLevel: vi.fn(),
}));
vi.mock("../speechLevel", () => ({ readSpeechLevel: () => null }));

import { MIN_SCALE, TICK_MS, VoiceTrace } from "../components/VoiceTrace";

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
    expect(tallest(container)).toBeLessThanOrEqual(MIN_SCALE * 100);

    takeMicLevel.mockImplementation(() => 0.05);
    act(() => void vi.advanceTimersByTime(TICK_MS * 5));

    // Read inside the updater, every one of these ticks would have drawn the
    // emptied read instead.
    expect(tallest(container)).toBeGreaterThan(40);
    takeMicLevel.mockImplementation(() => 0.05);
    vi.useRealTimers();
  });
});

describe("VoiceTrace colour", () => {
  it("keeps each column in the colour of the stage that recorded it", () => {
    // Switching from speaking to listening used to repaint the whole strip
    // green, so AutoPilot's words looked like the user's.
    vi.useFakeTimers();
    takeMicLevel.mockImplementation(() => 0.02);
    const { container, rerender } = render(
      <VoiceTrace source="pulse" color="bg-zinc-900" />,
    );
    act(() => void vi.advanceTimersByTime(TICK_MS * 5));

    rerender(<VoiceTrace source="mic" color="bg-emerald-500" />);
    act(() => void vi.advanceTimersByTime(TICK_MS * 2));

    const colors = [
      ...container.querySelectorAll<HTMLElement>("span[style]"),
    ].map((c) => (c.className.includes("bg-emerald-500") ? "mic" : "tts"));
    expect(colors.slice(-2)).toEqual(["mic", "mic"]);
    expect(colors.slice(-7, -2)).toEqual(["tts", "tts", "tts", "tts", "tts"]);
    vi.useRealTimers();
  });
});

function tallest(container: HTMLElement): number {
  const columns = container.querySelectorAll<HTMLElement>("span[style]");
  return Math.max(...[...columns].map((c) => parseFloat(c.style.height)));
}
