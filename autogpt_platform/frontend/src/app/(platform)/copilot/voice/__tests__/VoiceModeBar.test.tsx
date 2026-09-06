import { act } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { render, screen } from "@/tests/integrations/test-utils";

vi.mock("../speechLevel", () => ({ readSpeechLevel: vi.fn(() => null) }));

import { reportMicLevel } from "../micLevel";
import { readSpeechLevel } from "../speechLevel";

import { VoiceModeBar } from "../components/VoiceModeBar";
import { MIN_SCALE, TICK_MS } from "../components/VoiceTrace";
import { VoiceModeButton } from "../components/VoiceModeButton";

/** Height of a bar at rest, as the percentage the style carries. */
const REST = MIN_SCALE * 100;

describe("VoiceModeBar", () => {
  it("stays out of the way when voice mode is off", () => {
    const { container } = render(
      <VoiceModeBar state="off" statusLabel="Voice mode off" />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("announces the current state to screen readers", () => {
    render(<VoiceModeBar state="listening" statusLabel="Listening" />);
    const status = screen.getByRole("status");
    expect(status.textContent).toContain("Listening");
    expect(status.getAttribute("aria-live")).toBe("polite");
  });

  it("gives listening, thinking and speaking each their own colour", () => {
    // The three read as one continuous animation without it: the handover
    // from the user's turn to AutoPilot's was the part that did not land.
    vi.useFakeTimers();
    const seen = new Set<string>();
    const { container, rerender } = render(
      <VoiceModeBar state="listening" statusLabel="Listening" />,
    );
    act(() => void vi.advanceTimersByTime(TICK_MS));
    seen.add(newestColor(container));

    for (const state of ["thinking", "speaking"] as const) {
      rerender(<VoiceModeBar state={state} statusLabel={state} />);
      act(() => void vi.advanceTimersByTime(TICK_MS));
      seen.add(newestColor(container));
    }

    expect(seen.size).toBe(3);
    vi.useRealTimers();
  });

  it("follows the synthesised audio while AutoPilot speaks", () => {
    vi.useFakeTimers();
    const levels = vi.mocked(readSpeechLevel);
    levels.mockReturnValue(0.001);
    const { container } = render(
      <VoiceModeBar state="speaking" statusLabel="Speaking" />,
    );
    act(() => void vi.advanceTimersByTime(TICK_MS * 20));

    levels.mockReturnValue(0.2);
    act(() => void vi.advanceTimersByTime(TICK_MS));
    const loud = newestColumn(container);
    expect(loud).toBeGreaterThan(40);

    levels.mockReturnValue(0.001);
    act(() => void vi.advanceTimersByTime(TICK_MS));

    expect(newestColumn(container)).toBeLessThan(loud);
    vi.useRealTimers();
  });

  it("keeps moving when the audio cannot be measured", () => {
    // No Web Audio, or the element was already routed elsewhere. A dead flat
    // line would read as "it has stopped".
    vi.useFakeTimers();
    vi.mocked(readSpeechLevel).mockReturnValue(null);
    const { container } = render(
      <VoiceModeBar state="speaking" statusLabel="Speaking" />,
    );
    act(() => void vi.advanceTimersByTime(TICK_MS * 8));

    expect(newestColumn(container)).toBeGreaterThan(20);
    vi.useRealTimers();
  });

  it("draws a conversational voice at a readable height, and a quiet room flat", () => {
    // The defect this pins: a linear scale put normal speech (~0.03 RMS) at
    // 12% — visually flat — while one loud syllable clipped to 100%, so the
    // meter read as "zero most of the time with an occasional spike".
    vi.useFakeTimers();
    const { container } = render(
      <VoiceModeBar state="hearing" statusLabel="Hearing" />,
    );
    settleRoom(0.001);

    reportMicLevel(0.03);
    act(() => void vi.advanceTimersByTime(TICK_MS));
    expect(newestColumn(container)).toBeGreaterThan(25);

    reportMicLevel(0.001);
    act(() => void vi.advanceTimersByTime(TICK_MS));
    expect(newestColumn(container)).toBeLessThanOrEqual(REST);
    vi.useRealTimers();
  });

  it("draws the mic level while the mic is open, and stops when it shuts", () => {
    vi.useFakeTimers();
    const { container, rerender } = render(
      <VoiceModeBar state="hearing" statusLabel="Hearing" />,
    );
    settleRoom(0.001);

    reportMicLevel(0.9);
    act(() => void vi.advanceTimersByTime(TICK_MS));
    const loud = newestColumn(container);

    // Nothing reported since — a shut mic reads as silence within one tick.
    act(() => void vi.advanceTimersByTime(TICK_MS));
    expect(newestColumn(container)).toBeLessThan(loud);

    // Thinking has no input to draw. It must grow out of the flat line the
    // mic left — not jump — and then must not read as silence.
    rerender(<VoiceModeBar state="thinking" statusLabel="Thinking" />);
    act(() => void vi.advanceTimersByTime(TICK_MS));
    expect(newestColumn(container)).toBeLessThanOrEqual(REST);
    act(() => void vi.advanceTimersByTime(TICK_MS * 7));
    expect(newestColumn(container)).toBeGreaterThan(20);
    vi.useRealTimers();
  });
});

/** The adaptive scale learns the room before it can draw a voice against it. */
function settleRoom(level: number) {
  for (let i = 0; i < 20; i++) {
    reportMicLevel(level);
    act(() => void vi.advanceTimersByTime(TICK_MS));
  }
}

/** The newest column carries the current stage; older ones keep their own. */
function newestColor(container: HTMLElement): string {
  const columns = container.querySelectorAll<HTMLElement>("span[style]");
  return columns[columns.length - 1]?.className ?? "";
}

function newestColumn(container: HTMLElement): number {
  const columns = container.querySelectorAll<HTMLElement>("span[style]");
  return parseFloat(columns[columns.length - 1].style.height);
}

describe("VoiceModeButton", () => {
  it("labels itself by what the click will do", () => {
    const { rerender } = render(
      <VoiceModeButton isActive={false} onClick={vi.fn()} />,
    );
    expect(
      screen.getByRole("button", { name: "Talk to AutoPilot" }),
    ).toBeDefined();

    rerender(<VoiceModeButton isActive onClick={vi.fn()} />);
    const active = screen.getByRole("button", { name: "Leave voice mode" });
    expect(active.getAttribute("aria-pressed")).toBe("true");
  });

  it("becomes the stop control while AutoPilot speaks", () => {
    // The same click leaves voice mode either way; while a reply is playing
    // the user reads it as "make it stop", so the icon and label say that.
    render(<VoiceModeButton isActive speaking onClick={vi.fn()} />);

    expect(screen.getByRole("button", { name: "Stop" })).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Leave voice mode" }),
    ).toBeNull();
  });
});
