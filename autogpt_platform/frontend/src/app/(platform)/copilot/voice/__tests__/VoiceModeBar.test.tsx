import { act } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { render, screen } from "@/tests/integrations/test-utils";

import { reportMicLevel } from "../micLevel";

import { VoiceModeBar } from "../components/VoiceModeBar";
import { TICK_MS } from "../components/VoiceTrace";
import { VoiceModeButton } from "../components/VoiceModeButton";

describe("VoiceModeBar", () => {
  it("stays out of the way when voice mode is off", () => {
    const { container } = render(
      <VoiceModeBar
        state="off"
        statusLabel="Voice mode off"
        onStop={vi.fn()}
      />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("announces the current state to screen readers", () => {
    render(
      <VoiceModeBar
        state="listening"
        statusLabel="Listening"
        onStop={vi.fn()}
      />,
    );
    const status = screen.getByRole("status");
    expect(status.textContent).toContain("Listening");
    expect(status.getAttribute("aria-live")).toBe("polite");
  });

  it("offers Stop only while AutoPilot is speaking", () => {
    const { rerender } = render(
      <VoiceModeBar state="thinking" statusLabel="Thinking" onStop={vi.fn()} />,
    );
    expect(screen.queryByRole("button", { name: "Stop" })).toBeNull();

    rerender(
      <VoiceModeBar state="speaking" statusLabel="Speaking" onStop={vi.fn()} />,
    );
    expect(screen.getByRole("button", { name: "Stop speaking" })).toBeDefined();
  });

  it("draws the mic level while the mic is open, and stops when it shuts", () => {
    vi.useFakeTimers();
    const { container, rerender } = render(
      <VoiceModeBar state="hearing" statusLabel="Hearing" onStop={vi.fn()} />,
    );

    reportMicLevel(0.9);
    act(() => void vi.advanceTimersByTime(TICK_MS));
    const loud = newestColumn(container);

    // Nothing reported since — a shut mic reads as silence within one tick.
    act(() => void vi.advanceTimersByTime(TICK_MS));
    expect(newestColumn(container)).toBeLessThan(loud);

    // Thinking has no input to draw, so the trace must not read as silence.
    rerender(
      <VoiceModeBar state="thinking" statusLabel="Thinking" onStop={vi.fn()} />,
    );
    act(() => void vi.advanceTimersByTime(TICK_MS * 3));
    expect(newestColumn(container)).toBeGreaterThan(0);
    vi.useRealTimers();
  });
});

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
});
