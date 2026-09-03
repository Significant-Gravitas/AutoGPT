import { describe, expect, it, vi } from "vitest";

import { render, screen } from "@/tests/integrations/test-utils";

import { VoiceModeBar } from "../components/VoiceModeBar";
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
    expect(screen.getByRole("button", { name: "Stop" })).toBeDefined();
  });
});

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
