import {
  render as rtlRender,
  screen,
  fireEvent,
  cleanup,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ReactElement } from "react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { DryRunToggleButton } from "../DryRunToggleButton";

afterEach(cleanup);

function render(ui: ReactElement) {
  return rtlRender(<TooltipProvider>{ui}</TooltipProvider>);
}

// DryRunToggleButton only appears on new chats (no active session).
// It has no readOnly/isStreaming props — those scenarios are handled by hiding
// the button entirely at the ChatInput level when hasSession is true.
describe("DryRunToggleButton", () => {
  // The button is icon-only in the composer row, so its state has to reach
  // assistive tech through the label and show on the glyph itself — a tooltip
  // never opens on touch.
  it("names the active state for assistive tech when isDryRun is true", () => {
    render(<DryRunToggleButton isDryRun={true} onToggle={vi.fn()} />);
    expect(screen.getByLabelText("Test mode active")).toBeTruthy();
  });

  it("names the idle state for assistive tech when isDryRun is false", () => {
    render(<DryRunToggleButton isDryRun={false} onToggle={vi.fn()} />);
    expect(screen.getByLabelText("Enable Test mode")).toBeTruthy();
  });

  it("tints the glyph so the active state is visible without a tooltip", () => {
    render(<DryRunToggleButton isDryRun={true} onToggle={vi.fn()} />);
    const active = screen.getByRole("button").className;
    cleanup();

    render(<DryRunToggleButton isDryRun={false} onToggle={vi.fn()} />);
    expect(screen.getByRole("button").className).not.toBe(active);
  });

  it("calls onToggle when clicked", () => {
    const onToggle = vi.fn();
    render(<DryRunToggleButton isDryRun={false} onToggle={onToggle} />);
    fireEvent.click(screen.getByRole("button"));
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it("sets aria-pressed=true when isDryRun is true", () => {
    render(<DryRunToggleButton isDryRun={true} onToggle={vi.fn()} />);
    expect(screen.getByRole("button").getAttribute("aria-pressed")).toBe(
      "true",
    );
  });

  it("sets aria-pressed=false when isDryRun is false", () => {
    render(<DryRunToggleButton isDryRun={false} onToggle={vi.fn()} />);
    expect(screen.getByRole("button").getAttribute("aria-pressed")).toBe(
      "false",
    );
  });
});
