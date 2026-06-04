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
  it("shows enabled label when isDryRun is true", () => {
    render(<DryRunToggleButton isDryRun={true} onToggle={vi.fn()} />);
    expect(screen.getByText("Test mode enabled")).toBeTruthy();
  });

  it("shows enable label when isDryRun is false", () => {
    render(<DryRunToggleButton isDryRun={false} onToggle={vi.fn()} />);
    expect(screen.getByText("Enable test mode")).toBeTruthy();
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
