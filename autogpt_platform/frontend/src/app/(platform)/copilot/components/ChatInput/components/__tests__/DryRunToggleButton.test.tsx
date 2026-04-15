import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DryRunToggleButton } from "../DryRunToggleButton";

afterEach(cleanup);

// DryRunToggleButton only appears on new chats (no active session).
// It has no readOnly/isStreaming props — those scenarios are handled by hiding
// the button entirely at the ChatInput level when hasSession is true.
describe("DryRunToggleButton", () => {
  it("shows Test label when isDryRun is true", () => {
    render(<DryRunToggleButton isDryRun={true} onToggle={vi.fn()} />);
    expect(screen.getByText("Test")).toBeTruthy();
  });

  it("shows no text label when isDryRun is false", () => {
    render(<DryRunToggleButton isDryRun={false} onToggle={vi.fn()} />);
    expect(screen.queryByText("Test")).toBeNull();
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
