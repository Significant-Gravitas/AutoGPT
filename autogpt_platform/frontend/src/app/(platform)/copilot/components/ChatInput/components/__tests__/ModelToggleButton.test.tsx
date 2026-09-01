import {
  render as rtlRender,
  screen,
  fireEvent,
  cleanup,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ReactElement } from "react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ModelToggleButton } from "../ModelToggleButton";

afterEach(cleanup);

function render(ui: ReactElement) {
  return rtlRender(<TooltipProvider>{ui}</TooltipProvider>);
}

describe("ModelToggleButton", () => {
  it("carries no text label — the composer row is icon-only", () => {
    render(<ModelToggleButton model="standard" onToggle={vi.fn()} />);
    expect(screen.queryByText("Standard")).toBeNull();
    expect(screen.queryByText("Advanced")).toBeNull();
  });

  // Model tier drives cost and quality, and a tooltip never opens on touch,
  // so the raised tier has to be legible on the glyph itself.
  it("tints the glyph so the advanced tier is visible without a tooltip", () => {
    render(<ModelToggleButton model="advanced" onToggle={vi.fn()} />);
    const advanced = screen.getByRole("button").className;
    expect(advanced).toContain("emerald");
    cleanup();

    render(<ModelToggleButton model="standard" onToggle={vi.fn()} />);
    expect(screen.getByRole("button").className).not.toContain("emerald");
  });

  it("calls onToggle when clicked", () => {
    const onToggle = vi.fn();
    render(<ModelToggleButton model="standard" onToggle={onToggle} />);
    fireEvent.click(screen.getByRole("button"));
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it("sets aria-pressed=false for standard", () => {
    render(<ModelToggleButton model="standard" onToggle={vi.fn()} />);
    const btn = screen.getByLabelText("Switch to Advanced model");
    expect(btn.getAttribute("aria-pressed")).toBe("false");
  });

  it("sets aria-pressed=true for advanced", () => {
    render(<ModelToggleButton model="advanced" onToggle={vi.fn()} />);
    const btn = screen.getByLabelText("Switch to Balanced model");
    expect(btn.getAttribute("aria-pressed")).toBe("true");
  });
});
