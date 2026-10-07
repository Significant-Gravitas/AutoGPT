import { TooltipProvider } from "@/components/ui/tooltip";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";
import { ToggleChip } from "./ToggleChip";

function renderChip(locked: boolean) {
  const onToggle = vi.fn();
  render(
    <TooltipProvider>
      <ToggleChip
        icon={null}
        label="Thinking"
        tooltip="Switch mode"
        ariaLabel="Toggle thinking"
        pressed={false}
        onToggle={onToggle}
        locked={locked}
      />
    </TooltipProvider>,
  );
  return onToggle;
}

describe("ToggleChip", () => {
  test("calls onToggle when clicked", () => {
    const onToggle = renderChip(false);
    fireEvent.click(screen.getByRole("button", { name: "Toggle thinking" }));
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  test("does not call onToggle while locked", () => {
    const onToggle = renderChip(true);
    fireEvent.click(screen.getByRole("button", { name: "Toggle thinking" }));
    expect(onToggle).not.toHaveBeenCalled();
  });
});
