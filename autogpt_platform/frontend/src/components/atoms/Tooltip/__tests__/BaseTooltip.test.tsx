import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as UITooltip from "@/components/ui/tooltip";
import * as AtomTooltip from "../BaseTooltip";

afterEach(() => {
  vi.restoreAllMocks();
});

describe.each([
  ["UI", UITooltip],
  ["atom", AtomTooltip],
] as const)("%s tooltip pointer transit", (_, components) => {
  it("shows the next button's tooltip when moving beneath the previous tooltip", async () => {
    const { TooltipProvider, Tooltip, TooltipTrigger, TooltipContent } =
      components;
    render(
      <TooltipProvider delayDuration={0}>
        <Tooltip delayDuration={0}>
          <TooltipTrigger asChild>
            <button>First action</button>
          </TooltipTrigger>
          <TooltipContent data-testid="first-tooltip">
            First description
          </TooltipContent>
        </Tooltip>
        <Tooltip delayDuration={0}>
          <TooltipTrigger asChild>
            <button>Second action</button>
          </TooltipTrigger>
          <TooltipContent>Second description</TooltipContent>
        </Tooltip>
      </TooltipProvider>,
    );
    const first = screen.getByRole("button", { name: "First action" });
    const second = screen.getByRole("button", { name: "Second action" });
    vi.spyOn(first, "getBoundingClientRect").mockReturnValue(
      new DOMRect(0, 0, 20, 20),
    );
    fireEvent.pointerMove(first, {
      pointerType: "mouse",
      clientX: 10,
      clientY: 10,
    });
    expect(
      await screen.findByRole("tooltip", { name: "First description" }),
    ).toBeDefined();
    const firstTooltip = screen.getByTestId("first-tooltip");
    vi.spyOn(firstTooltip, "getBoundingClientRect").mockReturnValue(
      new DOMRect(24, 0, 60, 20),
    );
    const position = {
      pointerType: "mouse",
      clientX: 21,
      clientY: 10,
      relatedTarget: second,
    };
    fireEvent.pointerOut(first, position);
    fireEvent.pointerLeave(first, position);
    fireEvent.pointerOver(second, {
      pointerType: "mouse",
      clientX: 35,
      clientY: 10,
      relatedTarget: first,
    });
    fireEvent.pointerMove(second, {
      pointerType: "mouse",
      clientX: 35,
      clientY: 10,
    });
    expect(
      await screen.findByRole("tooltip", { name: "Second description" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("tooltip", { name: "First description" }),
    ).toBeNull();
  });
});
