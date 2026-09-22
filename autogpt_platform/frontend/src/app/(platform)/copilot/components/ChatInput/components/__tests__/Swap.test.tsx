import { act, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Swap } from "../ConnectionPicker/Swap";

// Matches SWAP_MS in useSwap.
const SWAP_MS = 150;

function isHidden(element: HTMLElement) {
  return element.className.includes("opacity-0");
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("Swap", () => {
  it("leaves nothing faded out when the value comes back mid-swap", () => {
    // Two clicks inside the exit's 150ms -- advanced, then straight back to
    // standard. The exit is cancelled with its timer, so nothing is left to
    // return the content to rest, and an exit left standing holds it invisible
    // until some later swap happens to clear it.
    const { rerender } = render(<Swap swapKey="standard">Balanced</Swap>);
    const swap = screen.getByText("Balanced");
    expect(isHidden(swap)).toBe(false);

    rerender(<Swap swapKey="advanced">Advanced</Swap>);
    expect(isHidden(swap)).toBe(true);

    rerender(<Swap swapKey="standard">Balanced</Swap>);
    expect(isHidden(swap)).toBe(false);

    // And the cancelled swap does not land late.
    act(() => vi.advanceTimersByTime(SWAP_MS * 2));
    expect(screen.getByText("Balanced")).toBeDefined();
    expect(isHidden(swap)).toBe(false);
  });

  it("still swaps content once the exit has run", () => {
    const { rerender } = render(<Swap swapKey="standard">Balanced</Swap>);

    rerender(<Swap swapKey="advanced">Advanced</Swap>);
    act(() => vi.advanceTimersByTime(SWAP_MS));

    expect(screen.getByText("Advanced")).toBeDefined();
  });
});
