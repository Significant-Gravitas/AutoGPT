import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";

import { ScrollArea, ScrollBar } from "../ScrollArea";

function viewportOf(container: HTMLElement) {
  const viewport = container.querySelector<HTMLDivElement>(
    '[data-slot="scroll-area-viewport"]',
  );
  if (!viewport) throw new Error("no viewport");
  return viewport;
}

describe("ScrollArea", () => {
  it("renders its children in a vertical viewport by default", () => {
    const { container } = render(
      <ScrollArea className="h-40">
        <p>Row</p>
      </ScrollArea>,
    );

    expect(screen.getByText("Row")).toBeDefined();
    const root = container.querySelector('[data-slot="scroll-area"]');
    expect(root?.getAttribute("class")).toContain("relative");
    expect(root?.getAttribute("class")).toContain("h-40");
    const viewport = viewportOf(container);
    expect(viewport.style.overflowX).toBe("hidden");
    expect(viewport.style.overflowY).toBe("scroll");
  });

  it("scrolls on the axes its orientation names", () => {
    const horizontal = render(
      <ScrollArea orientation="horizontal">
        <p>Wide</p>
      </ScrollArea>,
    );
    expect(viewportOf(horizontal.container).style.overflowX).toBe("scroll");
    expect(viewportOf(horizontal.container).style.overflowY).toBe("hidden");

    const both = render(
      <ScrollArea orientation="both">
        <p>Big</p>
      </ScrollArea>,
    );
    expect(viewportOf(both.container).style.overflowX).toBe("scroll");
    expect(viewportOf(both.container).style.overflowY).toBe("scroll");
  });

  it("forwards its ref to the root", () => {
    const ref = createRef<HTMLDivElement>();
    render(
      <ScrollArea ref={ref}>
        <p>Row</p>
      </ScrollArea>,
    );
    expect(ref.current?.getAttribute("data-slot")).toBe("scroll-area");
  });

  it("shows no scroll-to-top button unless asked", () => {
    const { container } = render(
      <ScrollArea>
        <p>Row</p>
      </ScrollArea>,
    );
    const viewport = viewportOf(container);
    viewport.scrollTop = 500;
    fireEvent.scroll(viewport);
    expect(screen.queryByRole("button", { name: "Scroll to top" })).toBeNull();
  });

  it("shows the scroll-to-top button past 200px and scrolls back up", () => {
    const { container } = render(
      <ScrollArea showScrollToTop>
        <p>Row</p>
      </ScrollArea>,
    );
    const viewport = viewportOf(container);
    const scrollTo = vi.fn();
    viewport.scrollTo = scrollTo;

    expect(screen.queryByRole("button", { name: "Scroll to top" })).toBeNull();

    act(() => {
      viewport.scrollTop = 300;
      fireEvent.scroll(viewport);
    });
    fireEvent.click(screen.getByRole("button", { name: "Scroll to top" }));

    expect(scrollTo).toHaveBeenCalledWith(expect.objectContaining({ top: 0 }));
  });

  it("exports the scroll bar for callers that compose their own root", () => {
    expect(ScrollBar.displayName).toBeDefined();
  });
});
