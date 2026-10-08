import { act, fireEvent, render, screen } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";
import { ScrollArea } from "./ScrollArea";

function getViewport(container: HTMLElement) {
  const viewport = container.querySelector<HTMLDivElement>(
    "[data-slot=scroll-area-viewport]",
  );
  if (!viewport) throw new Error("viewport not rendered");
  return viewport;
}

describe("ScrollArea", () => {
  it("renders its children inside the viewport and forwards its ref", () => {
    const ref = createRef<HTMLDivElement>();
    const { container } = render(
      <ScrollArea ref={ref} className="h-40" viewportClassName="pr-2">
        <p>Scrollable content</p>
      </ScrollArea>,
    );

    const viewport = getViewport(container);
    expect(viewport.textContent).toBe("Scrollable content");
    expect(viewport.className).toContain("pr-2");
    expect(ref.current?.className).toContain("h-40");
  });

  it("shows no scroll-to-top button unless asked", () => {
    const { container } = render(
      <ScrollArea className="h-40">
        <p>Content</p>
      </ScrollArea>,
    );

    const viewport = getViewport(container);
    viewport.scrollTop = 500;
    fireEvent.scroll(viewport);

    expect(screen.queryByRole("button", { name: "Scroll to top" })).toBeNull();
  });

  it("fades in a scroll-to-top button past the threshold and scrolls back up", () => {
    const { container } = render(
      <ScrollArea className="h-40" showScrollToTop>
        <p>Content</p>
      </ScrollArea>,
    );

    const viewport = getViewport(container);
    const scrollTo = vi.fn();
    viewport.scrollTo = scrollTo;

    expect(screen.queryByRole("button", { name: "Scroll to top" })).toBeNull();

    act(() => {
      viewport.scrollTop = 250;
      fireEvent.scroll(viewport);
    });

    fireEvent.click(screen.getByRole("button", { name: "Scroll to top" }));

    expect(scrollTo).toHaveBeenCalledWith(expect.objectContaining({ top: 0 }));
  });
});
