import { fireEvent, render, screen } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";
import { Sheet } from "./Sheet";

describe("Sheet", () => {
  it("opens from its trigger and names the dialog after the title", () => {
    render(
      <Sheet title="Run output" trigger={<button>Open</button>}>
        <p>Body</p>
      </Sheet>,
    );

    expect(screen.queryByRole("dialog")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    const dialog = screen.getByRole("dialog", { name: "Run output" });
    expect(dialog.textContent).toContain("Body");
  });

  it("links the description", () => {
    render(
      <Sheet title="Run output" description="Latest output" open>
        <p>Body</p>
      </Sheet>,
    );

    const dialog = screen.getByRole("dialog", { name: "Run output" });
    const descriptionId = dialog.getAttribute("aria-describedby");
    expect(document.getElementById(descriptionId ?? "")?.textContent).toBe(
      "Latest output",
    );
  });

  it("keeps a hidden title as the accessible name", () => {
    render(
      <Sheet title="Filters" hideTitle open>
        <p>Body</p>
      </Sheet>,
    );

    const dialog = screen.getByRole("dialog", { name: "Filters" });
    expect(dialog.getAttribute("aria-describedby")).toBeNull();
    expect(
      screen.getByRole("heading", { name: "Filters" }).className,
    ).toContain("sr-only");
  });

  it("closes through the close button", () => {
    const onOpenChange = vi.fn();
    render(
      <Sheet title="Run output" open onOpenChange={onOpenChange}>
        <p>Body</p>
      </Sheet>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Close" }));

    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("does not close on Escape while an IME is composing", () => {
    const onOpenChange = vi.fn();
    render(
      <Sheet title="Run output" open onOpenChange={onOpenChange}>
        <p>Body</p>
      </Sheet>,
    );

    fireEvent.keyDown(screen.getByRole("dialog"), {
      key: "Escape",
      isComposing: true,
    });
    expect(onOpenChange).not.toHaveBeenCalled();

    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("applies the side variant, renders actions and footer, and forwards its ref", () => {
    const ref = createRef<HTMLDivElement>();
    render(
      <Sheet
        ref={ref}
        title="Run output"
        side="left"
        open
        actions={<button>Export</button>}
        footer={<button>Save</button>}
        className="sm:max-w-xl"
      >
        <p>Body</p>
      </Sheet>,
    );

    expect(ref.current).toBe(screen.getByRole("dialog"));
    expect(ref.current?.className).toContain("left-0");
    expect(ref.current?.className).toContain("sm:max-w-xl");
    expect(screen.getByRole("button", { name: "Export" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Save" })).toBeDefined();
  });
});
