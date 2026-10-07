import { fireEvent, render, screen } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";
import { Checkbox } from "./Checkbox";

describe("Checkbox", () => {
  it("toggles through its label", () => {
    const onCheckedChange = vi.fn();
    render(<Checkbox label="Accept terms" onCheckedChange={onCheckedChange} />);

    const checkbox = screen.getByRole("checkbox", { name: "Accept terms" });
    expect(checkbox.getAttribute("aria-checked")).toBe("false");

    fireEvent.click(screen.getByText("Accept terms"));

    expect(onCheckedChange).toHaveBeenCalledWith(true);
    expect(checkbox.getAttribute("aria-checked")).toBe("true");
  });

  it("reports the indeterminate state as mixed", () => {
    render(<Checkbox label="Select all" checked="indeterminate" />);

    const checkbox = screen.getByRole("checkbox", { name: "Select all" });
    expect(checkbox.getAttribute("aria-checked")).toBe("mixed");
    expect(checkbox.getAttribute("data-state")).toBe("indeterminate");
  });

  it("links the description and error and marks the box invalid", () => {
    render(
      <Checkbox
        id="terms"
        label="Accept terms"
        description="Read them first."
        error="Required"
        aria-describedby="external-hint"
      />,
    );

    const checkbox = screen.getByRole("checkbox", { name: "Accept terms" });
    expect(checkbox.getAttribute("aria-invalid")).toBe("true");
    expect(checkbox.getAttribute("aria-describedby")).toBe(
      "external-hint terms-description terms-error",
    );
    expect(screen.getByText("Read them first.").id).toBe("terms-description");
    expect(screen.getByText("Required").id).toBe("terms-error");
  });

  it("renders only the box when there is no label", () => {
    const { container } = render(<Checkbox aria-label="Select row" />);

    expect(screen.getByRole("checkbox", { name: "Select row" })).toBe(
      container.firstChild,
    );
  });

  it("does not toggle when disabled", () => {
    const onCheckedChange = vi.fn();
    render(
      <Checkbox label="Locked" disabled onCheckedChange={onCheckedChange} />,
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "Locked" }));

    expect(onCheckedChange).not.toHaveBeenCalled();
  });

  it("applies the size classes and forwards its ref", () => {
    const ref = createRef<HTMLButtonElement>();
    render(<Checkbox ref={ref} aria-label="Large" size="md" />);

    expect(ref.current).toBe(screen.getByRole("checkbox", { name: "Large" }));
    expect(ref.current?.className).toContain("size-5");
  });
});
