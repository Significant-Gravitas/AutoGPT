import { render, screen } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it } from "vitest";
import { Separator } from "./Separator";

describe("Separator", () => {
  it("is decorative and horizontal by default", () => {
    const { container } = render(<Separator />);

    const separator = container.firstElementChild as HTMLElement;
    expect(separator.getAttribute("role")).toBe("none");
    expect(separator.getAttribute("data-orientation")).toBe("horizontal");
    expect(separator.className).toContain("h-px");
    expect(separator.className).toContain("w-full");
  });

  it("exposes a vertical separator when not decorative", () => {
    render(<Separator orientation="vertical" decorative={false} />);

    const separator = screen.getByRole("separator");
    expect(separator.getAttribute("aria-orientation")).toBe("vertical");
    expect(separator.className).toContain("w-px");
    expect(separator.className).toContain("h-full");
  });

  it("merges className and forwards its ref", () => {
    const ref = createRef<HTMLDivElement>();
    render(<Separator ref={ref} decorative={false} className="my-4" />);

    expect(ref.current).toBe(screen.getByRole("separator"));
    expect(ref.current?.className).toContain("my-4");
  });
});
