import { render, screen } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it } from "vitest";
import { Kbd } from "./Kbd";

describe("Kbd", () => {
  it("renders a kbd element with its key", () => {
    render(<Kbd>esc</Kbd>);

    const key = screen.getByText("esc");
    expect(key.tagName).toBe("KBD");
    expect(key.className).toContain("h-5");
  });

  it("applies the medium size and merges className last", () => {
    render(
      <Kbd size="md" className="text-zinc-600">
        K
      </Kbd>,
    );

    const key = screen.getByText("K");
    expect(key.className).toContain("h-6");
    expect(key.className).toContain("text-zinc-600");
    expect(key.className).not.toContain("text-zinc-800");
  });

  it("forwards its ref and extra attributes", () => {
    const ref = createRef<HTMLElement>();
    render(
      <Kbd ref={ref} aria-hidden="true">
        ↵
      </Kbd>,
    );

    expect(ref.current?.tagName).toBe("KBD");
    expect(ref.current?.getAttribute("aria-hidden")).toBe("true");
  });
});
