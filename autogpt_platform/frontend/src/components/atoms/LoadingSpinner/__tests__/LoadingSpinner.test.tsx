import { render } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";

import { LoadingSpinner } from "../LoadingSpinner";

function svgOf(container: HTMLElement) {
  const svg = container.querySelector("svg");
  if (!svg) throw new Error("no svg");
  return svg;
}

describe("LoadingSpinner", () => {
  it("draws the legacy loader-circle ring for variant circle", () => {
    const { container } = render(
      <LoadingSpinner variant="circle" aria-hidden="true" className="size-4" />,
    );

    const svg = svgOf(container);
    const paths = svg.querySelectorAll("path");
    expect(paths.length).toBe(1);
    expect(paths[0].getAttribute("d")).toBe("M21 12a9 9 0 1 1-6.219-8.56");
    expect(paths[0].getAttribute("stroke")).toBe("currentColor");
    expect(paths[0].getAttribute("stroke-linecap")).toBe("round");
    expect(svg.getAttribute("aria-hidden")).toBe("true");
    expect(svg.getAttribute("data-slot")).toBe("loading-spinner");
  });

  it("merges the caller's size over the default size", () => {
    const { container } = render(
      <LoadingSpinner variant="circle" className="size-16" />,
    );

    const classes = svgOf(container).getAttribute("class") ?? "";
    expect(classes).toContain("animate-spin");
    expect(classes).toContain("size-16");
    expect(classes).not.toContain("h-6");
  });

  it("keeps the default glyph when no variant is given", () => {
    const { container } = render(<LoadingSpinner />);

    expect(svgOf(container).querySelector("path")?.getAttribute("d")).not.toBe(
      "M21 12a9 9 0 1 1-6.219-8.56",
    );
  });
});
