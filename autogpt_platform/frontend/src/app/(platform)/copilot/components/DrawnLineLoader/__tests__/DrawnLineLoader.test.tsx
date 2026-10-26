import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { DrawnLineLoader } from "../DrawnLineLoader";
import { DRAWN_LINE_SHAPES } from "../drawnLineShapes";

function renderLoader(
  props: React.ComponentProps<typeof DrawnLineLoader> = {},
) {
  const { container } = render(<DrawnLineLoader {...props} />);
  const svg = container.firstElementChild as SVGSVGElement;
  const paths = Array.from(svg.querySelectorAll("path")) as SVGPathElement[];
  return { svg, paths };
}

afterEach(cleanup);

describe("DrawnLineLoader", () => {
  it("renders a hidden 20px svg by default", () => {
    const { svg } = renderLoader();

    expect(svg.tagName.toLowerCase()).toBe("svg");
    expect(svg.getAttribute("aria-hidden")).toBe("true");
    expect(svg.getAttribute("width")).toBe("20");
    expect(svg.getAttribute("height")).toBe("20");
    expect(svg.getAttribute("viewBox")).toBe("0 0 24 24");
  });

  it("strokes with currentColor and round caps", () => {
    const { svg } = renderLoader();

    expect(svg.getAttribute("stroke")).toBe("currentColor");
    expect(svg.getAttribute("fill")).toBe("none");
    expect(svg.getAttribute("stroke-linecap")).toBe("round");
    expect(svg.getAttribute("stroke-linejoin")).toBe("round");
  });

  it("cycles 3 shapes, one normalised stroke path each", () => {
    const { paths } = renderLoader();

    // The keyframes give each shape a third of the cycle.
    expect(DRAWN_LINE_SHAPES).toHaveLength(3);
    expect(paths).toHaveLength(DRAWN_LINE_SHAPES.length);
    paths.forEach((path, index) => {
      expect(path.getAttribute("d")).toBe(DRAWN_LINE_SHAPES[index].d);
      // pathLength=1 lets one CSS keyframe draw any artwork length.
      expect(path.getAttribute("pathLength")).toBe("1");
    });
  });

  it("gives each shape a 2.5s slot of one shared cycle", () => {
    const { paths } = renderLoader();
    const cycle = DRAWN_LINE_SHAPES.length * 2.5;

    paths.forEach((path, index) => {
      expect(path.style.animationDuration).toBe(`${cycle}s`);
      expect(path.style.animationDelay).toBe(`${index * 2.5}s`);
    });
  });

  it("scales with size and forwards className", () => {
    const { svg } = renderLoader({ size: 16, className: "text-red" });

    expect(svg.getAttribute("width")).toBe("16");
    expect(svg.getAttribute("height")).toBe("16");
    expect(svg.getAttribute("class")).toContain("text-red");
  });

  it("keeps every shape inside the 24x24 canvas", () => {
    DRAWN_LINE_SHAPES.forEach((shape) => {
      const numbers = shape.d.match(/-?\d+(\.\d+)?/g)!.map(Number);
      numbers.forEach((n) => {
        expect(n).toBeGreaterThanOrEqual(0);
        expect(n).toBeLessThanOrEqual(24);
      });
    });
  });
});
