import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { PixelGridLoader } from "../PixelGridLoader";

function renderGrid(props: React.ComponentProps<typeof PixelGridLoader> = {}) {
  const { container } = render(<PixelGridLoader {...props} />);
  const grid = container.firstElementChild as HTMLElement;
  const cells = Array.from(grid.children) as HTMLElement[];
  return { grid, cells };
}

afterEach(cleanup);

describe("PixelGridLoader", () => {
  it("renders a hidden 3x3 grid with staggered chevron delays by default", () => {
    const { grid, cells } = renderGrid();

    expect(grid.getAttribute("aria-hidden")).toBe("true");
    expect(cells).toHaveLength(9);
    expect(grid.style.gridTemplateColumns).toBe("repeat(3, 4px)");

    // Chevron wavefront: (column + |row - 1|) * 90ms.
    expect(cells[0].style.animationDelay).toBe("90ms");
    expect(cells[4].style.animationDelay).toBe("90ms");
    expect(cells[8].style.animationDelay).toBe("270ms");
    cells.forEach((cell) => {
      expect(cell.style.opacity).toBe("0.15");
      expect(cell.style.animationDuration).toBe("650ms");
      expect(cell.className).toContain("rounded-[1px]");
    });
  });

  it("rounds the cells in the dots variant", () => {
    const { cells } = renderGrid({ variant: "dots" });
    cells.forEach((cell) => {
      expect(cell.className).toContain("rounded-full");
    });
  });

  it("keeps the centre cell dark and unanimated in the orbit variant", () => {
    const { cells } = renderGrid({ variant: "orbit" });

    expect(cells[4].style.opacity).toBe("0.07");
    expect(cells[4].style.animationDelay).toBe("0ms");

    // Comet order [0, 1, 2, 5, 8, 7, 6, 3] at 110ms per step.
    expect(cells[0].style.animationDelay).toBe("0ms");
    expect(cells[1].style.animationDelay).toBe("110ms");
    expect(cells[5].style.animationDelay).toBe("330ms");
    expect(cells[3].style.animationDelay).toBe("770ms");
    expect(cells[0].style.opacity).toBe("0.15");
    expect(cells[0].style.animationDuration).toBe("950ms");
  });

  it("scales the cells and gap with cellSize and forwards className", () => {
    const { grid, cells } = renderGrid({ cellSize: 8, className: "text-red" });

    expect(grid.className).toContain("text-red");
    expect(grid.style.gridTemplateColumns).toBe("repeat(3, 8px)");
    expect(grid.style.gap).toBe("3px");
    expect(cells[0].style.width).toBe("8px");
    expect(cells[0].style.height).toBe("8px");
  });
});
