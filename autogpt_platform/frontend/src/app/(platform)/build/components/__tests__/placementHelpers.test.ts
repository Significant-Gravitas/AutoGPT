import { describe, expect, it } from "vitest";
import {
  findFreePosition,
  getFlowViewportBounds,
  type ExistingNodeForPlacement,
} from "../placementHelpers";

const viewport = getFlowViewportBounds(
  { x: 0, y: 0, zoom: 1 },
  1000,
  800,
  0,
);

function node(
  x: number,
  y: number,
  width = 500,
  height = 400,
): ExistingNodeForPlacement {
  return {
    position: { x, y },
    measured: { width, height },
  };
}

describe("getFlowViewportBounds", () => {
  it("maps screen corners to flow coordinates", () => {
    expect(
      getFlowViewportBounds({ x: -100, y: -50, zoom: 0.5 }, 1200, 900, 0),
    ).toEqual({
      minX: 200,
      minY: 100,
      maxX: 2600,
      maxY: 1900,
    });
  });
});

describe("findFreePosition", () => {
  it("returns default origin when there are no nodes", () => {
    expect(findFreePosition([])).toEqual({ x: 100, y: 100 });
  });

  it("places the first block inside the viewport when bounds are provided", () => {
    expect(findFreePosition([], 400, 30, viewport)).toEqual({ x: 30, y: 30 });
  });

  it("places a block to the right of the most recent node when space is available", () => {
    expect(findFreePosition([node(100, 100)], 400, 30)).toEqual({
      x: 630,
      y: 100,
    });
  });

  it("prefers a visible grid slot over an off-screen adjacent slot", () => {
    const crowdedCanvas = [node(5000, 5000)];

    const position = findFreePosition(crowdedCanvas, 400, 30, viewport);

    expect(position.x).toBeGreaterThanOrEqual(viewport.minX);
    expect(position.y).toBeGreaterThanOrEqual(viewport.minY);
    expect(position.x + 400).toBeLessThanOrEqual(viewport.maxX);
    expect(position.y + 400).toBeLessThanOrEqual(viewport.maxY);
  });

  it("does not jump far to the right when the viewport is full", () => {
    const gridNodes: ExistingNodeForPlacement[] = [];
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        gridNodes.push(node(col * 430, row * 430, 400, 400));
      }
    }

    const position = findFreePosition(gridNodes, 400, 30, viewport);

    expect(position).toEqual({ x: 30, y: 830 });
  });

  it("uses adjacent placement without viewport bounds when canvas is not crowded", () => {
    expect(findFreePosition([node(200, 150)], 400, 30)).toEqual({
      x: 730,
      y: 150,
    });
  });
});
