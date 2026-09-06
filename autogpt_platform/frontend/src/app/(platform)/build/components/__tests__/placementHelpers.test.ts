import { describe, expect, it } from "vitest";
import {
  findFreePosition,
  getFlowViewportBounds,
  getNodeDimensions,
  type ExistingNodeForPlacement,
} from "../placementHelpers";

const viewport = getFlowViewportBounds({ x: 0, y: 0, zoom: 1 }, 1000, 800, 0);

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

    expect(position).toEqual({ x: 30, y: 1260 });
  });

  it("uses adjacent placement without viewport bounds when canvas is not crowded", () => {
    expect(findFreePosition([node(200, 150)], 400, 30)).toEqual({
      x: 730,
      y: 150,
    });
  });

  it("uses default dimensions for nodes without measured property", () => {
    const unmeasuredNode: ExistingNodeForPlacement = {
      position: { x: 0, y: 0 },
    };
    const position = findFreePosition([unmeasuredNode], 400, 30);

    expect(position).toEqual({ x: 530, y: 0 });
  });

  it("picks an adjacent candidate visible in the viewport", () => {
    const nodes = [node(0, 0, 400, 400)];
    const smallViewport = getFlowViewportBounds(
      { x: 0, y: 0, zoom: 1 },
      900,
      900,
      0,
    );
    const position = findFreePosition(nodes, 400, 30, smallViewport, 400);

    expect(position.x + 400).toBeLessThanOrEqual(smallViewport.maxX);
    expect(position.y + 400).toBeLessThanOrEqual(smallViewport.maxY);
  });

  it("falls back to last-node position when no viewport and all adjacent spots blocked", () => {
    const tightCluster = [
      node(0, 0, 400, 400),
      node(430, 0, 400, 400),
      node(-430, 0, 400, 400),
      node(0, 430, 400, 400),
    ];

    const position = findFreePosition(tightCluster, 400, 30, undefined, 400);

    expect(position.x).toBeGreaterThanOrEqual(0);
  });

  it("finds adjacent gap when grid scan misses it due to alignment", () => {
    const smallNode = node(0, 0, 100, 100);
    const narrowViewport = getFlowViewportBounds(
      { x: 0, y: 0, zoom: 1 },
      350,
      350,
      0,
    );
    const position = findFreePosition(
      [smallNode],
      200,
      30,
      narrowViewport,
      200,
    );

    expect(position).toEqual({ x: 130, y: 0 });
  });
});

describe("getNodeDimensions", () => {
  it("returns measured dimensions when available", () => {
    const dims = getNodeDimensions({ measured: { width: 300, height: 200 } });
    expect(dims).toEqual({ width: 300, height: 200 });
  });

  it("prefers node.width over measured width", () => {
    const dims = getNodeDimensions({
      width: 600,
      measured: { width: 300, height: 200 },
    });
    expect(dims.width).toBe(600);
  });

  it("falls back to default dimensions for unmeasured nodes", () => {
    const dims = getNodeDimensions({});
    expect(dims).toEqual({ width: 500, height: 400 });
  });

  it("uses custom fallback width when provided", () => {
    const dims = getNodeDimensions({}, 300);
    expect(dims.width).toBe(300);
  });
});
