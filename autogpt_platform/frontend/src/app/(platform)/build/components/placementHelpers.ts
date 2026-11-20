import { XYPosition } from "@xyflow/react";
import { BlockUIType } from "./types";

export interface NodeDimensions {
  x: number;
  y: number;
  width: number;
  height: number;
}

export type FlowViewportBounds = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

export type ExistingNodeForPlacement = {
  position: XYPosition;
  width?: number;
  height?: number;
  measured?: { width?: number; height?: number };
  data?: { uiType?: string };
};

const DEFAULT_NODE_WIDTH = 350;
const DEFAULT_NODE_HEIGHT = 400;

export function getBlockPlacementDimensions(uiType?: string) {
  return uiType === BlockUIType.NOTE
    ? { width: 304, height: 304 }
    : { width: DEFAULT_NODE_WIDTH, height: DEFAULT_NODE_HEIGHT };
}

function rectanglesOverlap(a: NodeDimensions, b: NodeDimensions): boolean {
  return !(
    a.x + a.width <= b.x ||
    a.x >= b.x + b.width ||
    a.y + a.height <= b.y ||
    a.y >= b.y + b.height
  );
}

function nodeToRect(node: ExistingNodeForPlacement): NodeDimensions {
  return {
    x: node.position.x,
    y: node.position.y,
    ...getNodeDimensions(node),
  };
}

function overlapsAnyNode(
  candidate: NodeDimensions,
  nodes: ExistingNodeForPlacement[],
): boolean {
  return nodes.some((n) => rectanglesOverlap(candidate, nodeToRect(n)));
}

export function fitsInViewport(
  rect: NodeDimensions,
  bounds: FlowViewportBounds,
): boolean {
  return (
    rect.x >= bounds.minX &&
    rect.y >= bounds.minY &&
    rect.x + rect.width <= bounds.maxX &&
    rect.y + rect.height <= bounds.maxY
  );
}

export function getFlowViewportBounds(
  viewport: { x: number; y: number; zoom: number },
  screenWidth: number,
  screenHeight: number,
  padding = 40,
): FlowViewportBounds | undefined {
  const { x, y, zoom } = viewport;
  if (
    ![x, y, zoom, screenWidth, screenHeight, padding].every(Number.isFinite) ||
    zoom <= 0 ||
    screenWidth <= 0 ||
    screenHeight <= 0
  ) {
    return undefined;
  }
  const inset = Math.max(
    0,
    Math.min(padding, screenWidth / 4, screenHeight / 4),
  );
  return {
    minX: (-x + inset) / zoom,
    minY: (-y + inset) / zoom,
    maxX: (screenWidth - x - inset) / zoom,
    maxY: (screenHeight - y - inset) / zoom,
  };
}

function scanViewportGrid(
  nodes: ExistingNodeForPlacement[],
  width: number,
  height: number,
  margin: number,
  bounds: FlowViewportBounds,
): XYPosition | null {
  const stepX = width + margin;
  const stepY = height + margin;
  let attempts = 0;

  for (let y = bounds.minY; y + height <= bounds.maxY; y += stepY) {
    for (let x = bounds.minX; x + width <= bounds.maxX; x += stepX) {
      if (attempts++ >= 1000) return null;
      const candidate: NodeDimensions = { x, y, width, height };
      if (!overlapsAnyNode(candidate, nodes)) {
        return { x, y };
      }
    }
  }

  return null;
}

function findAdjacentPosition(
  nodes: ExistingNodeForPlacement[],
  width: number,
  height: number,
  margin: number,
  bounds?: FlowViewportBounds,
): XYPosition | undefined {
  for (let i = nodes.length - 1; i >= 0; i--) {
    const rect = nodeToRect(nodes[i]);

    const candidates: XYPosition[] = [
      { x: rect.x + rect.width + margin, y: rect.y },
      { x: rect.x - width - margin, y: rect.y },
      { x: rect.x, y: rect.y + rect.height + margin },
    ];

    for (const pos of candidates) {
      const rect = { ...pos, width, height };
      if (
        (!bounds || fitsInViewport(rect, bounds)) &&
        !overlapsAnyNode(rect, nodes)
      ) {
        return pos;
      }
    }
  }

  return undefined;
}

export function getNodeDimensions(node: {
  width?: number;
  height?: number;
  measured?: { width?: number; height?: number };
  data?: { uiType?: string };
}): { width: number; height: number } {
  const fallback = getBlockPlacementDimensions(node.data?.uiType);
  return {
    width: node.width ?? node.measured?.width ?? fallback.width,
    height: node.height ?? node.measured?.height ?? fallback.height,
  };
}

export function findFreePosition(
  existingNodes: ExistingNodeForPlacement[],
  newNodeWidth: number = DEFAULT_NODE_WIDTH,
  margin: number = 60,
  viewportBounds?: FlowViewportBounds,
  newNodeHeight: number = DEFAULT_NODE_HEIGHT,
): XYPosition {
  if (existingNodes.length === 0) {
    if (viewportBounds) {
      return {
        x: Math.max(
          viewportBounds.minX,
          Math.min(
            viewportBounds.minX + margin,
            viewportBounds.maxX - newNodeWidth,
          ),
        ),
        y: Math.max(
          viewportBounds.minY,
          Math.min(
            viewportBounds.minY + margin,
            viewportBounds.maxY - newNodeHeight,
          ),
        ),
      };
    }
    return { x: 100, y: 100 };
  }

  // First try: find an open slot in the visible viewport grid
  if (viewportBounds) {
    const gridSlot = scanViewportGrid(
      existingNodes,
      newNodeWidth,
      newNodeHeight,
      margin,
      viewportBounds,
    );
    if (gridSlot) return gridSlot;
  }

  // Second try: adjacent to existing nodes (right, left, below)
  const adjacent = findAdjacentPosition(
    existingNodes,
    newNodeWidth,
    newNodeHeight,
    margin,
    viewportBounds,
  );
  if (adjacent) return adjacent;

  const lastRect = nodeToRect(existingNodes[existingNodes.length - 1]);
  const bottom = existingNodes.reduce((max, node) => {
    const rect = nodeToRect(node);
    return Math.max(max, rect.y + rect.height);
  }, viewportBounds?.maxY ?? -Infinity);
  return {
    x: viewportBounds ? viewportBounds.minX + margin : lastRect.x,
    y: bottom + margin,
  };
}
