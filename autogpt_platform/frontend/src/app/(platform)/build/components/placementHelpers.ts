import { XYPosition } from "@xyflow/react";

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
  measured?: { width: number; height: number };
};

const DEFAULT_NODE_WIDTH = 500;
const DEFAULT_NODE_HEIGHT = 400;

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
    width: node.measured?.width ?? DEFAULT_NODE_WIDTH,
    height: node.measured?.height ?? DEFAULT_NODE_HEIGHT,
  };
}

function overlapsAnyNode(
  candidate: NodeDimensions,
  nodes: ExistingNodeForPlacement[],
): boolean {
  return nodes.some((n) => rectanglesOverlap(candidate, nodeToRect(n)));
}

function fitsInViewport(
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
): FlowViewportBounds {
  const { x, y, zoom } = viewport;
  return {
    minX: (-x + padding) / zoom,
    minY: (-y + padding) / zoom,
    maxX: (screenWidth - x - padding) / zoom,
    maxY: (screenHeight - y - padding) / zoom,
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

  for (let y = bounds.minY; y + height <= bounds.maxY; y += stepY) {
    for (let x = bounds.minX; x + width <= bounds.maxX; x += stepX) {
      const candidate: NodeDimensions = { x, y, width, height };
      if (!overlapsAnyNode(candidate, nodes)) {
        return { x, y };
      }
    }
  }

  return null;
}

function getAdjacentPositions(
  nodes: ExistingNodeForPlacement[],
  width: number,
  height: number,
  margin: number,
): XYPosition[] {
  const positions: XYPosition[] = [];

  for (let i = nodes.length - 1; i >= 0; i--) {
    const rect = nodeToRect(nodes[i]);

    const candidates: XYPosition[] = [
      { x: rect.x + rect.width + margin, y: rect.y },
      { x: rect.x - width - margin, y: rect.y },
      { x: rect.x, y: rect.y + rect.height + margin },
    ];

    for (const pos of candidates) {
      if (!overlapsAnyNode({ ...pos, width, height }, nodes)) {
        positions.push(pos);
      }
    }
  }

  return positions;
}

export function getNodeDimensions(
  node: {
    width?: number;
    height?: number;
    measured?: { width?: number; height?: number };
  },
  fallbackWidth = DEFAULT_NODE_WIDTH,
): { width: number; height: number } {
  return {
    width: node.width ?? node.measured?.width ?? fallbackWidth,
    height: node.height ?? node.measured?.height ?? DEFAULT_NODE_HEIGHT,
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
        x: viewportBounds.minX + margin,
        y: viewportBounds.minY + margin,
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
  const adjacent = getAdjacentPositions(
    existingNodes,
    newNodeWidth,
    newNodeHeight,
    margin,
  );

  if (viewportBounds && adjacent.length > 0) {
    const visibleAdj = adjacent.find((pos) =>
      fitsInViewport(
        { ...pos, width: newNodeWidth, height: newNodeHeight },
        viewportBounds,
      ),
    );
    if (visibleAdj) return visibleAdj;
  } else if (adjacent.length > 0) {
    return adjacent[0];
  }

  // Last resort: scan below the viewport
  if (viewportBounds) {
    const x = viewportBounds.minX + margin;
    let y = viewportBounds.maxY + margin;

    while (
      overlapsAnyNode(
        { x, y, width: newNodeWidth, height: newNodeHeight },
        existingNodes,
      )
    ) {
      y += newNodeHeight + margin;
    }

    return { x, y };
  }

  const lastRect = nodeToRect(existingNodes[existingNodes.length - 1]);
  return {
    x: lastRect.x + lastRect.width + margin,
    y: lastRect.y,
  };
}
