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

function rectanglesOverlap(
  rect1: NodeDimensions,
  rect2: NodeDimensions,
): boolean {
  return !(
    rect1.x + rect1.width <= rect2.x ||
    rect1.x >= rect2.x + rect2.width ||
    rect1.y + rect1.height <= rect2.y ||
    rect1.y >= rect2.y + rect2.height
  );
}

function toNodeRect(node: ExistingNodeForPlacement): NodeDimensions {
  return {
    x: node.position.x,
    y: node.position.y,
    width: node.measured?.width ?? DEFAULT_NODE_WIDTH,
    height: node.measured?.height ?? DEFAULT_NODE_HEIGHT,
  };
}

function candidateOverlapsAny(
  candidate: NodeDimensions,
  existingNodes: ExistingNodeForPlacement[],
): boolean {
  return existingNodes.some((node) =>
    rectanglesOverlap(candidate, toNodeRect(node)),
  );
}

function isInsideViewport(
  candidate: NodeDimensions,
  viewport: FlowViewportBounds,
): boolean {
  return (
    candidate.x >= viewport.minX &&
    candidate.y >= viewport.minY &&
    candidate.x + candidate.width <= viewport.maxX &&
    candidate.y + candidate.height <= viewport.maxY
  );
}

export function getFlowViewportBounds(
  viewport: { x: number; y: number; zoom: number },
  width: number,
  height: number,
  padding = 40,
): FlowViewportBounds {
  return {
    minX: (-viewport.x + padding) / viewport.zoom,
    minY: (-viewport.y + padding) / viewport.zoom,
    maxX: (width - viewport.x - padding) / viewport.zoom,
    maxY: (height - viewport.y - padding) / viewport.zoom,
  };
}

function findFreePositionInViewport(
  existingNodes: ExistingNodeForPlacement[],
  newNodeWidth: number,
  newNodeHeight: number,
  margin: number,
  viewport: FlowViewportBounds,
): XYPosition | null {
  const stepX = newNodeWidth + margin;
  const stepY = newNodeHeight + margin;

  for (
    let y = viewport.minY;
    y + newNodeHeight <= viewport.maxY;
    y += stepY
  ) {
    for (
      let x = viewport.minX;
      x + newNodeWidth <= viewport.maxX;
      x += stepX
    ) {
      const candidate: NodeDimensions = {
        x,
        y,
        width: newNodeWidth,
        height: newNodeHeight,
      };

      if (!candidateOverlapsAny(candidate, existingNodes)) {
        return { x, y };
      }
    }
  }

  return null;
}

function collectAdjacentCandidates(
  existingNodes: ExistingNodeForPlacement[],
  newNodeWidth: number,
  newNodeHeight: number,
  margin: number,
): XYPosition[] {
  const candidates: XYPosition[] = [];

  for (let i = existingNodes.length - 1; i >= 0; i--) {
    const lastRect = toNodeRect(existingNodes[i]);
    const offsets = [
      { x: lastRect.x + lastRect.width + margin, y: lastRect.y },
      { x: lastRect.x - newNodeWidth - margin, y: lastRect.y },
      { x: lastRect.x, y: lastRect.y + lastRect.height + margin },
    ];

    for (const offset of offsets) {
      const candidate: NodeDimensions = {
        x: offset.x,
        y: offset.y,
        width: newNodeWidth,
        height: newNodeHeight,
      };

      if (!candidateOverlapsAny(candidate, existingNodes)) {
        candidates.push({ x: offset.x, y: offset.y });
      }
    }
  }

  return candidates;
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

  if (viewportBounds) {
    const inViewport = findFreePositionInViewport(
      existingNodes,
      newNodeWidth,
      newNodeHeight,
      margin,
      viewportBounds,
    );
    if (inViewport) {
      return inViewport;
    }
  }

  const adjacentCandidates = collectAdjacentCandidates(
    existingNodes,
    newNodeWidth,
    newNodeHeight,
    margin,
  );

  if (viewportBounds && adjacentCandidates.length > 0) {
    for (const candidate of adjacentCandidates) {
      if (
        isInsideViewport(
          {
            ...candidate,
            width: newNodeWidth,
            height: newNodeHeight,
          },
          viewportBounds,
        )
      ) {
        return candidate;
      }
    }
  } else if (adjacentCandidates.length > 0) {
    return adjacentCandidates[0];
  }

  if (viewportBounds) {
    return {
      x: viewportBounds.minX + margin,
      y: viewportBounds.maxY + margin,
    };
  }

  const lastRect = toNodeRect(existingNodes[existingNodes.length - 1]);
  return {
    x: lastRect.x + lastRect.width + margin,
    y: lastRect.y,
  };
}
