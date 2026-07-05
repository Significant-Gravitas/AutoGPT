import { XYPosition } from "@xyflow/react";

export interface NodeDimensions {
  x: number;
  y: number;
  width: number;
  height: number;
}

function rectanglesOverlap(a: NodeDimensions, b: NodeDimensions): boolean {
  return !(
    a.x + a.width <= b.x ||
    a.x >= b.x + b.width ||
    a.y + a.height <= b.y ||
    a.y >= b.y + b.height
  );
}

export function findFreePosition(
  existingNodes: Array<{
    position: XYPosition;
    measured?: { width: number; height: number };
  }>,
  newNodeWidth: number = 500,
  margin: number = 60,
): XYPosition {
  if (existingNodes.length === 0) {
    return { x: 100, y: 100 };
  }

  for (let i = existingNodes.length - 1; i >= 0; i--) {
    const lastNode = existingNodes[i];
    const lastNodeWidth = lastNode.measured?.width ?? 500;
    const lastNodeHeight = lastNode.measured?.height ?? 400;

    const candidate = {
      x: lastNode.position.x + lastNodeWidth + margin,
      y: lastNode.position.y,
      width: newNodeWidth,
      height: 400,
    };

    if (
      !existingNodes.some((n) =>
        rectanglesOverlap(candidate, {
          x: n.position.x,
          y: n.position.y,
          width: n.measured?.width ?? 500,
          height: n.measured?.height ?? 400,
        }),
      )
    ) {
      return { x: candidate.x, y: candidate.y };
    }

    candidate.x = lastNode.position.x - newNodeWidth - margin;
    if (
      !existingNodes.some((n) =>
        rectanglesOverlap(candidate, {
          x: n.position.x,
          y: n.position.y,
          width: n.measured?.width ?? 500,
          height: n.measured?.height ?? 400,
        }),
      )
    ) {
      return { x: candidate.x, y: candidate.y };
    }

    candidate.x = lastNode.position.x;
    candidate.y = lastNode.position.y + lastNodeHeight + margin;
    if (
      !existingNodes.some((n) =>
        rectanglesOverlap(candidate, {
          x: n.position.x,
          y: n.position.y,
          width: n.measured?.width ?? 500,
          height: n.measured?.height ?? 400,
        }),
      )
    ) {
      return { x: candidate.x, y: candidate.y };
    }
  }

  const lastNode = existingNodes[existingNodes.length - 1];
  return {
    x: lastNode.position.x + 600,
    y: lastNode.position.y,
  };
}
