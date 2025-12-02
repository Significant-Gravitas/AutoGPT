import { CustomNode } from "../../nodes/CustomNode/CustomNode";
import Flatbush from "flatbush";

export type CollisionAlgorithmOptions = {
  maxIterations: number;
  overlapThreshold: number;
  margin: number;
};

export type CollisionAlgorithm = (
  nodes: CustomNode[],
  options: CollisionAlgorithmOptions,
) => CustomNode[];

type Box = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  id: string;
  moved: boolean;
  x: number;
  y: number;
  width: number;
  height: number;
  node: CustomNode;
};

function rebuildFlatbush(boxes: Box[]) {
  const index = new Flatbush(boxes.length);
  for (const box of boxes) {
    index.add(box.minX, box.minY, box.maxX, box.maxY);
  }
  index.finish();
  return index;
}

export const resolveCollisions: CollisionAlgorithm = (
  nodes,
  { maxIterations = 50, overlapThreshold = 0.5, margin = 0 },
) => {
  // Create boxes from nodes
  const boxes: Box[] = new Array(nodes.length);

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    // Use measured dimensions if available, otherwise use defaults
    const width = (node.width ?? node.measured?.width ?? 0) + margin * 2;
    const height = (node.height ?? node.measured?.height ?? 0) + margin * 2;

    console.log("width", width);
    console.log("height", height);
    const x = node.position.x - margin;
    const y = node.position.y - margin;

    const box: Box = {
      minX: x,
      minY: y,
      maxX: x + width,
      maxY: y + height,
      id: node.id,
      moved: false,
      x,
      y,
      width,
      height,
      node,
    };

    boxes[i] = box;
  }

  let numIterations = 0;
  let index = rebuildFlatbush(boxes);

  for (let iter = 0; iter <= maxIterations; iter++) {
    let moved = false;

    // For each box, find potential collisions using spatial search
    for (let i = 0; i < boxes.length; i++) {
      const A = boxes[i];
      // Search for boxes that might overlap with A
      const candidateIndices = index.search(A.minX, A.minY, A.maxX, A.maxY);

      for (const j of candidateIndices) {
        const B = boxes[j];
        // Skip self
        if (A.id === B.id) continue;

        // Calculate center positions
        const centerAX = A.x + A.width * 0.5;
        const centerAY = A.y + A.height * 0.5;
        const centerBX = B.x + B.width * 0.5;
        const centerBY = B.y + B.height * 0.5;

        // Calculate distance between centers
        const dx = centerAX - centerBX;
        const dy = centerAY - centerBY;

        // Calculate overlap along each axis
        const px = (A.width + B.width) * 0.5 - Math.abs(dx);
        const py = (A.height + B.height) * 0.5 - Math.abs(dy);

        // Check if there's significant overlap
        if (px > overlapThreshold && py > overlapThreshold) {
          A.moved = B.moved = moved = true;

          // Resolve along the smallest overlap axis
          if (px < py) {
            // Move along x-axis
            const sx = dx > 0 ? 1 : -1;
            const moveAmount = (px / 2) * sx;

            A.x += moveAmount;
            A.minX += moveAmount;
            A.maxX += moveAmount;
            B.x -= moveAmount;
            B.minX -= moveAmount;
            B.maxX -= moveAmount;
          } else {
            // Move along y-axis
            const sy = dy > 0 ? 1 : -1;
            const moveAmount = (py / 2) * sy;

            A.y += moveAmount;
            A.minY += moveAmount;
            A.maxY += moveAmount;
            B.y -= moveAmount;
            B.minY -= moveAmount;
            B.maxY -= moveAmount;
          }
        }
      }
    }

    numIterations = numIterations + 1;

    // Early exit if no overlaps were found
    if (!moved) {
      break;
    }

    index = rebuildFlatbush(boxes);
  }

  const newNodes = boxes.map((box) => {
    if (box.moved) {
      return {
        ...box.node,
        position: {
          x: box.x + margin,
          y: box.y + margin,
        },
      };
    }
    return box.node;
  });

  return newNodes;
};
