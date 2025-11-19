import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import {
  CustomNode,
  CustomNodeData,
} from "./FlowEditor/nodes/CustomNode/CustomNode";
import { BlockUIType } from "./types";
import { NodeModel } from "@/app/api/__generated__/models/nodeModel";
import { NodeModelMetadata } from "@/app/api/__generated__/models/nodeModelMetadata";
import { Link } from "@/app/api/__generated__/models/link";
import { CustomEdge } from "./FlowEditor/edges/CustomEdge";
import { XYPosition } from "@xyflow/react";

export const convertBlockInfoIntoCustomNodeData = (
  block: BlockInfo,
  hardcodedValues: Record<string, any> = {},
) => {
  const customNodeData: CustomNodeData = {
    hardcodedValues: hardcodedValues,
    title: block.name,
    description: block.description,
    inputSchema: block.inputSchema,
    outputSchema: block.outputSchema,
    categories: block.categories,
    uiType: block.uiType as BlockUIType,
    staticOutput: block.staticOutput,
    block_id: block.id,
    costs: block.costs,
  };
  return customNodeData;
};

export const convertNodesPlusBlockInfoIntoCustomNodes = (
  node: NodeModel,
  block: BlockInfo,
) => {
  const customNodeData = convertBlockInfoIntoCustomNodeData(
    block,
    node.input_default,
  );
  const customNode: CustomNode = {
    id: node.id ?? "",
    data: { ...customNodeData, metadata: node.metadata },
    type: "custom",
    position: {
      x:
        (
          (node.metadata as NodeModelMetadata).position as {
            x: number;
            y: number;
          }
        )?.x ?? 0,
      y:
        (
          (node.metadata as NodeModelMetadata).position as {
            x: number;
            y: number;
          }
        )?.y ?? 0,
    },
  };
  return customNode;
};

export const linkToCustomEdge = (link: Link): CustomEdge => ({
  id: link.id ?? "",
  type: "custom" as const,
  source: link.source_id,
  target: link.sink_id,
  sourceHandle: link.source_name,
  targetHandle: link.sink_name,
  data: {
    isStatic: link.is_static,
  },
});

export const customEdgeToLink = (edge: CustomEdge): Link => ({
  id: edge.id || undefined,
  source_id: edge.source,
  sink_id: edge.target,
  source_name: edge.sourceHandle || "",
  sink_name: edge.targetHandle || "",
  is_static: edge.data?.isStatic,
});

export enum BlockCategory {
  AI = "AI",
  SOCIAL = "SOCIAL",
  TEXT = "TEXT",
  SEARCH = "SEARCH",
  BASIC = "BASIC",
  INPUT = "INPUT",
  OUTPUT = "OUTPUT",
  LOGIC = "LOGIC",
  COMMUNICATION = "COMMUNICATION",
  DEVELOPER_TOOLS = "DEVELOPER_TOOLS",
  DATA = "DATA",
  HARDWARE = "HARDWARE",
  AGENT = "AGENT",
  CRM = "CRM",
  SAFETY = "SAFETY",
  PRODUCTIVITY = "PRODUCTIVITY",
  ISSUE_TRACKING = "ISSUE_TRACKING",
  MULTIMEDIA = "MULTIMEDIA",
  MARKETING = "MARKETING",
}

// Cost related helpers
export const isCostFilterMatch = (
  costFilter: any,
  inputValues: any,
): boolean => {
  return typeof costFilter === "object" && typeof inputValues === "object"
    ? Object.entries(costFilter).every(
        ([k, v]) =>
          (!v && !inputValues[k]) || isCostFilterMatch(v, inputValues[k]),
      )
    : costFilter === inputValues;
};

// ----- Position related helpers -----

export interface NodeDimensions {
  x: number;
  y: number;
  width: number;
  height: number;
}

function rectanglesOverlap(
  rect1: NodeDimensions,
  rect2: NodeDimensions,
): boolean {
  const x1 = rect1.x,
    y1 = rect1.y,
    w1 = rect1.width,
    h1 = rect1.height;
  const x2 = rect2.x,
    y2 = rect2.y,
    w2 = rect2.width,
    h2 = rect2.height;

  return !(x1 + w1 <= x2 || x1 >= x2 + w2 || y1 + h1 <= y2 || y1 >= y2 + h2);
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
    return { x: 100, y: 100 }; // Default starting position
  }

  // Start from the most recently added node
  for (let i = existingNodes.length - 1; i >= 0; i--) {
    const lastNode = existingNodes[i];
    const lastNodeWidth = lastNode.measured?.width ?? 500;
    const lastNodeHeight = lastNode.measured?.height ?? 400;

    // Try right
    const candidate = {
      x: lastNode.position.x + lastNodeWidth + margin,
      y: lastNode.position.y,
      width: newNodeWidth,
      height: 400, // Estimated height
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

    // Try left
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

    // Try below
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

  // Fallback: place it far to the right
  const lastNode = existingNodes[existingNodes.length - 1];
  return {
    x: lastNode.position.x + 600,
    y: lastNode.position.y,
  };
}
