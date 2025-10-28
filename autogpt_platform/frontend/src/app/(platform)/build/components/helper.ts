import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import {
  CustomNode,
  CustomNodeData,
} from "./FlowEditor/nodes/CustomNode/CustomNode";
import { BlockUIType } from "./types";
import { NodeModel } from "@/app/api/__generated__/models/nodeModel";
import { NodeModelMetadata } from "@/app/api/__generated__/models/nodeModelMetadata";

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
    data: customNodeData,
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
