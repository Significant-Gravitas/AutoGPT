import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { CustomNode, CustomNodeData } from "./FlowEditor/nodes/CustomNode";
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
    uiType: block.uiType as BlockUIType,
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
