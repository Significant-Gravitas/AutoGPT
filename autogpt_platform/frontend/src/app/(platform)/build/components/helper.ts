import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { CustomNodeData } from "./FlowEditor/nodes/CustomNode";

export const convertBlockInfoIntoCustomNodeData = (block: BlockInfo) => {
  const customNodeData: CustomNodeData = {
    hardcodedValues: {},
    title: block.name,
    description: block.description,
    inputSchema: block.inputSchema,
    outputSchema: block.outputSchema,
  };
  return customNodeData;
};
