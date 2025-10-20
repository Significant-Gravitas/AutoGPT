import React from "react";
import { Node as XYNode, NodeProps } from "@xyflow/react";
import { RJSFSchema } from "@rjsf/utils";
import { BlockUIType } from "../../../types";
import { StickyNoteBlock } from "./StickyNoteBlock";
import { BlockInfoCategoriesItem } from "@/app/api/__generated__/models/blockInfoCategoriesItem";
import { StandardNodeBlock } from "./StandardNodeBlock";
import { BlockCost } from "@/app/api/__generated__/models/blockCost";

export type CustomNodeData = {
  hardcodedValues: {
    [key: string]: any;
  };
  title: string;
  description: string;
  inputSchema: RJSFSchema;
  outputSchema: RJSFSchema;
  uiType: BlockUIType;
  block_id: string;
  // TODO : We need better type safety for the following backend fields.
  costs: BlockCost[];
  categories: BlockInfoCategoriesItem[];
};

export type CustomNode = XYNode<CustomNodeData, "custom">;

export const CustomNode: React.FC<NodeProps<CustomNode>> = React.memo(
  ({ data, id: nodeId, selected }) => {
    if (data.uiType === BlockUIType.NOTE) {
      return <StickyNoteBlock selected={selected} data={data} id={nodeId} />;
    }

    if (data.uiType === BlockUIType.STANDARD) {
      return (
        <StandardNodeBlock data={data} selected={selected} nodeId={nodeId} />
      );
    }

    return (
      <StandardNodeBlock data={data} selected={selected} nodeId={nodeId} />
    );
  },
);

CustomNode.displayName = "CustomNode";
