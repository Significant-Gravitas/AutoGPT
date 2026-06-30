import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { useReactFlow } from "@xyflow/react";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { BlockUIType } from "@/app/(platform)/build/components/types";
import {
  findFreePosition,
  getFlowViewportBounds,
} from "@/app/(platform)/build/components/placementHelpers";
import { CustomNode } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/CustomNode";

export function useAddBlockToBuilder() {
  const addBlock = useNodeStore((state) => state.addBlock);
  const nodes = useNodeStore((state) => state.nodes);
  const { getViewport } = useReactFlow();

  function addBlockWithPlacement(
    block: BlockInfo,
    hardcodedValues?: Record<string, unknown>,
  ): CustomNode {
    const viewport = getViewport();
    const viewportBounds = getFlowViewportBounds(
      viewport,
      window.innerWidth,
      window.innerHeight,
    );
    const isNote = block.uiType === BlockUIType.NOTE;
    const newNodeWidth = isNote ? 300 : 400;
    const newNodeHeight = isNote ? 300 : 400;

    const existingNodes = nodes.map((node) => ({
      position: node.position,
      measured: {
        width:
          node.width ??
          node.measured?.width ??
          (node.data.uiType === BlockUIType.NOTE ? 300 : 500),
        height: node.height ?? node.measured?.height ?? 400,
      },
    }));

    const position = findFreePosition(
      existingNodes,
      newNodeWidth,
      30,
      viewportBounds,
      newNodeHeight,
    );

    return addBlock(block, hardcodedValues, position);
  }

  return { addBlockWithPlacement };
}
