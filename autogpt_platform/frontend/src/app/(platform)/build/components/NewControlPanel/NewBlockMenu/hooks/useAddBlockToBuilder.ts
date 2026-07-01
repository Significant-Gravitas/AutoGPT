import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { useReactFlow, useStore } from "@xyflow/react";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { BlockUIType } from "@/app/(platform)/build/components/types";
import {
  findFreePosition,
  getFlowViewportBounds,
  getNodeDimensions,
} from "@/app/(platform)/build/components/placementHelpers";
import { CustomNode } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/CustomNode";

export function useAddBlockToBuilder() {
  const addBlock = useNodeStore((state) => state.addBlock);
  const nodes = useNodeStore((state) => state.nodes);
  const { getViewport } = useReactFlow();
  const flowWidth = useStore((s) => s.width);
  const flowHeight = useStore((s) => s.height);

  function addBlockWithPlacement(
    block: BlockInfo,
    hardcodedValues?: Record<string, unknown>,
  ): CustomNode {
    const viewport = getViewport();
    const viewportBounds = getFlowViewportBounds(
      viewport,
      flowWidth,
      flowHeight,
    );
    const isNote = block.uiType === BlockUIType.NOTE;
    const newNodeWidth = isNote ? 300 : 400;
    const newNodeHeight = isNote ? 300 : 400;

    const existingNodes = nodes.map((n) => ({
      position: n.position,
      measured: getNodeDimensions(
        n,
        n.data.uiType === BlockUIType.NOTE ? 300 : 500,
      ),
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
