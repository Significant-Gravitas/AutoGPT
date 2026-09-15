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

const NOTE_SIZE = 300;
const BLOCK_SIZE = 400;
const DEFAULT_MEASURED_WIDTH = 500;
const PLACEMENT_MARGIN = 30;

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
    const viewportBounds = getFlowViewportBounds(
      getViewport(),
      flowWidth,
      flowHeight,
    );

    const isNote = block.uiType === BlockUIType.NOTE;
    const width = isNote ? NOTE_SIZE : BLOCK_SIZE;
    const height = isNote ? NOTE_SIZE : BLOCK_SIZE;

    const existingNodes = nodes.map((n) => ({
      position: n.position,
      measured: getNodeDimensions(
        n,
        n.data.uiType === BlockUIType.NOTE ? NOTE_SIZE : DEFAULT_MEASURED_WIDTH,
      ),
    }));

    const position = findFreePosition(
      existingNodes,
      width,
      PLACEMENT_MARGIN,
      viewportBounds,
      height,
    );

    return addBlock(block, hardcodedValues, position);
  }

  return { addBlockWithPlacement };
}
