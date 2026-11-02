import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { useReactFlow, useStoreApi } from "@xyflow/react";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";
import {
  findFreePosition,
  fitsInViewport,
  getBlockPlacementDimensions,
  getFlowViewportBounds,
} from "@/app/(platform)/build/components/placementHelpers";
import { CustomNode } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/CustomNode";

const PLACEMENT_MARGIN = 30;

export function useAddBlockToBuilder() {
  const { getViewport, fitView } = useReactFlow();
  const flowStore = useStoreApi();

  function addBlockWithPlacement(
    block: BlockInfo,
    hardcodedValues?: Record<string, unknown>,
  ): CustomNode {
    const { width: flowWidth, height: flowHeight } = flowStore.getState();
    const viewportBounds = getFlowViewportBounds(
      getViewport(),
      flowWidth,
      flowHeight,
    );
    const dimensions = getBlockPlacementDimensions(block.uiType);
    const { nodes, addBlock } = useNodeStore.getState();
    const position = findFreePosition(
      nodes,
      dimensions.width,
      PLACEMENT_MARGIN,
      viewportBounds,
      dimensions.height,
    );
    const node = addBlock(block, hardcodedValues, position);

    if (
      viewportBounds &&
      !fitsInViewport({ ...position, ...dimensions }, viewportBounds)
    ) {
      function handleShowBlock() {
        if (
          !useNodeStore.getState().nodes.some((item) => item.id === node.id)
        ) {
          return;
        }
        void fitView({
          nodes: [{ id: node.id }],
          maxZoom: getViewport().zoom,
          duration: 300,
          padding: 0.2,
        });
        notification.dismiss();
      }
      const notification = toast({
        title: "Block added outside the current view",
        action: (
          <Button variant="secondary" size="small" onClick={handleShowBlock}>
            Show block
          </Button>
        ),
      });
    }

    return node;
  }

  return { addBlockWithPlacement };
}
