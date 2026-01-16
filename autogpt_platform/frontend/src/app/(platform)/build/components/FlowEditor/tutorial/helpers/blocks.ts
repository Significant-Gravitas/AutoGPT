import { BLOCK_IDS } from "../constants";
import { useNodeStore } from "../../../../stores/nodeStore";
import { getV2GetSpecificBlocks } from "@/app/api/__generated__/endpoints/default/default";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";

const prefetchedBlocks: Map<string, BlockInfo> = new Map();

export const prefetchTutorialBlocks = async (): Promise<void> => {
  try {
    const blockIds = [BLOCK_IDS.CALCULATOR];
    const response = await getV2GetSpecificBlocks({ block_ids: blockIds });

    if (response.status === 200 && response.data) {
      response.data.forEach((block) => {
        prefetchedBlocks.set(block.id, block);
      });
      console.debug("Tutorial blocks prefetched:", prefetchedBlocks.size);
    }
  } catch (error) {
    console.error("Failed to prefetch tutorial blocks:", error);
  }
};

export const getPrefetchedBlock = (blockId: string): BlockInfo | undefined => {
  return prefetchedBlocks.get(blockId);
};

export const clearPrefetchedBlocks = (): void => {
  prefetchedBlocks.clear();
};

export const addPrefetchedBlock = (
  blockId: string,
  position?: { x: number; y: number },
): void => {
  const block = prefetchedBlocks.get(blockId);
  if (block) {
    useNodeStore.getState().addBlock(block, {}, position);
  } else {
    console.error(`Block ${blockId} not found in prefetched blocks`);
  }
};

export const getNodeByBlockId = (blockId: string) => {
  const nodes = useNodeStore.getState().nodes;
  return nodes.find((n) => n.data?.block_id === blockId);
};

export const addSecondCalculatorBlock = (): void => {
  const firstCalculatorNode = getNodeByBlockId(BLOCK_IDS.CALCULATOR);

  if (firstCalculatorNode) {
    const calcX = firstCalculatorNode.position.x;
    const calcY = firstCalculatorNode.position.y;

    addPrefetchedBlock(BLOCK_IDS.CALCULATOR, {
      x: calcX + 500,
      y: calcY,
    });
  } else {
    addPrefetchedBlock(BLOCK_IDS.CALCULATOR);
  }
};

export const getCalculatorNodes = () => {
  const nodes = useNodeStore.getState().nodes;
  return nodes.filter((n) => n.data?.block_id === BLOCK_IDS.CALCULATOR);
};

export const getSecondCalculatorNode = () => {
  const calculatorNodes = getCalculatorNodes();
  return calculatorNodes.length >= 2 ? calculatorNodes[1] : null;
};

export const getFormContainerSelector = (blockId: string): string | null => {
  const node = getNodeByBlockId(blockId);
  if (node) {
    return `[data-id="form-creator-container-${node.id}"]`;
  }
  return null;
};

export const getFormContainerElement = (blockId: string): Element | null => {
  const selector = getFormContainerSelector(blockId);
  if (selector) {
    return document.querySelector(selector);
  }
  return null;
};
