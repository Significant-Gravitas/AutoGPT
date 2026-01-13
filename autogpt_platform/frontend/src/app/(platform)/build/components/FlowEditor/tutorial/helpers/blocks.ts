/**
 * Block-related helpers for the tutorial
 */

import { BLOCK_IDS } from "../constants";
import { useNodeStore } from "../../../../stores/nodeStore";
import { getV2GetSpecificBlocks } from "@/app/api/__generated__/endpoints/default/default";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";

// Cache for prefetched blocks
let prefetchedBlocks: Map<string, BlockInfo> = new Map();

/**
 * Prefetches Calculator block at tutorial start
 * Call this when the tutorial is initialized
 */
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

/**
 * Gets a prefetched block by ID
 */
export const getPrefetchedBlock = (blockId: string): BlockInfo | undefined => {
  return prefetchedBlocks.get(blockId);
};

/**
 * Clears the prefetched blocks cache
 */
export const clearPrefetchedBlocks = (): void => {
  prefetchedBlocks.clear();
};

/**
 * Adds a prefetched block to the canvas at a specific position
 */
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

/**
 * Gets a node by its block_id
 */
export const getNodeByBlockId = (blockId: string) => {
  const nodes = useNodeStore.getState().nodes;
  return nodes.find((n) => n.data?.block_id === blockId);
};

/**
 * Adds a second Calculator block positioned to the right of the first Calculator
 */
export const addSecondCalculatorBlock = (): void => {
  // Find the first Calculator node to position relative to it
  const firstCalculatorNode = getNodeByBlockId(BLOCK_IDS.CALCULATOR);

  if (firstCalculatorNode) {
    const calcX = firstCalculatorNode.position.x;
    const calcY = firstCalculatorNode.position.y;

    // Second Calculator: 500px to the right of first Calculator
    addPrefetchedBlock(BLOCK_IDS.CALCULATOR, {
      x: calcX + 500,
      y: calcY,
    });
  } else {
    // Fallback: Add without specific positioning if first Calculator not found
    addPrefetchedBlock(BLOCK_IDS.CALCULATOR);
  }
};

/**
 * Gets all Calculator nodes on the canvas
 */
export const getCalculatorNodes = () => {
  const nodes = useNodeStore.getState().nodes;
  return nodes.filter((n) => n.data?.block_id === BLOCK_IDS.CALCULATOR);
};

/**
 * Gets the second Calculator node (if exists)
 */
export const getSecondCalculatorNode = () => {
  const calculatorNodes = getCalculatorNodes();
  return calculatorNodes.length >= 2 ? calculatorNodes[1] : null;
};

/**
 * Gets the form container selector for a specific block
 */
export const getFormContainerSelector = (blockId: string): string | null => {
  const node = getNodeByBlockId(blockId);
  if (node) {
    return `[data-id="form-creator-container-${node.id}"]`;
  }
  return null;
};

/**
 * Gets the form container element for a specific block
 */
export const getFormContainerElement = (blockId: string): Element | null => {
  const selector = getFormContainerSelector(blockId);
  if (selector) {
    return document.querySelector(selector);
  }
  return null;
};

