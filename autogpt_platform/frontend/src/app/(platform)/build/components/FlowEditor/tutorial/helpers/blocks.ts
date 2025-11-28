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
 * Prefetches Agent Input and Agent Output blocks at tutorial start
 * Call this when the tutorial is initialized
 */
export const prefetchTutorialBlocks = async (): Promise<void> => {
  try {
    const blockIds = [BLOCK_IDS.AGENT_INPUT, BLOCK_IDS.AGENT_OUTPUT];
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
 * Adds Agent Input and Agent Output blocks positioned relative to Calculator
 * Agent Input: Left side of Calculator
 * Agent Output: Right side of Calculator
 */
export const addAgentIOBlocks = (): void => {
  // Find the Calculator node to position relative to it
  const calculatorNode = getNodeByBlockId(BLOCK_IDS.CALCULATOR);

  if (calculatorNode) {
    const calcX = calculatorNode.position.x;
    const calcY = calculatorNode.position.y;

    // Agent Input: 600px to the left of Calculator
    addPrefetchedBlock(BLOCK_IDS.AGENT_INPUT, {
      x: calcX - 600,
      y: calcY,
    });

    // Agent Output: 600px to the right of Calculator
    addPrefetchedBlock(BLOCK_IDS.AGENT_OUTPUT, {
      x: calcX + 600,
      y: calcY,
    });
  } else {
    // Fallback: Add without specific positioning if Calculator not found
    addPrefetchedBlock(BLOCK_IDS.AGENT_INPUT);
    addPrefetchedBlock(BLOCK_IDS.AGENT_OUTPUT);
  }
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

