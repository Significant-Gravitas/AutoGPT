import {
  CSS_CLASSES,
  TUTORIAL_SELECTORS,
  TUTORIAL_CONFIG,
  BLOCK_IDS,
} from "./constants";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import { useNodeStore } from "../../../stores/nodeStore";
import { useEdgeStore } from "../../../stores/edgeStore";
import { Key, storage } from "@/services/storage/local-storage";
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
 * Gets a node by its block_id
 */
export const getNodeByBlockId = (blockId: string) => {
  const nodes = useNodeStore.getState().nodes;
  return nodes.find((n) => n.data?.block_id === blockId);
};

/**
 * Checks if a specific connection exists between two nodes
 */
export const isConnectionMade = (
  sourceBlockId: string,
  targetBlockId: string,
): boolean => {
  const edges = useEdgeStore.getState().edges;
  const nodes = useNodeStore.getState().nodes;

  const sourceNode = nodes.find((n) => n.data?.block_id === sourceBlockId);
  const targetNode = nodes.find((n) => n.data?.block_id === targetBlockId);

  if (!sourceNode || !targetNode) return false;

  return edges.some((edge) => {
    return edge.source === sourceNode.id && edge.target === targetNode.id;
  });
};

/**
 * Waits for a specific number of nodes on canvas
 */
export const waitForNodesCount = (
  count: number,
  timeout = 10000,
): Promise<boolean> => {
  return new Promise((resolve) => {
    const startTime = Date.now();

    const checkNodes = () => {
      const currentCount = useNodeStore.getState().nodes.length;
      if (currentCount >= count) {
        resolve(true);
      } else if (Date.now() - startTime > timeout) {
        resolve(false);
      } else {
        setTimeout(checkNodes, TUTORIAL_CONFIG.ELEMENT_CHECK_INTERVAL);
      }
    };
    checkNodes();
  });
};

/**
 * Waits for an element to appear in the DOM
 */
export const waitForElement = (
  selector: string,
  timeout = 10000,
): Promise<Element> => {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();

    const checkElement = () => {
      const element = document.querySelector(selector);
      if (element) {
        resolve(element);
      } else if (Date.now() - startTime > timeout) {
        reject(new Error(`Element ${selector} not found within ${timeout}ms`));
      } else {
        setTimeout(checkElement, TUTORIAL_CONFIG.ELEMENT_CHECK_INTERVAL);
      }
    };
    checkElement();
  });
};

/**
 * Waits for an input to contain a specific value (case-insensitive, partial match)
 */
export const waitForInputValue = (
  selector: string,
  targetValue: string,
  timeout = 30000,
): Promise<void> => {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();

    const checkInput = () => {
      const input = document.querySelector(selector) as HTMLInputElement;
      if (input) {
        const currentValue = input.value.toLowerCase().trim();
        const target = targetValue.toLowerCase().trim();

        if (currentValue.includes(target) || target.includes(currentValue)) {
          // Check if user has typed enough characters (at least 4 chars or the full string)
          if (currentValue.length >= 4 || currentValue === target) {
            resolve();
            return;
          }
        }
      }

      if (Date.now() - startTime > timeout) {
        resolve(); // Don't reject, just continue after timeout
      } else {
        setTimeout(checkInput, TUTORIAL_CONFIG.INPUT_CHECK_INTERVAL);
      }
    };
    checkInput();
  });
};

/**
 * Waits for a specific element to appear in the search results
 */
export const waitForSearchResult = (
  selector: string,
  timeout = 15000,
): Promise<Element | null> => {
  return new Promise((resolve) => {
    const startTime = Date.now();

    const checkResult = () => {
      const element = document.querySelector(selector);
      if (element) {
        resolve(element);
      } else if (Date.now() - startTime > timeout) {
        resolve(null); // Don't reject, just return null
      } else {
        setTimeout(checkResult, TUTORIAL_CONFIG.ELEMENT_CHECK_INTERVAL);
      }
    };
    checkResult();
  });
};

/**
 * Waits for a node to appear on the canvas using nodeStore
 */
export const waitForNodeOnCanvas = (
  timeout = 10000,
): Promise<Element | null> => {
  return new Promise((resolve) => {
    const startTime = Date.now();

    const checkNode = () => {
      // First check nodeStore
      const storeNodes = useNodeStore.getState().nodes;
      if (storeNodes.length > 0) {
        // Node exists in store, now wait for DOM element
        const domNode = document.querySelector(
          TUTORIAL_SELECTORS.REACT_FLOW_NODE,
        );
        if (domNode) {
          resolve(domNode);
          return;
        }
      }

      if (Date.now() - startTime > timeout) {
        resolve(null);
      } else {
        setTimeout(checkNode, TUTORIAL_CONFIG.ELEMENT_CHECK_INTERVAL);
      }
    };
    checkNode();
  });
};

/**
 * Gets the count of nodes on canvas from nodeStore
 */
export const getNodesCount = (): number => {
  return useNodeStore.getState().nodes.length;
};

/**
 * Gets the first node from nodeStore
 */
export const getFirstNode = () => {
  const nodes = useNodeStore.getState().nodes;
  return nodes.length > 0 ? nodes[0] : null;
};

/**
 * Gets a node by ID from nodeStore
 */
export const getNodeById = (nodeId: string) => {
  const nodes = useNodeStore.getState().nodes;
  return nodes.find((n) => n.id === nodeId);
};

/**
 * Checks if a node has hardcoded values set
 */
export const nodeHasValues = (nodeId: string): boolean => {
  const node = getNodeById(nodeId);
  if (!node) return false;
  const hardcodedValues = node.data?.hardcodedValues || {};
  return Object.values(hardcodedValues).some(
    (value) => value !== undefined && value !== null && value !== "",
  );
};

/**
 * Waits for any block card to appear in the block menu
 */
export const waitForAnyBlockCard = (
  timeout = 10000,
): Promise<Element | null> => {
  return new Promise((resolve) => {
    const startTime = Date.now();

    const checkBlock = () => {
      const block = document.querySelector(
        TUTORIAL_SELECTORS.BLOCK_CARD_PREFIX,
      );
      if (block) {
        resolve(block);
      } else if (Date.now() - startTime > timeout) {
        resolve(null);
      } else {
        setTimeout(checkBlock, TUTORIAL_CONFIG.ELEMENT_CHECK_INTERVAL);
      }
    };
    checkBlock();
  });
};

/**
 * Sets focus on an input element
 */
export const focusElement = (selector: string): void => {
  const element = document.querySelector(selector) as HTMLElement;
  if (element) {
    element.focus();
  }
};

/**
 * Scrolls an element into view smoothly
 */
export const scrollIntoView = (selector: string): void => {
  const element = document.querySelector(selector);
  if (element) {
    element.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }
};

/**
 * Triggers the fit view button to center the canvas
 */
export const fitViewToScreen = () => {
  const fitViewButton = document.querySelector(
    TUTORIAL_SELECTORS.FIT_VIEW_BUTTON,
  ) as HTMLButtonElement;
  if (fitViewButton) {
    fitViewButton.click();
  }
};

/**
 * Disables all blocks except the target block
 */
export const disableOtherBlocks = (targetBlockSelector: string) => {
  document
    .querySelectorAll(TUTORIAL_SELECTORS.BLOCK_CARD_PREFIX)
    .forEach((block) => {
      const isTarget = block.matches(targetBlockSelector);
      block.classList.toggle(CSS_CLASSES.DISABLE, !isTarget);
      block.classList.toggle(CSS_CLASSES.HIGHLIGHT, isTarget);
    });
};

/**
 * Enables all blocks (removes disable and highlight classes)
 */
export const enableAllBlocks = () => {
  document
    .querySelectorAll(TUTORIAL_SELECTORS.BLOCK_CARD_PREFIX)
    .forEach((block) => {
      block.classList.remove(
        CSS_CLASSES.DISABLE,
        CSS_CLASSES.HIGHLIGHT,
        CSS_CLASSES.PULSE,
      );
    });
};

/**
 * Adds highlight class to an element
 */
export const highlightElement = (selector: string) => {
  const element = document.querySelector(selector);
  if (element) {
    element.classList.add(CSS_CLASSES.HIGHLIGHT);
  }
};

/**
 * Removes highlight from all elements
 */
export const removeAllHighlights = () => {
  document.querySelectorAll(`.${CSS_CLASSES.HIGHLIGHT}`).forEach((el) => {
    el.classList.remove(CSS_CLASSES.HIGHLIGHT);
  });
  document.querySelectorAll(`.${CSS_CLASSES.PULSE}`).forEach((el) => {
    el.classList.remove(CSS_CLASSES.PULSE);
  });
};

/**
 * Adds pulse animation to an element
 */
export const pulseElement = (selector: string) => {
  const element = document.querySelector(selector);
  if (element) {
    element.classList.add(CSS_CLASSES.PULSE);
  }
};

/**
 * Forces the block menu to stay open during tutorial
 */
export const forceBlockMenuOpen = (force: boolean) => {
  useControlPanelStore.getState().setForceOpenBlockMenu(force);
};

/**
 * Opens the block menu
 */
export const openBlockMenu = () => {
  useControlPanelStore.getState().setBlockMenuOpen(true);
};

/**
 * Closes the block menu
 */
export const closeBlockMenu = () => {
  useControlPanelStore.getState().setBlockMenuOpen(false);
  useControlPanelStore.getState().setForceOpenBlockMenu(false);
};

/**
 * Opens the save control popover
 */
export const openSaveControl = () => {
  useControlPanelStore.getState().setSaveControlOpen(true);
};

/**
 * Closes the save control popover
 */
export const closeSaveControl = () => {
  useControlPanelStore.getState().setSaveControlOpen(false);
  useControlPanelStore.getState().setForceOpenSave(false);
};

/**
 * Forces the save control to stay open during tutorial
 */
export const forceSaveOpen = (force: boolean) => {
  useControlPanelStore.getState().setForceOpenSave(force);
};

/**
 * Clears the search input in block menu
 */
export const clearBlockMenuSearch = () => {
  const input = document.querySelector(
    TUTORIAL_SELECTORS.BLOCKS_SEARCH_INPUT,
  ) as HTMLInputElement;
  if (input) {
    input.value = "";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }
};

/**
 * Types text into an input element with event dispatch
 */
export const typeIntoInput = (selector: string, text: string) => {
  const input = document.querySelector(selector) as HTMLInputElement;
  if (input) {
    input.focus();
    input.value = text;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }
};

/**
 * Simulates a click on the save button
 */
export const clickSaveButton = () => {
  const saveButton = document.querySelector(
    TUTORIAL_SELECTORS.SAVE_AGENT_BUTTON,
  ) as HTMLButtonElement;
  if (saveButton && !saveButton.disabled) {
    saveButton.click();
  }
};

/**
 * Check if the agent has been saved (by checking if version exists)
 */
export const isAgentSaved = (): boolean => {
  const versionInput = document.querySelector(
    '[data-testid="save-control-version-output"]',
  ) as HTMLInputElement;
  return !!(versionInput && versionInput.value && versionInput.value !== "-");
};

/**
 * Handles tutorial cancellation
 */
export const handleTutorialCancel = (tour: any) => {
  closeBlockMenu();
  closeSaveControl();
  forceSaveOpen(false);
  removeAllHighlights();
  enableAllBlocks();
  tour.cancel();
  storage.set(Key.SHEPHERD_TOUR, "canceled");
};

/**
 * Handles tutorial skip
 */
export const handleTutorialSkip = (tour: any) => {
  closeBlockMenu();
  closeSaveControl();
  forceSaveOpen(false);
  removeAllHighlights();
  enableAllBlocks();
  tour.cancel();
  storage.set(Key.SHEPHERD_TOUR, "skipped");
};

/**
 * Handles tutorial completion
 */
export const handleTutorialComplete = () => {
  closeBlockMenu();
  closeSaveControl();
  forceSaveOpen(false);
  removeAllHighlights();
  enableAllBlocks();
  storage.set(Key.SHEPHERD_TOUR, "completed");
};

/**
 * Creates a mutation observer to watch for element appearance
 */
export const observeElement = (
  selector: string,
  callback: (element: Element) => void,
): MutationObserver => {
  const observer = new MutationObserver((mutations, obs) => {
    const element = document.querySelector(selector);
    if (element) {
      callback(element);
      obs.disconnect();
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });

  // Also check immediately
  const element = document.querySelector(selector);
  if (element) {
    callback(element);
    observer.disconnect();
  }

  return observer;
};

/**
 * Watches for search input changes and calls callback when target is typed
 */
export const watchSearchInput = (
  targetValue: string,
  onMatch: () => void,
): (() => void) => {
  const input = document.querySelector(
    TUTORIAL_SELECTORS.BLOCKS_SEARCH_INPUT,
  ) as HTMLInputElement;
  if (!input) return () => {};

  let hasMatched = false;

  const handler = () => {
    if (hasMatched) return;

    const currentValue = input.value.toLowerCase().trim();
    const target = targetValue.toLowerCase().trim();

    // Match when user types at least 4 characters that match
    if (currentValue.length >= 4 && target.startsWith(currentValue)) {
      hasMatched = true;
      onMatch();
    }
  };

  input.addEventListener("input", handler);

  return () => {
    input.removeEventListener("input", handler);
  };
};

/**
 * Highlights the first matching block in search results
 */
export const highlightFirstBlockInSearch = () => {
  const firstBlock = document.querySelector(
    TUTORIAL_SELECTORS.BLOCK_CARD_PREFIX,
  );
  if (firstBlock) {
    firstBlock.classList.add(CSS_CLASSES.PULSE);
    // Scroll it into view
    firstBlock.scrollIntoView({ behavior: "smooth", block: "center" });
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
