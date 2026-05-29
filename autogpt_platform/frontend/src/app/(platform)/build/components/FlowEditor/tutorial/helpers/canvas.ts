import { TUTORIAL_CONFIG, TUTORIAL_SELECTORS } from "../constants";
import { useNodeStore } from "../../../../stores/nodeStore";

export const waitForNodeOnCanvas = (
  timeout = 10000,
): Promise<Element | null> => {
  return new Promise((resolve) => {
    const startTime = Date.now();

    const checkNode = () => {
      const storeNodes = useNodeStore.getState().nodes;
      if (storeNodes.length > 0) {
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

export const getNodesCount = (): number => {
  return useNodeStore.getState().nodes.length;
};

export const getFirstNode = () => {
  const nodes = useNodeStore.getState().nodes;
  return nodes.length > 0 ? nodes[0] : null;
};

export const getNodeById = (nodeId: string) => {
  const nodes = useNodeStore.getState().nodes;
  return nodes.find((n) => n.id === nodeId);
};

export const nodeHasValues = (nodeId: string): boolean => {
  const node = getNodeById(nodeId);
  if (!node) return false;
  const hardcodedValues = node.data?.hardcodedValues || {};
  return Object.values(hardcodedValues).some(
    (value) => value !== undefined && value !== null && value !== "",
  );
};

export const fitViewToScreen = () => {
  const fitViewButton = document.querySelector(
    TUTORIAL_SELECTORS.FIT_VIEW_BUTTON,
  ) as HTMLButtonElement;
  if (fitViewButton) {
    fitViewButton.click();
  }
};
