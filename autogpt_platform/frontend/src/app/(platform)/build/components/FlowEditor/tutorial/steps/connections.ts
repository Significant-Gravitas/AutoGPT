/**
 * Connection steps - Steps 14-16
 * Connect blocks together
 */

import { StepOptions } from "shepherd.js";
import { BLOCK_IDS, TUTORIAL_SELECTORS } from "../constants";
import {
  fitViewToScreen,
  highlightElement,
  removeAllHighlights,
  pulseElement,
  getNodeByBlockId,
  isConnectionMade,
} from "../helpers";
import { ICONS } from "../icons";
import { banner } from "../styles";
import { useEdgeStore } from "../../../../stores/edgeStore";

/**
 * Creates the connection steps
 */
export const createConnectionSteps = (tour: any): StepOptions[] => {
  let isConnecting = false;

  // Helper to detect when user starts dragging from output handle
  const handleMouseDown = () => {
    isConnecting = true;
    setTimeout(() => {
      if (isConnecting) {
        tour.next();
      }
    }, 100);
  };

  // Helper to reset connection state
  const resetConnectionState = () => {
    isConnecting = false;
  };

  return [
    // STEP 14a: Highlight Agent Input's OUTPUT handle - start dragging
    {
      id: "connect-input-output-handle",
      title: "Connect Agent Input → Calculator (Step 1)",
      text: `
        <div class="text-sm leading-[1.375rem] text-zinc-800">
          <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Now let's connect the blocks together!</p>
          
          <div class="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p class="text-sm font-medium text-blue-800 m-0 mb-2">Drag from the output:</p>
            <p class="text-[0.8125rem] text-blue-700 m-0">Click and drag from the <strong>result</strong> output handle (right side) of Agent Input block.</p>
          </div>
          ${banner(ICONS.Drag, "Click and drag from the highlighted output handle")}
        </div>
      `,
      attachTo: {
        element: `${TUTORIAL_SELECTORS.INPUT_NODE} [data-handlepos="right"]`,
        on: "right",
      },
      when: {
        show: () => {
          resetConnectionState();
          const inputNode = getNodeByBlockId(BLOCK_IDS.AGENT_INPUT);
          if (inputNode) {
            // Highlight the output handle specifically
            const outputHandle = document.querySelector(
              `[data-id="custom-node-${inputNode.id}"] [data-handlepos="right"]`,
            );
            if (outputHandle) {
              highlightElement(
                `[data-id="custom-node-${inputNode.id}"] [data-handlepos="right"]`,
              );
              outputHandle.addEventListener("mousedown", handleMouseDown);
            }
          }
        },
        hide: () => {
          removeAllHighlights();
          const inputNode = getNodeByBlockId(BLOCK_IDS.AGENT_INPUT);
          if (inputNode) {
            const outputHandle = document.querySelector(
              `[data-id="custom-node-${inputNode.id}"] [data-handlepos="right"]`,
            );
            if (outputHandle) {
              outputHandle.removeEventListener("mousedown", handleMouseDown);
            }
          }
        },
      },
      buttons: [
        {
          text: "Back",
          action: () => tour.back(),
          classes: "shepherd-button-secondary",
        },
      ],
    },

    // STEP 14b: Highlight Calculator's INPUT handle - complete connection
    {
      id: "connect-input-to-calculator-target",
      title: "Connect Agent Input → Calculator (Step 2)",
      text: `
        <div class="text-sm leading-[1.375rem] text-zinc-800">
          <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Now connect to the Calculator's input!</p>
          
          <div class="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p class="text-sm font-medium text-blue-800 m-0 mb-2">Drop on the input:</p>
            <p class="text-[0.8125rem] text-blue-700 m-0">Drag to the <strong>A</strong> or <strong>B</strong> input handle (left side) of the Calculator block.</p>
          </div>
          
          <div id="connection-status-1" class="mt-3 p-2 bg-amber-100 ring-1 ring-amber-500 rounded text-center text-sm text-amber-700">
            Waiting for connection...
          </div>
        </div>
      `,
      attachTo: {
        element: `${TUTORIAL_SELECTORS.CALCULATOR_NODE} [data-handlepos="left"]`,
        on: "left",
      },
      when: {
        show: () => {
          const calcNode = getNodeByBlockId(BLOCK_IDS.CALCULATOR);
          if (calcNode) {
            // Highlight/pulse the input handles
            pulseElement(
              `[data-id="custom-node-${calcNode.id}"] [data-handlepos="left"]`,
            );
          }

          // Subscribe to edge store changes to detect connection
          const unsubscribe = useEdgeStore.subscribe(() => {
            const connected = isConnectionMade(
              BLOCK_IDS.AGENT_INPUT,
              BLOCK_IDS.CALCULATOR,
            );
            const statusEl = document.querySelector("#connection-status-1");

            if (connected && statusEl) {
              statusEl.innerHTML = "✅ Connected!";
              statusEl.classList.remove(
                "bg-amber-100",
                "ring-amber-500",
                "text-amber-700",
              );
              statusEl.classList.add("bg-green-100", "text-green-700");

              // Auto-advance after brief delay
              setTimeout(() => {
                unsubscribe();
                tour.next();
              }, 500);
            }
          });

          (tour.getCurrentStep() as any)._edgeUnsubscribe = unsubscribe;

          // Also handle mouseup to detect failed connection attempts
          const handleMouseUp = (event: MouseEvent) => {
            setTimeout(() => {
              const connected = isConnectionMade(
                BLOCK_IDS.AGENT_INPUT,
                BLOCK_IDS.CALCULATOR,
              );
              if (!connected) {
                // Connection failed, go back to output handle step
                isConnecting = false;
                tour.show("connect-input-output-handle");
              }
            }, 200);
          };
          document.addEventListener("mouseup", handleMouseUp, true);
          (tour.getCurrentStep() as any)._mouseUpHandler = handleMouseUp;
        },
        hide: () => {
          removeAllHighlights();
          const step = tour.getCurrentStep() as any;
          if (step?._edgeUnsubscribe) {
            step._edgeUnsubscribe();
          }
          if (step?._mouseUpHandler) {
            document.removeEventListener("mouseup", step._mouseUpHandler, true);
          }
        },
      },
      buttons: [
        {
          text: "Back",
          action: () => tour.show("connect-input-output-handle"),
          classes: "shepherd-button-secondary",
        },
        {
          text: "Skip (already connected)",
          action: () => tour.next(),
          classes: "shepherd-button-secondary",
        },
      ],
    },

    // STEP 15a: Highlight Calculator's OUTPUT handle - start dragging
    {
      id: "connect-calculator-output-handle",
      title: "Connect Calculator → Agent Output (Step 1)",
      text: `
        <div class="text-sm leading-[1.375rem] text-zinc-800">
          <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Great! Now let's connect Calculator to Agent Output.</p>
          
          <div class="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p class="text-sm font-medium text-blue-800 m-0 mb-2">Drag from the output:</p>
            <p class="text-[0.8125rem] text-blue-700 m-0">Click and drag from the <strong>result</strong> output handle (right side) of Calculator block.</p>
          </div>
          ${banner(ICONS.Drag, "Click and drag from the highlighted output handle")}
        </div>
      `,
      attachTo: {
        element: `${TUTORIAL_SELECTORS.CALCULATOR_NODE} [data-handlepos="right"]`,
        on: "right",
      },
      beforeShowPromise: async () => {
        fitViewToScreen();
        return Promise.resolve();
      },
      when: {
        show: () => {
          resetConnectionState();
          const calcNode = getNodeByBlockId(BLOCK_IDS.CALCULATOR);
          if (calcNode) {
            const outputHandle = document.querySelector(
              `[data-id="custom-node-${calcNode.id}"] [data-handlepos="right"]`,
            );
            if (outputHandle) {
              highlightElement(
                `[data-id="custom-node-${calcNode.id}"] [data-handlepos="right"]`,
              );
              outputHandle.addEventListener("mousedown", handleMouseDown);
            }
          }
        },
        hide: () => {
          removeAllHighlights();
          const calcNode = getNodeByBlockId(BLOCK_IDS.CALCULATOR);
          if (calcNode) {
            const outputHandle = document.querySelector(
              `[data-id="custom-node-${calcNode.id}"] [data-handlepos="right"]`,
            );
            if (outputHandle) {
              outputHandle.removeEventListener("mousedown", handleMouseDown);
            }
          }
        },
      },
      buttons: [
        {
          text: "Back",
          action: () => tour.back(),
          classes: "shepherd-button-secondary",
        },
      ],
    },

    // STEP 15b: Highlight Agent Output's INPUT handle - complete connection
    {
      id: "connect-calculator-to-output-target",
      title: "Connect Calculator → Agent Output (Step 2)",
      text: `
        <div class="text-sm leading-[1.375rem] text-zinc-800">
          <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Now connect to the Agent Output's input!</p>
          
          <div class="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p class="text-sm font-medium text-blue-800 m-0 mb-2">Drop on the input:</p>
            <p class="text-[0.8125rem] text-blue-700 m-0">Drag to the <strong>Value</strong> input handle (left side) of the Agent Output block.</p>
          </div>
          
          <div id="connection-status-2" class="mt-3 p-2 bg-amber-100 ring-1 ring-amber-500 rounded text-center text-sm text-amber-700">
            Waiting for connection...
          </div>
        </div>
      `,
      attachTo: {
        element: `${TUTORIAL_SELECTORS.OUTPUT_NODE} [data-handlepos="left"]`,
        on: "left",
      },
      when: {
        show: () => {
          const outputNode = getNodeByBlockId(BLOCK_IDS.AGENT_OUTPUT);
          if (outputNode) {
            pulseElement(
              `[data-id="custom-node-${outputNode.id}"] [data-handlepos="left"]`,
            );
          }

          // Subscribe to edge store changes
          const unsubscribe = useEdgeStore.subscribe(() => {
            const connected = isConnectionMade(
              BLOCK_IDS.CALCULATOR,
              BLOCK_IDS.AGENT_OUTPUT,
            );
            const statusEl = document.querySelector("#connection-status-2");

            if (connected && statusEl) {
              statusEl.innerHTML = "✅ Connected!";
              statusEl.classList.remove(
                "bg-amber-100",
                "ring-amber-500",
                "text-amber-700",
              );
              statusEl.classList.add("bg-green-100", "text-green-700");

              setTimeout(() => {
                unsubscribe();
                tour.next();
              }, 500);
            }
          });

          (tour.getCurrentStep() as any)._edgeUnsubscribe = unsubscribe;

          // Handle failed connection attempts
          const handleMouseUp = (event: MouseEvent) => {
            setTimeout(() => {
              const connected = isConnectionMade(
                BLOCK_IDS.CALCULATOR,
                BLOCK_IDS.AGENT_OUTPUT,
              );
              if (!connected) {
                isConnecting = false;
                tour.show("connect-calculator-output-handle");
              }
            }, 200);
          };
          document.addEventListener("mouseup", handleMouseUp, true);
          (tour.getCurrentStep() as any)._mouseUpHandler = handleMouseUp;
        },
        hide: () => {
          removeAllHighlights();
          const step = tour.getCurrentStep() as any;
          if (step?._edgeUnsubscribe) {
            step._edgeUnsubscribe();
          }
          if (step?._mouseUpHandler) {
            document.removeEventListener("mouseup", step._mouseUpHandler, true);
          }
        },
      },
      buttons: [
        {
          text: "Back",
          action: () => tour.show("connect-calculator-output-handle"),
          classes: "shepherd-button-secondary",
        },
        {
          text: "Skip (already connected)",
          action: () => tour.next(),
          classes: "shepherd-button-secondary",
        },
      ],
    },

    // STEP 16: Connections Complete (keep as-is)
    {
      id: "connections-complete",
      title: "Connections Complete! 🎉",
      text: `
        <div class="text-sm leading-[1.375rem] text-zinc-800">
          <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Excellent! Your agent workflow is now connected:</p>
          
          <div class="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg">
            <div class="flex items-center justify-center gap-2 text-sm font-medium text-green-800">
              <span>Agent Input</span>
              <span>→</span>
              <span>Calculator</span>
              <span>→</span>
              <span>Agent Output</span>
            </div>
          </div>
          
          <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.75rem;">Data will flow from input, through the calculator, and out as the result.</p>
          <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.5rem;">Now let's save your agent!</p>
        </div>
      `,
      beforeShowPromise: async () => {
        fitViewToScreen();
        return Promise.resolve();
      },
      buttons: [
        {
          text: "Save My Agent",
          action: () => tour.next(),
        },
      ],
    },
  ];
};
