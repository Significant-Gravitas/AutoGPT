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

// Helper to get connection status HTML
const getConnectionStatusHtml = (id: string, isConnected: boolean = false) => `
  <div id="${id}" class="mt-3 p-2 ${isConnected ? "bg-green-50 ring-1 ring-green-200" : "bg-amber-50 ring-1 ring-amber-200"} rounded-2xl text-center text-sm ${isConnected ? "text-green-600" : "text-amber-600"}">
    ${isConnected ? "✅ Connection already exists!" : "Waiting for connection..."}
  </div>
`;

// Helper to update connection status box
const updateConnectionStatus = (
  id: string,
  isConnected: boolean,
  message?: string,
) => {
  const statusEl = document.querySelector(`#${id}`);
  if (statusEl) {
    statusEl.innerHTML =
      message || (isConnected ? "✅ Connected!" : "Waiting for connection...");
    statusEl.classList.remove(
      "bg-amber-50",
      "ring-amber-200",
      "text-amber-600",
      "bg-green-50",
      "ring-green-200",
      "text-green-600",
    );
    if (isConnected) {
      statusEl.classList.add("bg-green-50", "ring-green-200", "text-green-600");
    } else {
      statusEl.classList.add("bg-amber-50", "ring-amber-200", "text-amber-600");
    }
  }
};

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
          
          <div class="mt-3 p-3 bg-blue-50 ring-1 ring-blue-200 rounded-2xl">
            <p class="text-sm font-medium text-blue-600 m-0 mb-2">Drag from the output:</p>
            <p class="text-[0.8125rem] text-blue-600 m-0">Click and drag from the <strong>result</strong> output handle (right side) of Agent Input block.</p>
          </div>
          ${getConnectionStatusHtml("connection-status-1-check", false)}
          ${banner(ICONS.Drag, "Click and drag from the highlighted output handle", "action")}
        </div>
      `,
      attachTo: {
        element: TUTORIAL_SELECTORS.INPUT_BLOCK_RESULT_OUTPUT_HANDLEER,
        on: "right",
      },
      modalOverlayOpeningPadding: 10,
      when: {
        show: () => {
          resetConnectionState();

          // Check if connection already exists
          const alreadyConnected = isConnectionMade(
            BLOCK_IDS.AGENT_INPUT,
            BLOCK_IDS.CALCULATOR,
          );

          if (alreadyConnected) {
            updateConnectionStatus(
              "connection-status-1-check",
              true,
              "✅ Connection already exists!",
            );
            // Auto-advance after brief delay since connection exists
            setTimeout(() => {
              tour.show("connect-calculator-output-handle");
            }, 1000);
            return;
          }

          const inputNode = getNodeByBlockId(BLOCK_IDS.AGENT_INPUT);
          if (inputNode) {
            const outputHandle = document.querySelector(
              TUTORIAL_SELECTORS.INPUT_BLOCK_RESULT_OUTPUT_HANDLEER,
            );
            if (outputHandle) {
              outputHandle.addEventListener("mousedown", handleMouseDown);
            }
          }

          // Subscribe to edge store for real-time detection
          const unsubscribe = useEdgeStore.subscribe(() => {
            const connected = isConnectionMade(
              BLOCK_IDS.AGENT_INPUT,
              BLOCK_IDS.CALCULATOR,
            );
            if (connected) {
              updateConnectionStatus("connection-status-1-check", true);
              setTimeout(() => {
                unsubscribe();
                tour.show("connect-calculator-output-handle");
              }, 500);
            }
          });

          (tour.getCurrentStep() as any)._edgeUnsubscribe = unsubscribe;
        },
        hide: () => {
          removeAllHighlights();
          const step = tour.getCurrentStep() as any;
          if (step?._edgeUnsubscribe) {
            step._edgeUnsubscribe();
          }
          const inputNode = getNodeByBlockId(BLOCK_IDS.AGENT_INPUT);
          if (inputNode) {
            const outputHandle = document.querySelector(
              TUTORIAL_SELECTORS.INPUT_BLOCK_RESULT_OUTPUT_HANDLEER,
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
        {
          text: "Skip (already connected)",
          action: () => tour.show("connect-calculator-output-handle"),
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
          
          <div class="mt-3 p-3 bg-blue-50 ring-1 ring-blue-200 rounded-2xl">
            <p class="text-sm font-medium text-blue-600 m-0 mb-2">Drop on the input:</p>
            <p class="text-[0.8125rem] text-blue-600 m-0">Drag to the <strong>A</strong> or <strong>B</strong> input handle (left side) of the Calculator block.</p>
          </div>
          ${getConnectionStatusHtml("connection-status-1", false)}
        </div>
      `,
      attachTo: {
        element: TUTORIAL_SELECTORS.INPUT_BLOCK_RESULT_OUTPUT_HANDLEER,
        on: "bottom",
      },
      modalOverlayOpeningPadding: 10,
      extraHighlights: [TUTORIAL_SELECTORS.CALCULATOR_NUMBER_A_INPUT_HANDLER],
      when: {
        show: () => {
          // Check if already connected
          const alreadyConnected = isConnectionMade(
            BLOCK_IDS.AGENT_INPUT,
            BLOCK_IDS.CALCULATOR,
          );

          if (alreadyConnected) {
            updateConnectionStatus(
              "connection-status-1",
              true,
              "✅ Connection already exists!",
            );
            setTimeout(() => {
              tour.next();
            }, 500);
            return;
          }

          // Subscribe to edge store changes to detect connection
          const unsubscribe = useEdgeStore.subscribe(() => {
            const connected = isConnectionMade(
              BLOCK_IDS.AGENT_INPUT,
              BLOCK_IDS.CALCULATOR,
            );

            if (connected) {
              updateConnectionStatus("connection-status-1", true);
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
          
          <div class="mt-3 p-3 bg-blue-50 ring-1 ring-blue-200 rounded-2xl">
            <p class="text-sm font-medium text-blue-600 m-0 mb-2">Drag from the output:</p>
            <p class="text-[0.8125rem] text-blue-600 m-0">Click and drag from the <strong>result</strong> output handle (right side) of Calculator block.</p>
          </div>
          ${getConnectionStatusHtml("connection-status-2-check", false)}
          ${banner(ICONS.Drag, "Click and drag from the highlighted output handle", "action")}
        </div>
      `,
      attachTo: {
        element: TUTORIAL_SELECTORS.CALCULATOR_RESULT_OUTPUT_HANDLEER,
        on: "right",
      },
      modalOverlayOpeningPadding: 10,
      extraHighlights: [TUTORIAL_SELECTORS.OUTPUT_VALUE_INPUT_HANDLEER],
      beforeShowPromise: async () => {
        fitViewToScreen();
        return Promise.resolve();
      },
      when: {
        show: () => {
          resetConnectionState();

          // Check if connection already exists
          const alreadyConnected = isConnectionMade(
            BLOCK_IDS.CALCULATOR,
            BLOCK_IDS.AGENT_OUTPUT,
          );

          if (alreadyConnected) {
            updateConnectionStatus(
              "connection-status-2-check",
              true,
              "✅ Connection already exists!",
            );
            // Auto-advance after brief delay since connection exists
            setTimeout(() => {
              tour.show("connections-complete");
            }, 1000);
            return;
          }

          const calcNode = getNodeByBlockId(BLOCK_IDS.CALCULATOR);
          if (calcNode) {
            const outputHandle = document.querySelector(
              TUTORIAL_SELECTORS.CALCULATOR_RESULT_OUTPUT_HANDLEER,
            );
            if (outputHandle) {
              highlightElement(
                TUTORIAL_SELECTORS.CALCULATOR_RESULT_OUTPUT_HANDLEER,
              );
              outputHandle.addEventListener("mousedown", handleMouseDown);
            }
          }

          // Subscribe to edge store for real-time detection
          const unsubscribe = useEdgeStore.subscribe(() => {
            const connected = isConnectionMade(
              BLOCK_IDS.CALCULATOR,
              BLOCK_IDS.AGENT_OUTPUT,
            );
            if (connected) {
              updateConnectionStatus("connection-status-2-check", true);
              setTimeout(() => {
                unsubscribe();
                tour.show("connections-complete");
              }, 500);
            }
          });

          (tour.getCurrentStep() as any)._edgeUnsubscribe = unsubscribe;
        },
        hide: () => {
          removeAllHighlights();
          const step = tour.getCurrentStep() as any;
          if (step?._edgeUnsubscribe) {
            step._edgeUnsubscribe();
          }
          const calcNode = getNodeByBlockId(BLOCK_IDS.CALCULATOR);
          if (calcNode) {
            const outputHandle = document.querySelector(
              TUTORIAL_SELECTORS.CALCULATOR_RESULT_OUTPUT_HANDLEER,
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
        {
          text: "Skip (already connected)",
          action: () => tour.show("connections-complete"),
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
          
          <div class="mt-3 p-3 bg-blue-50 ring-1 ring-blue-200 rounded-2xl">
            <p class="text-sm font-medium text-blue-600 m-0 mb-2">Drop on the input:</p>
            <p class="text-[0.8125rem] text-blue-600 m-0">Drag to the <strong>Value</strong> input handle (left side) of the Agent Output block.</p>
          </div>
          ${getConnectionStatusHtml("connection-status-2", false)}
        </div>
      `,
      attachTo: {
        element: TUTORIAL_SELECTORS.OUTPUT_VALUE_INPUT_HANDLEER,
        on: "top",
      },
      when: {
        show: () => {
          // Check if already connected
          const alreadyConnected = isConnectionMade(
            BLOCK_IDS.CALCULATOR,
            BLOCK_IDS.AGENT_OUTPUT,
          );

          if (alreadyConnected) {
            updateConnectionStatus(
              "connection-status-2",
              true,
              "✅ Connection already exists!",
            );
            setTimeout(() => {
              tour.next();
            }, 500);
            return;
          }

          // Subscribe to edge store changes
          const unsubscribe = useEdgeStore.subscribe(() => {
            const connected = isConnectionMade(
              BLOCK_IDS.CALCULATOR,
              BLOCK_IDS.AGENT_OUTPUT,
            );

            if (connected) {
              updateConnectionStatus("connection-status-2", true);
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
          
          <div class="mt-3 p-3 bg-green-50 ring-1 ring-green-200 rounded-2xl">
            <div class="flex items-center justify-center gap-2 text-sm font-medium text-green-600">
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
