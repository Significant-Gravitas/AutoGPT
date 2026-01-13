import { StepOptions } from "shepherd.js";
import {
  fitViewToScreen,
  highlightElement,
  removeAllHighlights,
} from "../helpers";
import { ICONS } from "../icons";
import { banner } from "../styles";
import { useEdgeStore } from "../../../../stores/edgeStore";
import { TUTORIAL_SELECTORS } from "../constants";

const getConnectionStatusHtml = (id: string, isConnected: boolean = false) => `
  <div id="${id}" class="mt-3 p-2 ${isConnected ? "bg-green-50 ring-1 ring-green-200" : "bg-amber-50 ring-1 ring-amber-200"} rounded-2xl text-center text-sm ${isConnected ? "text-green-600" : "text-amber-600"}">
    ${isConnected ? "✅ Connected!" : "Waiting for connection..."}
  </div>
`;

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

const hasAnyEdge = (): boolean => {
  return useEdgeStore.getState().edges.length > 0;
};

export const createConnectionSteps = (tour: any): StepOptions[] => {
  let isConnecting = false;

  const handleMouseDown = () => {
    isConnecting = true;

    const inputSelector =
      TUTORIAL_SELECTORS.FIRST_CALCULATOR_RESULT_OUTPUT_HANDLER;
    if (inputSelector) {
      highlightElement(inputSelector);
    }

    setTimeout(() => {
      if (isConnecting) {
        tour.next();
      }
    }, 100);
  };

  const resetConnectionState = () => {
    isConnecting = false;
  };

  return [
    {
      id: "connect-blocks-output",
      title: "Connect the Blocks: Output",
      text: `
        <div class="text-sm leading-[1.375rem] text-zinc-800">
          <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Now, let's connect the <strong>Result output</strong> of the first Calculator to the <strong>input (A)</strong> of the second Calculator.</p>
          
          <div class="mt-3 p-3 bg-blue-50 ring-1 ring-blue-200 rounded-2xl">
            <p class="text-sm font-medium text-blue-600 m-0 mb-2">Drag from the Result output:</p>
            <p class="text-[0.8125rem] text-blue-600 m-0">Click and drag from the <strong>Result</strong> output pin (right side) of the <strong>first Calculator block</strong>.</p>
          </div>
          ${getConnectionStatusHtml("connection-status-output", false)}
          ${banner(ICONS.Drag, "Drag from the Result output pin", "action")}
        </div>
      `,
      attachTo: {
        element: TUTORIAL_SELECTORS.FIRST_CALCULATOR_RESULT_OUTPUT_HANDLER,
        on: "left",
      },

      when: {
        show: () => {
          resetConnectionState();

          if (hasAnyEdge()) {
            updateConnectionStatus(
              "connection-status-output",
              true,
              "✅ Connection already exists!",
            );
            setTimeout(() => {
              tour.next();
            }, 1000);
            return;
          }

          const outputSelector =
            TUTORIAL_SELECTORS.FIRST_CALCULATOR_RESULT_OUTPUT_HANDLER;
          if (outputSelector) {
            const outputHandle = document.querySelector(outputSelector);
            if (outputHandle) {
              highlightElement(outputSelector);
              outputHandle.addEventListener("mousedown", handleMouseDown);
            }
          }

          const unsubscribe = useEdgeStore.subscribe(() => {
            if (hasAnyEdge()) {
              updateConnectionStatus("connection-status-output", true);
              setTimeout(() => {
                unsubscribe();
                tour.next();
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
          const outputSelector =
            TUTORIAL_SELECTORS.FIRST_CALCULATOR_RESULT_OUTPUT_HANDLER;
          if (outputSelector) {
            const outputHandle = document.querySelector(outputSelector);
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
          action: () => tour.show("connection-complete"),
          classes: "shepherd-button-secondary",
        },
      ],
    },

    {
      id: "connect-blocks-input",
      title: "Connect the Blocks: Input",
      text: `
        <div class="text-sm leading-[1.375rem] text-zinc-800">
          <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Now, connect to the <strong>input (A)</strong> of the second Calculator block.</p>
          
          <div class="mt-3 p-3 bg-blue-50 ring-1 ring-blue-200 rounded-2xl">
            <p class="text-sm font-medium text-blue-600 m-0 mb-2">Drop on the A input:</p>
            <p class="text-[0.8125rem] text-blue-600 m-0">Drag to the <strong>A</strong> input handle (left side) of the <strong>second Calculator block</strong>.</p>
          </div>
          ${getConnectionStatusHtml("connection-status-input", false)}
        </div>
      `,
      attachTo: {
        element: TUTORIAL_SELECTORS.SECOND_CALCULATOR_NUMBER_A_INPUT_HANDLER,
        on: "right",
      },
      when: {
        show: () => {
          const inputSelector =
            TUTORIAL_SELECTORS.SECOND_CALCULATOR_NUMBER_A_INPUT_HANDLER;
          if (inputSelector) {
            highlightElement(inputSelector);
          }

          if (hasAnyEdge()) {
            updateConnectionStatus(
              "connection-status-input",
              true,
              "✅ Connected!",
            );
            setTimeout(() => {
              tour.next();
            }, 500);
            return;
          }

          const unsubscribe = useEdgeStore.subscribe(() => {
            if (hasAnyEdge()) {
              updateConnectionStatus("connection-status-input", true);
              setTimeout(() => {
                unsubscribe();
                tour.next();
              }, 500);
            }
          });

          (tour.getCurrentStep() as any)._edgeUnsubscribe = unsubscribe;

          const handleMouseUp = () => {
            setTimeout(() => {
              if (!hasAnyEdge()) {
                isConnecting = false;
                tour.show("connect-blocks-output");
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
          action: () => tour.show("connect-blocks-output"),
          classes: "shepherd-button-secondary",
        },
        {
          text: "Skip (already connected)",
          action: () => tour.next(),
          classes: "shepherd-button-secondary",
        },
      ],
    },

    {
      id: "connection-complete",
      title: "Blocks Connected! 🎉",
      text: `
        <div class="text-sm leading-[1.375rem] text-zinc-800">
          <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Excellent! Your Calculator blocks are now connected:</p>
          
          <div class="mt-3 p-3 bg-green-50 ring-1 ring-green-200 rounded-2xl">
            <div class="flex items-center justify-center gap-2 text-sm font-medium text-green-600">
              <span>Calculator 1</span>
              <span>→</span>
              <span>Calculator 2</span>
            </div>
            <p class="text-[0.75rem] text-green-500 m-0 mt-2 text-center italic">The result of Calculator 1 flows into Calculator 2's input A</p>
          </div>
          
          <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.75rem;">Now let's save and run your agent!</p>
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
