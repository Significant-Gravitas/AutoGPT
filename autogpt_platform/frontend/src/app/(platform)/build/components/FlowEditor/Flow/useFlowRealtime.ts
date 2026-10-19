// In this hook, I am only keeping websocket related code.

import { GraphExecutionID } from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { parseAsString, useQueryStates } from "nuqs";
import { useEffect } from "react";
import { useNodeStore } from "../../../stores/nodeStore";
import { useShallow } from "zustand/react/shallow";
import { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { useGraphStore } from "../../../stores/graphStore";
import { useEdgeStore } from "../../../stores/edgeStore";
import { useQueryClient } from "@tanstack/react-query";
import { getGetV1GetExecutionDetailsQueryKey } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { create } from "zustand";

const useSubscribedExecutionStore = create<{ executionID: string | null }>(
  () => ({ executionID: null }),
);

function setSubscribedExecutionID(executionID: string | null) {
  useSubscribedExecutionStore.setState({ executionID });
}

/**
 * True once the server has acknowledged the websocket subscription to this
 * execution, so every later node event arrives over the websocket.
 */
export function useIsExecutionSubscribed(executionID: string | null) {
  return useSubscribedExecutionStore(
    (state) => executionID !== null && state.executionID === executionID,
  );
}

export const useFlowRealtime = () => {
  const api = useBackendAPI();
  const queryClient = useQueryClient();
  const applyNodeExecutionEvent = useNodeStore(
    useShallow((state) => state.applyNodeExecutionEvent),
  );
  const setGraphExecutionStatus = useGraphStore(
    useShallow((state) => state.setGraphExecutionStatus),
  );
  const resetEdgeBeads = useEdgeStore(
    useShallow((state) => state.resetEdgeBeads),
  );

  const [{ flowExecutionID, flowID }] = useQueryStates({
    flowExecutionID: parseAsString,
    flowID: parseAsString,
  });

  useEffect(() => {
    const deregisterNodeExecutionEvent = api.onWebSocketMessage(
      "node_execution_event",
      (data) => {
        if (data.graph_exec_id != flowExecutionID) {
          return;
        }
        applyNodeExecutionEvent(
          data.node_id,
          data as unknown as NodeExecutionResult,
        );
      },
    );

    const deregisterGraphExecutionStatusEvent = api.onWebSocketMessage(
      "graph_execution_event",
      (graphExecution) => {
        if (graphExecution.id != flowExecutionID) {
          return;
        }

        setGraphExecutionStatus(graphExecution.status as AgentExecutionStatus);
      },
    );

    function refetchExecutionDetails() {
      void queryClient.invalidateQueries({
        queryKey: getGetV1GetExecutionDetailsQueryKey(
          flowID!,
          flowExecutionID!,
        ),
      });
    }

    let hasSentSubscribe = false;

    // The server acknowledges a subscription once it is registered. Refetch
    // then too: anything published before that point is in the response, and
    // everything after it arrives over the websocket, so useFlow stops polling.
    const deregisterSubscribeAck = api.onWebSocketMessage(
      "subscribe_graph_execution",
      () => {
        if (!hasSentSubscribe) return;
        setSubscribedExecutionID(flowExecutionID);
        refetchExecutionDetails();
      },
    );

    const deregisterDisconnect = api.onWebSocketDisconnect(() => {
      hasSentSubscribe = false;
      setSubscribedExecutionID(null);
    });

    const deregisterGraphExecutionSubscription =
      flowID && flowExecutionID
        ? api.onWebSocketConnect(() => {
            hasSentSubscribe = true;
            // Subscribe to execution updates
            api
              .subscribeToGraphExecution(flowExecutionID as GraphExecutionID) // TODO: We are currently using a manual type, we need to fix it in future
              .then(() => {
                console.debug(
                  `Subscribed to updates for execution #${flowExecutionID}`,
                );
                // Refetch execution details to catch any events that were
                // published before the WebSocket subscription was established.
                // This closes the race-condition window for fast-completing
                // executions like dry-runs / simulations.
                refetchExecutionDetails();
              })
              .catch((error) =>
                console.error(
                  `Failed to subscribe to updates for execution #${flowExecutionID}:`,
                  error,
                ),
              );
          })
        : () => {};

    return () => {
      deregisterNodeExecutionEvent();
      deregisterGraphExecutionSubscription();
      deregisterGraphExecutionStatusEvent();
      deregisterSubscribeAck();
      deregisterDisconnect();
      setSubscribedExecutionID(null);
      resetEdgeBeads();
    };
  }, [
    api,
    flowExecutionID,
    resetEdgeBeads,
    queryClient,
    flowID,
    applyNodeExecutionEvent,
    setGraphExecutionStatus,
  ]);

  return {};
};
