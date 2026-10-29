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
import {
  forgetDeliveries,
  recordFetchStart,
  recordFetchSuccess,
  recordGraphStatusDelivery,
  recordNodeExecutionDelivery,
} from "./executionUpdateOrder";

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
        const result = data as unknown as NodeExecutionResult;
        recordNodeExecutionDelivery(result);
        applyNodeExecutionEvent(data.node_id, result);
      },
    );

    const deregisterGraphExecutionStatusEvent = api.onWebSocketMessage(
      "graph_execution_event",
      (graphExecution) => {
        if (graphExecution.id != flowExecutionID) {
          return;
        }

        recordGraphStatusDelivery();
        setGraphExecutionStatus(graphExecution.status as AgentExecutionStatus);
      },
    );

    const [executionDetailsURL] = getGetV1GetExecutionDetailsQueryKey(
      flowID ?? "",
      flowExecutionID ?? "",
    );

    function refetchExecutionDetails() {
      void queryClient.invalidateQueries({ queryKey: [executionDetailsURL] });
    }

    // Note when each execution-details fetch starts and lands, so useFlow can
    // tell which websocket deliveries a snapshot already includes.
    const unsubscribeQueryCache = queryClient
      .getQueryCache()
      .subscribe((event) => {
        if (event.type !== "updated") return;
        if (event.query.queryKey[0] !== executionDetailsURL) return;
        if (event.action.type === "fetch") recordFetchStart();
        if (event.action.type === "success") {
          recordFetchSuccess(event.query.state.dataUpdatedAt);
        }
      });

    let hasSentSubscribe = false;

    // The server acknowledges a subscription once it is registered. Refetch
    // then too: anything published before that point is in the response, and
    // everything after it arrives over the websocket, so useFlow stops polling.
    // The ack carries no payload, so a late one for a previously opened run
    // can land here first; that is harmless because the server answers
    // subscribes in the order they were sent on the socket, so this run's own
    // ack still follows and its refetch runs after this run is registered.
    const deregisterSubscribeAck = api.onWebSocketMessage(
      "subscribe_graph_execution",
      () => {
        if (!hasSentSubscribe) return;
        setSubscribedExecutionID(flowExecutionID);
        refetchExecutionDetails();
      },
    );

    // Events published while the socket is down are lost, so snapshots are
    // the source of truth again until the resubscribe.
    const deregisterDisconnect = api.onWebSocketDisconnect(() => {
      hasSentSubscribe = false;
      setSubscribedExecutionID(null);
      forgetDeliveries();
    });

    const deregisterGraphExecutionSubscription =
      flowID && flowExecutionID
        ? api.onWebSocketConnect(() => {
            hasSentSubscribe = true;
            forgetDeliveries();
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
      unsubscribeQueryCache();
      setSubscribedExecutionID(null);
      forgetDeliveries();
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
