// In this hook, I am only keeping websocket related code.

import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { GetV1GetExecutionDetails200 } from "@/app/api/__generated__/models/getV1GetExecutionDetails200";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { GraphExecutionID, GraphID } from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { parseAsString, useQueryStates } from "nuqs";
import { useEffect } from "react";

export const useFlowRealtime = () => {
  const api = useBackendAPI();
  const { toast } = useToast();

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
        // TODO: Update the states of nodes
      },
    );

    const deregisterGraphExecutionSubscription =
      flowID && flowExecutionID
        ? api.onWebSocketConnect(() => {
            // Subscribe to execution updates
            api
              .subscribeToGraphExecution(flowExecutionID as GraphExecutionID) // TODO: We are currently using a manual type, we need to fix it in future
              .then(() => {
                toast({
                  title: `✅ Connected to the backend successfully.`,
                });
                console.debug(
                  `Subscribed to updates for execution #${flowExecutionID}`,
                );
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
    };
  }, [api, flowExecutionID]);

  return {};
};
