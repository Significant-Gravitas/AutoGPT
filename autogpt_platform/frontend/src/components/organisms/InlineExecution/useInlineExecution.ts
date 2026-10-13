import { useRef, useState } from "react";
import {
  useGetV1GetExecutionDetails,
  usePostV1StopGraphExecution,
} from "@/app/api/__generated__/endpoints/graphs/graphs";

const active = new Set(["QUEUED", "RUNNING", "INCOMPLETE", "REVIEW"]);
export function useInlineExecution(
  graphId: string,
  executionId: string,
  enabled: boolean,
) {
  const [error, setError] = useState("");
  const [cancelRequested, setCancelRequested] = useState(false);
  const cancelling = useRef(false);
  const query = useGetV1GetExecutionDetails(graphId, executionId, {
    query: {
      enabled,
      retry: false,
      refetchInterval: (query) =>
        query.state.data?.status === 200 &&
        active.has(query.state.data.data.status)
          ? 3000
          : false,
    },
  });
  const stop = usePostV1StopGraphExecution();
  const execution =
    query.data?.status === 200 &&
    query.data.data.id === executionId &&
    query.data.data.graph_id === graphId
      ? query.data.data
      : null;
  async function cancel() {
    if (
      !enabled ||
      !execution ||
      !active.has(execution.status) ||
      cancelling.current
    )
      return;
    cancelling.current = true;
    setError("");
    try {
      const result = await stop.mutateAsync({
        graphId,
        graphExecId: executionId,
      });
      if (result.status !== 200) throw new Error("cancel");
      setCancelRequested(true);
      await query.refetch();
    } catch {
      setError(
        "The stop request could not be confirmed. Refresh the execution before trying again.",
      );
    } finally {
      cancelling.current = false;
    }
  }
  return {
    query,
    execution,
    error,
    cancel,
    cancelRequested,
    cancelling: stop.isPending,
    running: execution ? active.has(execution.status) : false,
  };
}
