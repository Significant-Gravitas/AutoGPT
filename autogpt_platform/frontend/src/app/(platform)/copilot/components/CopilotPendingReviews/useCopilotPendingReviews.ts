import { useEffect } from "react";
import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { okData } from "@/app/api/helpers";
import { usePendingReviewsForExecution } from "@/hooks/usePendingReviews";

interface Args {
  graphExecId: string;
  graphId?: string;
}

const POLL_MS = 2000;

export function useCopilotPendingReviews({ graphExecId, graphId }: Args) {
  const isRun = !!graphId;
  // A run's chat message never changes, so its live status decides polling.
  const { data: execution } = useGetV1GetExecutionDetails(
    graphId ?? "",
    graphExecId,
    {
      query: {
        enabled: isRun,
        select: okData,
        refetchInterval: (q) =>
          isLiveStatus(readStatus(q.state.data)) ? POLL_MS : false,
      },
    },
  );
  const status = execution?.status;

  const { pendingReviews, refetch } = usePendingReviewsForExecution(
    graphExecId,
    {
      enabled: !!graphExecId && (!isRun || !!status),
      refetchInterval:
        !isRun || status === AgentExecutionStatus.REVIEW ? POLL_MS : false,
    },
  );

  useEffect(() => {
    if (status) refetch();
  }, [status, refetch]);

  return { pendingReviews, refetch };
}

function readStatus(raw: unknown) {
  const response = raw as { status?: number; data?: { status?: string } };
  return response?.status === 200 ? response.data?.status : undefined;
}

function isLiveStatus(status: string | undefined) {
  return (
    status === AgentExecutionStatus.QUEUED ||
    status === AgentExecutionStatus.RUNNING ||
    status === AgentExecutionStatus.INCOMPLETE ||
    status === AgentExecutionStatus.REVIEW
  );
}
