import { useEffect, useState } from "react";
import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { okData } from "@/app/api/helpers";
import { usePendingReviewsForExecution } from "@/hooks/usePendingReviews";

interface Args {
  graphExecId: string;
  graphId?: string;
  // Off for a chat's session-level list while nothing on screen was held:
  // one fetch finds cards whose call paged out of the loaded history.
  pollWhileEmpty?: boolean;
}

const POLL_MS = 2000;

export function useCopilotPendingReviews({
  graphExecId,
  graphId,
  pollWhileEmpty = true,
}: Args) {
  const [hasRows, setHasRows] = useState(false);
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
        (!isRun && (pollWhileEmpty || hasRows)) ||
        status === AgentExecutionStatus.REVIEW
          ? POLL_MS
          : false,
    },
  );

  useEffect(() => {
    setHasRows(pendingReviews.length > 0);
  }, [pendingReviews.length]);

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
