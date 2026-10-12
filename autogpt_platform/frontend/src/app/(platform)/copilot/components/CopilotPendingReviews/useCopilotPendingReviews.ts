import { useEffect, useState } from "react";
import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { okData } from "@/app/api/helpers";
import {
  usePendingReviewsForChatSession,
  usePendingReviewsForExecution,
} from "@/hooks/usePendingReviews";

// A run the chat started, or the chat's own queue.
interface Args {
  graphExecId?: string;
  graphId?: string;
  chatSessionId?: string;
  // Off for the chat's own list: it polls only while it holds cards, and
  // fetches again whenever ``refetchKey`` changes (a new held call).
  pollWhileEmpty?: boolean;
  refetchKey?: number;
}

const POLL_MS = 2000;
// A run can go minutes before it pauses, so poll it slowly until it does.
const PRE_REVIEW_POLL_MS = 5000;

export function useCopilotPendingReviews({
  graphExecId = "",
  graphId,
  chatSessionId = "",
  pollWhileEmpty = true,
  refetchKey,
}: Args) {
  const [hasRows, setHasRows] = useState(false);
  const isRun = !!graphId;
  // A run's chat message never changes, so its live status decides polling.
  const { data: execution, isFetched } = useGetV1GetExecutionDetails(
    graphId ?? "",
    graphExecId,
    {
      query: {
        enabled: isRun,
        select: okData,
        refetchInterval: (q) =>
          q.state.status === "error"
            ? PRE_REVIEW_POLL_MS
            : pollInterval(readStatus(q.state.data)),
      },
    },
  );
  const status = execution?.status;
  // Without the run's status, the reviews themselves are the only signal.
  const statusUnavailable = isRun && isFetched && !status;

  const forChat = usePendingReviewsForChatSession(chatSessionId, {
    enabled: !!chatSessionId,
    refetchInterval: pollWhileEmpty || hasRows ? POLL_MS : false,
  });
  const forRun = usePendingReviewsForExecution(graphExecId, {
    enabled: !!graphExecId && (!isRun || !!status || statusUnavailable),
    refetchInterval:
      !isRun || status === AgentExecutionStatus.REVIEW
        ? POLL_MS
        : statusUnavailable
          ? PRE_REVIEW_POLL_MS
          : false,
  });
  const { pendingReviews, refetch } = chatSessionId ? forChat : forRun;

  useEffect(() => {
    setHasRows(pendingReviews.length > 0);
  }, [pendingReviews.length]);

  useEffect(() => {
    if (refetchKey !== undefined) refetch();
  }, [refetchKey, refetch]);

  useEffect(() => {
    if (status) refetch();
  }, [status, refetch]);

  return { pendingReviews, refetch };
}

function readStatus(raw: unknown) {
  const response = raw as { status?: number; data?: { status?: string } };
  return response?.status === 200 ? response.data?.status : undefined;
}

function pollInterval(status: string | undefined) {
  if (status === AgentExecutionStatus.REVIEW) return POLL_MS;
  if (
    status === AgentExecutionStatus.QUEUED ||
    status === AgentExecutionStatus.RUNNING ||
    status === AgentExecutionStatus.INCOMPLETE
  )
    return PRE_REVIEW_POLL_MS;
  return false;
}
