import { useQueries } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import {
  getGetV2GetPendingReviewsForExecutionQueryKey,
  getGetV2GetPendingReviewsForExecutionQueryOptions,
  type getV2GetPendingReviewsForExecutionResponse,
  useGetV2GetAnsweredReviewsWaitingToRunForChatSession,
} from "@/app/api/__generated__/endpoints/executions/executions";
import { getGetV1GetExecutionDetailsQueryOptions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { okData } from "@/app/api/helpers";
import { usePendingReviewsForChatSession } from "@/hooks/usePendingReviews";
import type { RunReviewTarget } from "../ChatMessagesContainer/helpers";

// The runs the chat started, or the chat's own queue.
interface Args {
  runs?: RunReviewTarget[];
  chatSessionId?: string;
  // Off for the chat's own list: it polls only while it holds cards, and
  // fetches again whenever ``refetchKey`` changes (a new held call).
  pollWhileEmpty?: boolean;
  refetchKey?: number;
}

const POLL_MS = 2000;
// A run can go minutes before it pauses, so poll it slowly until it does.
const PRE_REVIEW_POLL_MS = 5000;
const NO_RUNS: RunReviewTarget[] = [];

export function useCopilotPendingReviews({
  runs = NO_RUNS,
  chatSessionId = "",
  pollWhileEmpty = true,
  refetchKey,
}: Args) {
  const [hasRows, setHasRows] = useState(false);
  // A run's chat message never changes, so its live status decides polling.
  const executions = useQueries({
    queries: runs.map((run) =>
      getGetV1GetExecutionDetailsQueryOptions(
        run.graphId ?? "",
        run.graphExecId,
        {
          query: {
            enabled: !!run.graphId,
            refetchInterval: (q) =>
              q.state.status === "error"
                ? PRE_REVIEW_POLL_MS
                : pollInterval(readStatus(q.state.data)),
          },
        },
      ),
    ),
  });
  const runReviews = useQueries({
    queries: runs.map((run, i) => {
      const isRun = !!run.graphId;
      const status = readStatus(executions[i]?.data);
      // Without the run's status, the reviews themselves are the only signal.
      const statusUnavailable = isRun && !!executions[i]?.isFetched && !status;
      const interval =
        !isRun || status === AgentExecutionStatus.REVIEW
          ? POLL_MS
          : statusUnavailable
            ? PRE_REVIEW_POLL_MS
            : false;
      return getGetV2GetPendingReviewsForExecutionQueryOptions(
        run.graphExecId,
        {
          query: {
            // Keyed on the status too, so a run that moves on reads its reviews again.
            queryKey: [
              ...getGetV2GetPendingReviewsForExecutionQueryKey(run.graphExecId),
              status ?? null,
            ],
            enabled:
              !!run.graphExecId && (!isRun || !!status || statusUnavailable),
            refetchInterval: interval,
            refetchIntervalInBackground: !!interval,
          },
        },
      );
    }),
  });

  const forChat = usePendingReviewsForChatSession(chatSessionId, {
    enabled: !!chatSessionId,
    refetchInterval: pollWhileEmpty || hasRows ? POLL_MS : false,
  });
  const runRows = mergeReviews(runReviews.map((query) => query.data));
  const pendingReviews = chatSessionId ? forChat.pendingReviews : runRows;
  // Fresh on every mount, never cached across one: the box seeds its receipts
  // from the first answer, so a stale list would seed the wrong ones.
  const { data: answeredReviews } =
    useGetV2GetAnsweredReviewsWaitingToRunForChatSession(chatSessionId, {
      query: {
        enabled: !!chatSessionId,
        select: okData,
        gcTime: 0,
        refetchOnWindowFocus: false,
      },
    });

  useEffect(() => {
    setHasRows(pendingReviews.length > 0);
  }, [pendingReviews.length]);

  const refetchChat = forChat.refetch;
  useEffect(() => {
    if (refetchKey !== undefined) refetchChat();
  }, [refetchKey, refetchChat]);

  async function refetch(): Promise<PendingHumanReviewModel[]> {
    if (chatSessionId) return okData((await refetchChat()).data) ?? [];
    const results = await Promise.all(
      runReviews.map((query) => query.refetch()),
    );
    return mergeReviews(results.map((result) => result.data));
  }

  return { pendingReviews, answeredReviews, refetch };
}

// One list for all the runs, oldest first, as the chat's own list is.
function mergeReviews(
  responses: (getV2GetPendingReviewsForExecutionResponse | undefined)[],
): PendingHumanReviewModel[] {
  const byID = new Map<string, PendingHumanReviewModel>();
  for (const response of responses) {
    for (const row of okData(response) ?? []) byID.set(row.node_exec_id, row);
  }
  return [...byID.values()].sort(
    (a, b) => +new Date(a.created_at) - +new Date(b.created_at),
  );
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
