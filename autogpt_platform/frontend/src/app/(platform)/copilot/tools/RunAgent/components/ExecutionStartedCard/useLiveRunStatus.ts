import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { ExecutionStartedResponse } from "@/app/api/__generated__/models/executionStartedResponse";
import { okData } from "@/app/api/helpers";
import { useContext } from "react";
import { CopilotChatActionsContext } from "../../../../components/CopilotChatActionsProvider/useCopilotChatActions";
import {
  isFinalStatus,
  nextPollInterval,
  retryUnlessPermanent,
  runErrorFor,
  shouldWatchRun,
} from "./helpers";

type WatchedRun = Pick<
  ExecutionStartedResponse,
  "graph_id" | "execution_id" | "status"
>;

export function useLiveRunStatus(output: WatchedRun | null) {
  // Only the signed-in /copilot chat polls: the public share page can't call
  // the API, and the builder panel already shows the run in place.
  const chatSurface = useContext(CopilotChatActionsContext)?.chatSurface;
  const enabled =
    !!output?.graph_id &&
    !!output.execution_id &&
    chatSurface === "copilot" &&
    shouldWatchRun(output.status);

  const { data: run } = useGetV1GetExecutionDetails(
    output?.graph_id ?? "",
    output?.execution_id ?? "",
    {
      query: {
        enabled,
        select: okData,
        refetchInterval: (query) => nextPollInterval(query.state),
        retry: retryUnlessPermanent,
        refetchOnWindowFocus: false,
      },
    },
  );

  const liveRun = enabled ? run : undefined;

  return {
    status: liveRun?.status ?? output?.status,
    error: liveRun ? runErrorFor(liveRun) : null,
    endedAfterStart: !!liveRun && isFinalStatus(liveRun.status),
  };
}
