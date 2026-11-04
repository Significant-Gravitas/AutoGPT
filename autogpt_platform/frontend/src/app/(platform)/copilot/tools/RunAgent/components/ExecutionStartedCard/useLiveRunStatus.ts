import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { ExecutionStartedResponse } from "@/app/api/__generated__/models/executionStartedResponse";
import { okData } from "@/app/api/helpers";
import { useContext } from "react";
import { CopilotChatActionsContext } from "../../../../components/CopilotChatActionsProvider/useCopilotChatActions";
import {
  isFinalStatus,
  nextPollInterval,
  runErrorFor,
  shouldWatchRun,
} from "./helpers";

export function useLiveRunStatus(output: ExecutionStartedResponse | null) {
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
