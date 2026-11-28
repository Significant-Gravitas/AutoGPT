import {
  type getV1GetSpecificGraphResponse,
  useGetV1GetSpecificGraph,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { retryUnlessClientError } from "../../../../../helpers";
import { getAgentAtVersion } from "./helpers";

interface Args {
  agent?: LibraryAgent;
  run?: GraphExecution;
  /** Whether the workflow is wanted now: the rerun form is opening. */
  enabled: boolean;
}

/**
 * The workflow at the version the run ran, for a form that reruns it. The
 * library's entry describes the current version, so a run of another one
 * reads its own from the server; nothing is read for a run of the current one.
 */
export function useRunVersionAgent({ agent, run, enabled }: Args) {
  const isOtherVersion = Boolean(
    agent && run && run.graph_version !== agent.graph_version,
  );
  const isReading = enabled && isOtherVersion;
  const { data, isPending, isError, isFetching, refetch } =
    useGetV1GetSpecificGraph(
      run?.graph_id ?? "",
      { version: run?.graph_version },
      {
        query: {
          enabled: isReading,
          retry: retryUnlessClientError,
          // A saved version never changes
          staleTime: Infinity,
          select: selectGraph,
        },
      },
    );

  // A read that fails once it has loaded keeps the version it had
  const hasError = isReading && isError && data === undefined;
  const runAgent =
    agent && run && isOtherVersion
      ? data && getAgentAtVersion(agent, data, run.graph_version)
      : agent;

  return {
    runAgent,
    isLoading: isReading && isPending,
    hasError,
    isRetrying: hasError && isFetching,
    retry: () => {
      void refetch();
    },
  };
}

function selectGraph(response: getV1GetSpecificGraphResponse) {
  return okData(response);
}
