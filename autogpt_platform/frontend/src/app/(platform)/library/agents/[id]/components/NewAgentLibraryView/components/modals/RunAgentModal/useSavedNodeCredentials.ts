import { useGetV1GetSpecificGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import {
  getSavedNodeCredentials,
  getStepInputs,
} from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/node-credentials";
import { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { retryUnlessClientError } from "../../../helpers";

const NO_NODE_CREDENTIALS: CredentialsMetaInput[] = [];

/**
 * The credentials picked on the agent's steps in the builder, its
 * sub-workflows' steps included. Only the graph's owner gets them back: the
 * API strips them for everyone else.
 */
export function useSavedNodeCredentials(agent: LibraryAgent, enabled: boolean) {
  const { data, isPending, isError, isFetching, refetch } =
    useGetV1GetSpecificGraph(
      agent.graph_id,
      { version: agent.graph_version },
      {
        query: {
          enabled,
          retry: retryUnlessClientError,
          select: (response) => {
            const graph = okData(response);
            const graphs = graph ? [graph, ...(graph.sub_graphs ?? [])] : [];
            return getSavedNodeCredentials(getStepInputs(graphs));
          },
        },
      },
    );

  // A failed read says nothing about the steps' keys; one that fails again
  // after loading keeps the keys it had.
  const hasNodeCredentialsError = enabled && isError && data === undefined;

  return {
    nodeCredentials: data ?? NO_NODE_CREDENTIALS,
    isLoadingNodeCredentials: enabled && isPending,
    hasNodeCredentialsError,
    isRetryingNodeCredentials: hasNodeCredentialsError && isFetching,
    retryNodeCredentials: () => {
      void refetch();
    },
  };
}
