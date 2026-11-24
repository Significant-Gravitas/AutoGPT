import { useGetV1GetSpecificGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { getSavedNodeCredentials } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/node-credentials";
import { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { retryUnlessClientError } from "../../../helpers";

const NO_NODE_CREDENTIALS: CredentialsMetaInput[] = [];

/**
 * The credentials picked on the agent's steps in the builder. Only the
 * graph's owner gets them back: the API strips them for everyone else.
 */
export function useSavedNodeCredentials(agent: LibraryAgent, enabled: boolean) {
  const { data, isPending } = useGetV1GetSpecificGraph(
    agent.graph_id,
    { version: agent.graph_version },
    {
      query: {
        enabled,
        retry: retryUnlessClientError,
        select: (response) => {
          const graph = okData(response);
          const nodes = [
            ...(graph?.nodes ?? []),
            ...(graph?.sub_graphs ?? []).flatMap(
              (subGraph) => subGraph.nodes ?? [],
            ),
          ];
          return getSavedNodeCredentials(
            nodes.map((node) => node.input_default),
          );
        },
      },
    },
  );

  return {
    nodeCredentials: data ?? NO_NODE_CREDENTIALS,
    isLoadingNodeCredentials: enabled && isPending,
  };
}
