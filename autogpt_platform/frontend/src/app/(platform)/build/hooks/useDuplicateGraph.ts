import {
  getGetV2GetLibraryAgentByGraphIdQueryKey,
  useGetV2GetLibraryAgentByGraphId,
  usePostV2ForkLibraryAgent,
} from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useRouter } from "next/navigation";
import { parseAsString, useQueryStates } from "nuqs";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useBuilderTenantScope } from "./useBuilderTenantScope";
import { getBuilderHref } from "@/services/org-team/builder";

export function useDuplicateGraph(teamId: string | null = null) {
  const router = useRouter();
  const { toast } = useToast();
  const tenantScope = useBuilderTenantScope();

  const [{ flowID }] = useQueryStates({
    flowID: parseAsString,
  });

  const { data: libraryAgent, isLoading: isCheckingLibrary } =
    useGetV2GetLibraryAgentByGraphId(
      flowID ?? "",
      {},
      {
        query: {
          select: (res) => res.data as LibraryAgent,
          enabled: !!flowID && tenantScope.isReady,
          queryKey: getTeamScopedQueryKey(
            getGetV2GetLibraryAgentByGraphIdQueryKey(flowID ?? "", {}),
            tenantScope.organizationId,
            tenantScope.teamId,
          ),
        },
        request: getTenantRequestInit(
          tenantScope.organizationId,
          tenantScope.teamId,
          tenantScope.isReady,
        ),
      },
    );

  const { mutateAsync: forkAgent, isPending: isDuplicating } =
    usePostV2ForkLibraryAgent({
      request: getTenantRequestInit(
        tenantScope.organizationId,
        teamId,
        tenantScope.isReady,
      ),
    });

  async function duplicate() {
    if (!libraryAgent) return;

    try {
      const result = await forkAgent({ libraryAgentId: libraryAgent.id });
      const forked = result.data as LibraryAgent;
      if (!forked?.graph_id) {
        throw new Error("Fork did not return a graph to open.");
      }
      router.push(
        getBuilderHref({
          graphId: forked.graph_id,
          graphVersion: forked.graph_version,
          organizationId: forked.organization_id ?? tenantScope.organizationId,
          teamId: forked.team_id ?? teamId,
        }),
      );
    } catch (error) {
      toast({
        title: "Failed to duplicate agent",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    }
  }

  return {
    duplicate,
    isDuplicating,
    canDuplicate: !!libraryAgent,
    isCheckingLibrary,
  };
}
