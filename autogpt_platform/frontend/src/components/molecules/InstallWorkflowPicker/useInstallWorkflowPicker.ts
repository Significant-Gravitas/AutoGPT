import {
  getListExpertsQueryKey,
  useInstallExpertWorkflow,
  useListExperts,
} from "@/app/api/__generated__/endpoints/experts/experts";
import {
  getV2GetSpecificAgent,
  useGetV2ListStoreAgents,
} from "@/app/api/__generated__/endpoints/store/store";
import { Expert } from "@/app/api/__generated__/models/expert";
import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

const SEARCH_DEBOUNCE_MS = 300;

interface Args {
  mode: "pick-expert" | "pick-workflow";
  storeListingVersionId?: string;
  expertId?: string;
  open: boolean;
  onClose: () => void;
}

export function useInstallWorkflowPicker({
  mode,
  storeListingVersionId,
  expertId,
  open,
  onClose,
}: Args) {
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState("");
  const [pendingKey, setPendingKey] = useState<string | null>(null);

  const expertsQuery = useListExperts({
    query: { select: (x) => x.data as Expert[], enabled: open },
  });
  const hiredExperts = (expertsQuery.data ?? []).filter(
    (expert) => !expert.is_template && !expert.is_archived,
  );
  const targetExpert = hiredExperts.find((expert) => expert.id === expertId);

  const debouncedSearchQuery = useDebouncedValue(
    searchQuery,
    SEARCH_DEBOUNCE_MS,
  );

  const searchResultsQuery = useGetV2ListStoreAgents(
    { search_query: debouncedSearchQuery || undefined, page_size: 10 },
    {
      query: {
        select: (x) => (x.data as StoreAgentsResponse).agents,
        enabled: open && mode === "pick-workflow",
      },
    },
  );

  const { mutateAsync: installWorkflow } = useInstallExpertWorkflow();

  async function install(expert: Expert, versionId: string) {
    if (
      expert.workflows.some(
        (workflow) => workflow.store_listing_version_id === versionId,
      )
    ) {
      toast({
        title: `Already installed on ${expert.name}`,
        variant: "success",
      });
      onClose();
      return;
    }
    try {
      await installWorkflow({
        expertId: expert.id,
        data: { store_listing_version_id: versionId },
      });
      await queryClient.invalidateQueries({
        queryKey: getListExpertsQueryKey(),
      });
      toast({ title: `Installed on ${expert.name}`, variant: "success" });
      onClose();
    } catch {
      toast({
        title: `Couldn't install on ${expert.name}`,
        description: "Something went wrong. Please try again.",
        variant: "destructive",
      });
    }
  }

  async function installOnExpert(expert: Expert) {
    if (!storeListingVersionId) return;
    setPendingKey(expert.id);
    try {
      await install(expert, storeListingVersionId);
    } finally {
      setPendingKey(null);
    }
  }

  async function installFromListing(agent: StoreAgent) {
    if (!targetExpert) return;
    setPendingKey(agent.agent_graph_id);
    try {
      const details = await getV2GetSpecificAgent(
        agent.creator.toLowerCase(),
        agent.slug,
      );
      if (details.status !== 200) {
        throw new Error("Failed to fetch agent details");
      }
      await install(targetExpert, details.data.store_listing_version_id);
    } catch {
      toast({
        title: `Couldn't install on ${targetExpert.name}`,
        description: "Something went wrong. Please try again.",
        variant: "destructive",
      });
    } finally {
      setPendingKey(null);
    }
  }

  const title =
    mode === "pick-expert"
      ? "Install on an expert"
      : `Install a workflow${targetExpert ? ` on ${targetExpert.name}` : ""}`;

  return {
    title,
    hiredExperts,
    searchQuery,
    setSearchQuery,
    searchResults: searchResultsQuery.data ?? [],
    isSearching: searchResultsQuery.isLoading,
    pendingKey,
    installOnExpert,
    installFromListing,
  };
}
