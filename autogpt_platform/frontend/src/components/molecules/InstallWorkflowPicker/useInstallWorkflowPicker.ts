import {
  getGetExpertQueryKey,
  getListExpertsQueryKey,
  useInstallExpertWorkflow,
  useListExperts,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV2ListLibraryAgents } from "@/app/api/__generated__/endpoints/library/library";
import {
  getV2GetSpecificAgent,
  useGetV2ListStoreAgents,
} from "@/app/api/__generated__/endpoints/store/store";
import { Expert } from "@/app/api/__generated__/models/expert";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { InstallWorkflowSource, WorkflowInstallData } from "./helpers";

const SEARCH_DEBOUNCE_MS = 300;
const RESULTS_PAGE_SIZE = 10;

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
  const [source, setSource] = useState<InstallWorkflowSource>("library");
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
  const isPickingWorkflow = open && mode === "pick-workflow";

  const libraryResultsQuery = useGetV2ListLibraryAgents(
    {
      search_term: debouncedSearchQuery || undefined,
      page_size: RESULTS_PAGE_SIZE,
      is_hidden: false,
    },
    {
      query: {
        select: (x) => (x.data as LibraryAgentResponse).agents,
        enabled: isPickingWorkflow && source === "library",
      },
    },
  );

  const marketplaceResultsQuery = useGetV2ListStoreAgents(
    {
      search_query: debouncedSearchQuery || undefined,
      page_size: RESULTS_PAGE_SIZE,
    },
    {
      query: {
        select: (x) => (x.data as StoreAgentsResponse).agents,
        enabled: isPickingWorkflow && source === "marketplace",
      },
    },
  );

  const { mutateAsync: installWorkflow } = useInstallExpertWorkflow();

  async function installOnExpert(expert: Expert) {
    if (!storeListingVersionId) return;
    setPendingKey(expert.id);
    try {
      await install(expert, {
        store_listing_version_id: storeListingVersionId,
      });
    } finally {
      setPendingKey(null);
    }
  }

  async function installLibraryAgent(agent: LibraryAgent) {
    if (!targetExpert) return;
    setPendingKey(agent.id);
    try {
      await install(targetExpert, { library_agent_id: agent.id });
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
      await install(targetExpert, {
        store_listing_version_id: details.data.store_listing_version_id,
      });
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

  async function install(expert: Expert, data: WorkflowInstallData) {
    if (isAlreadyInstalled(expert, data)) {
      toast({
        title: `Already installed on ${expert.name}`,
        variant: "success",
      });
      onClose();
      return;
    }
    try {
      await installWorkflow({ expertId: expert.id, data });
      // Both: the picker is opened from the team list and from one expert's
      // own page, whose workflow section reads the single-expert query.
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: getListExpertsQueryKey() }),
        queryClient.invalidateQueries({
          queryKey: getGetExpertQueryKey(expert.id),
        }),
      ]);
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

  // Only the library source can tell: a marketplace row carries no listing
  // version until its detail is fetched at click time.
  function isLibraryAgentInstalled(agent: LibraryAgent) {
    return (
      targetExpert?.workflows.some(
        (workflow) => workflow.library_agent_id === agent.id,
      ) ?? false
    );
  }

  const title =
    mode === "pick-expert"
      ? "Install on an expert"
      : `Install a workflow${targetExpert ? ` on ${targetExpert.name}` : ""}`;

  const resultsQuery =
    source === "library" ? libraryResultsQuery : marketplaceResultsQuery;

  return {
    title,
    hiredExperts,
    isLibraryAgentInstalled,
    source,
    setSource,
    searchQuery,
    setSearchQuery,
    libraryResults: libraryResultsQuery.data ?? [],
    marketplaceResults: marketplaceResultsQuery.data ?? [],
    isSearching: resultsQuery.isLoading,
    pendingKey,
    installOnExpert,
    installLibraryAgent,
    installFromListing,
  };
}

function isAlreadyInstalled(expert: Expert, data: WorkflowInstallData) {
  return expert.workflows.some((workflow) =>
    "library_agent_id" in data
      ? workflow.library_agent_id === data.library_agent_id
      : workflow.store_listing_version_id === data.store_listing_version_id,
  );
}
