import {
  getGetExpertQueryKey,
  useUpdateExpertSkills,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useListCopilotSkills } from "@/app/api/__generated__/endpoints/skills/skills";
import {
  getV2GetSpecificAgent,
  useGetV2ListStoreAgents,
} from "@/app/api/__generated__/endpoints/store/store";
import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { CopilotSkillInfo } from "@/app/api/__generated__/models/copilotSkillInfo";
import { Expert } from "@/app/api/__generated__/models/expert";
import { okData } from "@/app/api/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

export interface ExpertSkillEntry {
  name: string;
  library: CopilotSkillInfo | null;
}

export function useExpertSkills(expert: Expert) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [query, setQuery] = useState("");
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [source, setSource] = useState<"library" | "marketplace">("library");
  const [marketQuery, setMarketQuery] = useState("");
  const debouncedMarketQuery = useDebouncedValue(marketQuery, 250);
  const marketplaceSkills = useGetV2ListStoreAgents(
    { search_query: debouncedMarketQuery.trim(), page_size: 20 },
    {
      query: {
        enabled: isAddOpen && source === "marketplace",
        select: (res) => okData(res)?.agents ?? [],
      },
    },
  );
  const librarySkills = useListCopilotSkills({
    query: { select: (res) => okData(res) ?? [] },
  });
  const { mutateAsync: updateSkills, isPending } = useUpdateExpertSkills();

  const library = librarySkills.data ?? [];
  const byName = new Map(
    library.map((skill) => [skill.name.toLowerCase(), skill]),
  );
  const attached: ExpertSkillEntry[] = expert.skills.map((name) => ({
    name,
    library: byName.get(name.toLowerCase()) ?? null,
  }));
  const attachedNames = new Set(
    expert.skills.map((name) => name.toLowerCase()),
  );
  const available = library.filter(
    (skill) => !attachedNames.has(skill.name.toLowerCase()),
  );
  const needle = query.trim().toLowerCase();
  const visible = needle
    ? attached.filter(
        (entry) =>
          entry.name.toLowerCase().includes(needle) ||
          (entry.library?.description ?? "").toLowerCase().includes(needle),
      )
    : attached;

  async function save(
    skills: string[],
    successTitle: string,
    marketplaceListingIds: string[] = [],
  ) {
    try {
      await updateSkills({
        expertId: expert.id,
        data: { skills, marketplace_listing_ids: marketplaceListingIds },
      });
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: getGetExpertQueryKey(expert.id),
        }),
        invalidateExpertRosterQueries(queryClient),
      ]);
      toast({ title: successTitle, variant: "success" });
      return true;
    } catch (error) {
      toast({
        title: "Couldn't update skills",
        description: error instanceof ApiError ? error.message : undefined,
        variant: "destructive",
      });
      return false;
    }
  }

  async function addSkill(name: string) {
    const saved = await save([...expert.skills, name], `Added ${name}`);
    if (saved) setIsAddOpen(false);
  }

  async function addMarketplaceSkill(agent: StoreAgent) {
    try {
      const details = await getV2GetSpecificAgent(
        agent.creator.toLowerCase(),
        agent.slug,
      );
      if (details.status !== 200) throw new Error("listing unavailable");
      const saved = await save(expert.skills, `Added ${agent.agent_name}`, [
        details.data.store_listing_version_id,
      ]);
      if (saved) setIsAddOpen(false);
    } catch (error) {
      toast({
        title: "Couldn't add that skill",
        description: error instanceof ApiError ? error.message : undefined,
        variant: "destructive",
      });
    }
  }

  function removeSkill(name: string) {
    return save(
      expert.skills.filter((skill) => skill !== name),
      `Removed ${name}`,
    );
  }

  return {
    query,
    setQuery,
    visible,
    hasAny: attached.length > 0,
    available,
    isLibraryLoading: librarySkills.isLoading,
    isAddOpen,
    openAdd: () => setIsAddOpen(true),
    closeAdd: () => setIsAddOpen(false),
    source,
    setSource,
    marketQuery,
    setMarketQuery,
    marketplaceSkills: marketplaceSkills.data ?? [],
    isMarketplaceLoading: marketplaceSkills.isFetching,
    addSkill,
    addMarketplaceSkill,
    removeSkill,
    isSaving: isPending,
  };
}
