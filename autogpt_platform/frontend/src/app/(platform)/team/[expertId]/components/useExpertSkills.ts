import {
  getGetExpertQueryKey,
  useUpdateExpertSkills,
} from "@/app/api/__generated__/endpoints/experts/experts";
import {
  getListCopilotSkillsQueryKey,
  useListCopilotSkills,
} from "@/app/api/__generated__/endpoints/skills/skills";
import {
  useGetV2ListMarketplaceSkills,
  usePostV2InstallMarketplaceSkill,
} from "@/app/api/__generated__/endpoints/store/store";
import { MarketplaceSkill } from "@/app/api/__generated__/models/marketplaceSkill";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { CopilotSkillInfo } from "@/app/api/__generated__/models/copilotSkillInfo";
import { Expert } from "@/app/api/__generated__/models/expert";
import { okData } from "@/app/api/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

export interface ExpertSkillEntry {
  name: string;
  library: CopilotSkillInfo | null;
  skill: CopilotSkillInfo | null;
}

export function useExpertSkills(expert: Expert) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [query, setQuery] = useState("");
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [source, setSource] = useState<"library" | "marketplace">("library");
  const [marketQuery, setMarketQuery] = useState("");
  const debouncedMarketQuery = useDebouncedValue(marketQuery, 250);
  const hub = useFlagStatus(Flag.SKILLS_HUB);
  const marketplaceSkills = useGetV2ListMarketplaceSkills(
    { search_query: debouncedMarketQuery.trim(), page_size: 20 },
    {
      query: {
        enabled: hub.enabled && isAddOpen && source === "marketplace",
        select: (res) => okData(res)?.skills ?? [],
      },
    },
  );
  const { mutateAsync: installHubSkill, isPending: isInstalling } =
    usePostV2InstallMarketplaceSkill();
  const librarySkills = useListCopilotSkills(undefined, {
    query: { select: (res) => okData(res) ?? [] },
  });
  const expertSkills = useListCopilotSkills(
    { expert_id: expert.id },
    {
      query: { select: (res) => okData(res) ?? [] },
    },
  );
  const { mutateAsync: updateSkills, isPending } = useUpdateExpertSkills();

  const library = librarySkills.data ?? [];
  const byName = new Map(
    library.map((skill) => [skill.name.toLowerCase(), skill]),
  );
  const ownedByName = new Map(
    (expertSkills.data ?? []).map((skill) => [skill.name.toLowerCase(), skill]),
  );
  const attached: ExpertSkillEntry[] = expert.skills.map((name) => ({
    name,
    skill: ownedByName.get(name.toLowerCase()) ?? null,
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
          (entry.skill?.description ?? entry.library?.description ?? "")
            .toLowerCase()
            .includes(needle),
      )
    : attached;

  async function save(skills: string[], successTitle: string) {
    try {
      await updateSkills({ expertId: expert.id, data: { skills } });
      await refreshExpert();
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

  /** Copies the Hub skill's instructions into this expert's own folder, so
   *  the expert runs it rather than only listing its name. */
  async function addMarketplaceSkill(skill: MarketplaceSkill) {
    try {
      const response = await installHubSkill({
        slug: skill.slug,
        params: { expert_id: expert.id },
      });
      if (response.status !== 200) throw new Error("install failed");
      await refreshExpert();
      toast({ title: `Added ${skill.name}`, variant: "success" });
      setIsAddOpen(false);
    } catch (error) {
      toast({
        title: "Couldn't add that skill",
        description: error instanceof ApiError ? error.message : undefined,
        variant: "destructive",
      });
    }
  }

  function refreshExpert() {
    return Promise.all([
      queryClient.invalidateQueries({
        queryKey: getGetExpertQueryKey(expert.id),
      }),
      queryClient.invalidateQueries({
        queryKey: getListCopilotSkillsQueryKey({ expert_id: expert.id }),
      }),
      invalidateExpertRosterQueries(queryClient),
    ]);
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
    hasMarketplace: hub.enabled,
    addSkill,
    addMarketplaceSkill,
    removeSkill,
    isSaving: isPending || isInstalling,
  };
}
