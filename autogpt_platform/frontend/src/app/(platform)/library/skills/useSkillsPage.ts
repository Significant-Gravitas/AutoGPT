import { useListCopilotSkills } from "@/app/api/__generated__/endpoints/skills/skills";
import { okData } from "@/app/api/helpers";

export function useSkillsPage() {
  const query = useListCopilotSkills({
    query: {
      select: (res) => okData(res) ?? [],
    },
  });

  return {
    skills: query.data ?? [],
    isLoading: query.isLoading,
    error: query.error,
  };
}
