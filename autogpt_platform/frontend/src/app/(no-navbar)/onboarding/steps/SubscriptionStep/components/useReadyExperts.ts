import { useGetBrainDumpRecommendedExperts } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { useOnboardingWizardStore } from "../../../store";

export interface ReadyExpert {
  id: string;
  name: string;
  avatarUrl: string | null;
}

// The Experts hired a step earlier, read from the recommendation the hire
// step already loaded, so the welcome can greet the user's own setup.
export function useReadyExperts() {
  const hiredTemplateIds = useOnboardingWizardStore((s) => s.hiredTemplateIds);
  const { data } = useGetBrainDumpRecommendedExperts({
    query: { enabled: hiredTemplateIds.length > 0, staleTime: Infinity },
  });
  const experts = data?.status === 200 ? (data.data.team?.experts ?? []) : [];
  return experts
    .filter((expert) => hiredTemplateIds.includes(expert.template_id))
    .map(
      (expert): ReadyExpert => ({
        id: expert.template_id,
        name: expert.name,
        avatarUrl: expert.avatar_url ?? null,
      }),
    );
}
