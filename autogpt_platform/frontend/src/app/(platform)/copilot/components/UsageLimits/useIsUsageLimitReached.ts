import { useGetV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { useUsageExperience } from "@/services/usageExperience/useUsageExperience";

export function useIsUsageLimitReached(sessionID?: string | null) {
  const state = useUsageExperience();
  const session = useGetV2GetSession(sessionID ?? "", undefined, {
    query: { enabled: !!sessionID },
  });
  const provider =
    session.data?.status === 200
      ? session.data.data.metadata?.llm_auth_provider
      : null;
  if (provider && provider !== "platform") return false;
  return state.experience.blocked || state.isError;
}
