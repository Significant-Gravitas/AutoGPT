import { useGetHomeDashboard } from "@/app/api/__generated__/endpoints/home/home";
import { okData } from "@/app/api/helpers";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

// Presence is a nice-to-have, so lean on a stale window and share the home
// dashboard cache instead of spinning up another poll loop for the sidebar.
const TEAM_PRESENCE_STALE_TIME_MS = 60_000;

export function useSidebarTeamMembers() {
  const isEnabled = useGetFlag(Flag.HIRE_EXPERTS);

  const query = useGetHomeDashboard({
    query: {
      enabled: isEnabled,
      staleTime: TEAM_PRESENCE_STALE_TIME_MS,
      select: (response) => okData(response) ?? null,
    },
  });

  return {
    isEnabled,
    members: query.data?.agents ?? [],
  };
}
