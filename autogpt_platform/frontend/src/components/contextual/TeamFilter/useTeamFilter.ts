import { useOrgTeamStore } from "@/services/org-team/store";
import { useState } from "react";
import { matchesTeamFilter, TEAM_FILTER_ALL } from "./helpers";

// State + predicate for a team filter on a list. Defaults to "All"; exposes a
// `matches` predicate lists apply to each row's team id.
export function useTeamFilter() {
  const teams = useOrgTeamStore((s) => s.teams);
  const [value, setValue] = useState<string>(TEAM_FILTER_ALL);

  return {
    value,
    setValue,
    hasTeams: teams.length > 0,
    matches: (rowTeamId: string | null | undefined) =>
      matchesTeamFilter(rowTeamId, value),
  };
}
