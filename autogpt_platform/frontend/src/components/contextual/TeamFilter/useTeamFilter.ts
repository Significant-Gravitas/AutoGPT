import { useOrgTeamStore } from "@/services/org-team/store";
import { useEffect, useState } from "react";
import {
  matchesTeamFilter,
  TEAM_FILTER_ALL,
  TEAM_FILTER_ORG_HOME,
} from "./helpers";

// State + predicate for a team filter on a list. Defaults to "All"; exposes a
// `matches` predicate lists apply to each row's team id.
export function useTeamFilter() {
  const teams = useOrgTeamStore((s) => s.teams);
  const isLoaded = useOrgTeamStore((s) => s.isLoaded);
  const [value, setValue] = useState<string>(TEAM_FILTER_ALL);

  // Once the store loads, reset a filter that can no longer apply. When the
  // user has no teams the control is hidden, so any non-"All" value would
  // silently hide rows with no way to clear it; and a team-specific value
  // whose team disappeared (org switch, team deleted) falls back to "All".
  useEffect(() => {
    if (!isLoaded) return;
    if (teams.length === 0) {
      if (value !== TEAM_FILTER_ALL) setValue(TEAM_FILTER_ALL);
      return;
    }
    if (
      value !== TEAM_FILTER_ALL &&
      value !== TEAM_FILTER_ORG_HOME &&
      !teams.some((t) => t.id === value)
    ) {
      setValue(TEAM_FILTER_ALL);
    }
  }, [isLoaded, teams, value]);

  return {
    value,
    setValue,
    hasTeams: teams.length > 0,
    matches: (rowTeamId: string | null | undefined) =>
      matchesTeamFilter(rowTeamId, value),
  };
}
