import { useOrgTeamStore } from "@/services/org-team/store";
import { useEffect, useState } from "react";
import {
  getLastUsedTeam,
  getTeamRequestInit,
  setLastUsedTeam,
} from "./helpers";

// State hook for a create surface's team ownership. Seeds from the surface's
// last-used team (org-home if never used), persists on change, and exposes the
// Orval request options that stamp X-Team-Id on the create call.
export function useCreateTeamSelection(surfaceKey: string) {
  const teams = useOrgTeamStore((s) => s.teams);
  const isLoaded = useOrgTeamStore((s) => s.isLoaded);
  const [teamId, setTeamIdState] = useState<string | null>(() =>
    getLastUsedTeam(surfaceKey),
  );

  // Once the store has loaded, clamp a last-used team that no longer exists
  // (deleted, or the user left the org / now has no teams at all) back to
  // org-home. Guarding on isLoaded avoids clearing during the initial load;
  // dropping the teams.length check covers the solo case (empty team list).
  // Also clear the persisted value so a remount can't briefly resurrect the
  // stale id and stamp a bogus X-Team-Id on the create.
  useEffect(() => {
    if (isLoaded && teamId && !teams.some((t) => t.id === teamId)) {
      setTeamIdState(null);
      setLastUsedTeam(surfaceKey, null);
    }
  }, [isLoaded, teams, teamId, surfaceKey]);

  function setTeamId(next: string | null) {
    setTeamIdState(next);
    setLastUsedTeam(surfaceKey, next);
  }

  return {
    teamId,
    setTeamId,
    hasTeams: teams.length > 0,
    teamRequestInit: getTeamRequestInit(teamId),
  };
}
