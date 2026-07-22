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
  const [teamId, setTeamIdState] = useState<string | null>(() =>
    getLastUsedTeam(surfaceKey),
  );

  // Clamp to a still-valid team once the store loads (e.g. last-used team was
  // deleted): fall back to org-home rather than sending a stale header.
  useEffect(() => {
    if (teamId && teams.length > 0 && !teams.some((t) => t.id === teamId)) {
      setTeamIdState(null);
    }
  }, [teams, teamId]);

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
