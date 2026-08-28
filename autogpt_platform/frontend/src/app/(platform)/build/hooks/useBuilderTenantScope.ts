"use client";

import { parseAsString, useQueryStates } from "nuqs";
import {
  decodeBuilderTenantScope,
  type BuilderTenantScope,
} from "@/services/org-team/builder";
import { useOrgTeamStore } from "@/services/org-team/store";

export function useBuilderTenantScope(): BuilderTenantScope & {
  isReady: boolean;
} {
  const [{ organizationId, teamId }] = useQueryStates({
    organizationId: parseAsString,
    teamId: parseAsString,
  });
  const explicitScope = decodeBuilderTenantScope(organizationId, teamId);
  const activeOrgID = useOrgTeamStore((s) => s.activeOrgID);
  const activeTeamID = useOrgTeamStore((s) => s.activeTeamID);
  const isLoaded = useOrgTeamStore((s) => s.isLoaded);

  return explicitScope
    ? { ...explicitScope, isReady: true }
    : {
        organizationId: activeOrgID,
        teamId: activeTeamID,
        isReady: isLoaded,
      };
}
