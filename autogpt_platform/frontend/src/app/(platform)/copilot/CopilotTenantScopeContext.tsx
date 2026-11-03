"use client";

import { createContext, useContext, type ReactNode } from "react";
import { useOrgTeamStore } from "@/services/org-team/store";

interface CopilotTenantScope {
  organizationId: string | null;
  teamId: string | null;
}

const CopilotTenantScopeContext = createContext<CopilotTenantScope | undefined>(
  undefined,
);

// One object per scope, so the context value keeps its identity across the
// chat's renders and a streamed flush does not re-render every consumer
// (the composer among them) behind its memo boundary.
const scopes = new Map<string, CopilotTenantScope>();

function getScope(organizationId: string | null, teamId: string | null) {
  const key = JSON.stringify([organizationId, teamId]);
  let scope = scopes.get(key);
  if (!scope) {
    scope = { organizationId, teamId };
    scopes.set(key, scope);
  }
  return scope;
}

interface Props extends CopilotTenantScope {
  children: ReactNode;
}

export function CopilotTenantScopeProvider({
  organizationId,
  teamId,
  children,
}: Props) {
  return (
    <CopilotTenantScopeContext.Provider
      value={getScope(organizationId, teamId)}
    >
      {children}
    </CopilotTenantScopeContext.Provider>
  );
}

export function useCopilotTenantScope() {
  const scope = useContext(CopilotTenantScopeContext);
  const organizationId = useOrgTeamStore((state) => state.activeOrgID);
  const teamId = useOrgTeamStore((state) => state.activeTeamID);
  return scope ?? getScope(organizationId, teamId);
}
