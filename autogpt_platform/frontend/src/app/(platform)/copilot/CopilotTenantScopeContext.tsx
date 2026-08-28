"use client";

import { createContext, useContext, type ReactNode } from "react";

interface CopilotTenantScope {
  organizationId: string | null;
  teamId: string | null;
}

const CopilotTenantScopeContext = createContext<CopilotTenantScope>({
  organizationId: null,
  teamId: null,
});

interface Props extends CopilotTenantScope {
  children: ReactNode;
}

export function CopilotTenantScopeProvider({
  organizationId,
  teamId,
  children,
}: Props) {
  return (
    <CopilotTenantScopeContext.Provider value={{ organizationId, teamId }}>
      {children}
    </CopilotTenantScopeContext.Provider>
  );
}

export function useCopilotTenantScope() {
  return useContext(CopilotTenantScopeContext);
}
