"use client";

import { useCanCreateOrganization } from "@/services/org-team/useCanCreateOrganization";

// Accounts that only inherited organizations from someone else's team
// organization can open these settings but may not create or convert one.
export function usePersonalOrgSettings() {
  return { canCreateOrganization: useCanCreateOrganization() };
}
