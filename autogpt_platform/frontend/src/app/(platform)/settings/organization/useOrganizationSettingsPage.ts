"use client";

import { useGetV2GetOrganizationDetails } from "@/app/api/__generated__/endpoints/orgs/orgs";
import { useGetV2ListOrganizationMembers } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgMemberResponse } from "@/app/api/__generated__/models/orgMemberResponse";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useOrgTeamStore } from "@/services/org-team/store";

export function useOrganizationSettingsPage() {
  const { user } = useAuth();
  const { activeOrgID, isLoaded: isOrgContextLoaded } = useOrgTeamStore();

  const orgQuery = useGetV2GetOrganizationDetails(activeOrgID ?? "", {
    query: {
      enabled: Boolean(activeOrgID),
      select: (res) => res.data as OrgResponse,
    },
  });

  const membersQuery = useGetV2ListOrganizationMembers(activeOrgID ?? "", {
    query: {
      enabled: Boolean(activeOrgID),
      select: (res) => res.data as OrgMemberResponse[],
    },
  });

  const members = membersQuery.data ?? [];
  const currentMember = members.find((m) => m.user_id === user?.id) ?? null;
  const isAdmin = Boolean(currentMember?.is_owner || currentMember?.is_admin);
  // MANAGE_BILLING is owner + billing_manager only — admins do NOT get it
  // (matches _ORG_PERMISSIONS in autogpt_libs/auth/permissions.py).
  const canManageBilling = Boolean(
    currentMember?.is_owner || currentMember?.is_billing_manager,
  );

  return {
    org: orgQuery.data ?? null,
    members,
    currentMember,
    isAdmin,
    canManageBilling,
    isLoading:
      !isOrgContextLoaded || orgQuery.isLoading || membersQuery.isLoading,
    isError: orgQuery.isError,
    error: orgQuery.error,
    refetchMembers: membersQuery.refetch,
    refetchOrg: orgQuery.refetch,
  };
}
