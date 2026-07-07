"use client";

import { useGetV2GetOrganizationDetails } from "@/app/api/__generated__/endpoints/orgs/orgs";
import { useGetV2ListOrganizationMembers } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgMemberResponse } from "@/app/api/__generated__/models/orgMemberResponse";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useOrgTeamStore } from "@/services/org-team/store";

export function useOrganizationSettingsPage() {
  const { user } = useSupabase();
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

  return {
    org: orgQuery.data ?? null,
    members,
    currentMember,
    isAdmin,
    isLoading:
      !isOrgContextLoaded || orgQuery.isLoading || membersQuery.isLoading,
    isError: orgQuery.isError,
    error: orgQuery.error,
    refetchMembers: membersQuery.refetch,
    refetchOrg: orgQuery.refetch,
  };
}
