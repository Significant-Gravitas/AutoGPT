"use client";

import { useMemo } from "react";
import { useSupabaseStore } from "@/lib/supabase/hooks/useSupabaseStore";
import {
  useGetV2GetTheAgentWaitlist,
  useGetV2GetWaitlistIdsTheCurrentUserHasJoined,
  getGetV2GetWaitlistIdsTheCurrentUserHasJoinedQueryKey,
} from "@/app/api/__generated__/endpoints/store/store";
import type { StoreWaitlistEntry } from "@/app/api/__generated__/models/storeWaitlistEntry";
import { useQueryClient } from "@tanstack/react-query";

export function useWaitlistSection() {
  const { user } = useSupabaseStore();
  const queryClient = useQueryClient();

  // Fetch waitlists
  const {
    data: waitlistsResponse,
    isLoading: waitlistsLoading,
    isError: waitlistsError,
  } = useGetV2GetTheAgentWaitlist();

  // Fetch memberships if logged in
  const { data: membershipsResponse, isLoading: membershipsLoading } =
    useGetV2GetWaitlistIdsTheCurrentUserHasJoined({
      query: {
        enabled: !!user,
      },
    });

  const waitlists: StoreWaitlistEntry[] = useMemo(() => {
    if (waitlistsResponse?.status === 200) {
      return waitlistsResponse.data.listings;
    }
    return [];
  }, [waitlistsResponse]);

  const joinedWaitlistIds: Set<string> = useMemo(() => {
    if (membershipsResponse?.status === 200) {
      return new Set(membershipsResponse.data);
    }
    return new Set();
  }, [membershipsResponse]);

  const isLoading = waitlistsLoading || (!!user && membershipsLoading);
  const hasError = waitlistsError;

  // Function to add a waitlist ID to joined set (called after successful join)
  function markAsJoined(_waitlistId: string) {
    // Invalidate the memberships query to refetch
    queryClient.invalidateQueries({
      queryKey: getGetV2GetWaitlistIdsTheCurrentUserHasJoinedQueryKey(),
    });
  }

  return { waitlists, joinedWaitlistIds, isLoading, hasError, markAsJoined };
}
