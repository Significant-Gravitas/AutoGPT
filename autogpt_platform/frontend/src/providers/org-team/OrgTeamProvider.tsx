"use client";

import { getV2ListUserOrganizations } from "@/app/api/__generated__/endpoints/orgs/orgs";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import {
  selectIsSessionUnknown,
  useAuthStore,
} from "@/lib/auth/hooks/useAuthStore";
import { useOrgTeamStore } from "@/services/org-team/store";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useEffect, useRef } from "react";

interface Props {
  children: React.ReactNode;
}

/**
 * Initializes org/team context on login and clears it on logout.
 *
 * On mount (when logged in):
 * 1. Fetches the user's org list from GET /api/orgs
 * 2. If no activeOrgID is stored, or the stored one isn't among them, sets
 *    the personal org as default
 * 3. Fetches teams for the active org
 *
 * On org/team switch: clears React Query cache to force refetch.
 */
export default function OrgTeamProvider({ children }: Props) {
  const { isLoggedIn, user, isUserLoading } = useAuth();
  const isSessionUnknown = useAuthStore(selectIsSessionUnknown);
  const { activeOrgID, setActiveOrg, setOrgs, setLoaded, clearContext } =
    useOrgTeamStore();

  const prevOrgID = useRef(activeOrgID);

  // Fetch orgs when logged in
  useEffect(() => {
    // While the session is still hydrating, isLoggedIn is transiently
    // false — clearing context here would flip activeOrgID to null and
    // (via the effect below) wipe the query cache mid-flight, stranding
    // every in-flight page query in a forever-pending state.
    if (isUserLoading) {
      return;
    }

    if (!isLoggedIn || !user) {
      // Only a session known to be gone drops the stored org and team. One
      // nobody could check keeps them for when a retry finds the user.
      if (!isSessionUnknown) clearContext();
      return;
    }

    async function loadOrgs() {
      try {
        const response = await getV2ListUserOrganizations();
        if (response.status !== 200) {
          setLoaded(true);
          return;
        }
        const orgs = response.data;
        setOrgs(orgs);

        // No active org, or one this user isn't in (left by an earlier session
        // on this browser): fall back to the personal org, else the first,
        // which clears the team. Only a list fetched successfully says so, and
        // an empty one has no org to fall back to.
        const fallbackID = (orgs.find((o) => o.is_personal) ?? orgs.at(0))?.id;
        if (fallbackID && !orgs.some((o) => o.id === activeOrgID)) {
          setActiveOrg(fallbackID);
        }

        setLoaded(true);
      } catch {
        setLoaded(true);
      }
    }

    loadOrgs();
  }, [isLoggedIn, user, isUserLoading, isSessionUnknown]);

  // Drop org-scoped data when the org switches. resetQueries (NOT
  // clear) — clear() removes queries without notifying mounted
  // observers, which leaves them pending forever; resetQueries
  // refetches everything that's still on screen.
  useEffect(() => {
    if (prevOrgID.current !== activeOrgID && prevOrgID.current !== null) {
      const queryClient = getQueryClient();
      queryClient.resetQueries();
    }
    prevOrgID.current = activeOrgID;
  }, [activeOrgID]);

  return <>{children}</>;
}
