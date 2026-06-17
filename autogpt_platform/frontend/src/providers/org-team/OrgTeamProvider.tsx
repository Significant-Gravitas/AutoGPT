"use client";

import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
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
 * 2. If no activeOrgID is stored, sets the personal org as default
 * 3. Fetches teams for the active org
 *
 * On org/team switch: clears React Query cache to force refetch.
 */
export default function OrgTeamProvider({ children }: Props) {
  const { isLoggedIn, user, isUserLoading } = useSupabase();
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
      clearContext();
      return;
    }

    async function loadOrgs() {
      try {
        const res = await fetch("/api/proxy/api/orgs", {
          headers: { "Content-Type": "application/json" },
        });
        if (!res.ok) {
          setLoaded(true);
          return;
        }
        const data = await res.json();
        const orgs = data.data || data;
        setOrgs(orgs);

        // If no active org, set the personal org as default
        if (!activeOrgID && orgs.length > 0) {
          const personal = orgs.find(
            (o: { isPersonal: boolean }) => o.isPersonal,
          );
          if (personal) {
            setActiveOrg(personal.id);
          } else {
            setActiveOrg(orgs[0].id);
          }
        }

        setLoaded(true);
      } catch {
        setLoaded(true);
      }
    }

    loadOrgs();
  }, [isLoggedIn, user, isUserLoading]);

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
