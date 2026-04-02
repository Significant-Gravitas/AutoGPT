"use client";

import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useOrgWorkspaceStore } from "@/services/org-workspace/store";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useEffect, useRef } from "react";

interface Props {
  children: React.ReactNode;
}

/**
 * Initializes org/workspace context on login and clears it on logout.
 *
 * On mount (when logged in):
 * 1. Fetches the user's org list from GET /api/orgs
 * 2. If no activeOrgID is stored, sets the personal org as default
 * 3. Fetches workspaces for the active org
 *
 * On org/workspace switch: clears React Query cache to force refetch.
 */
export default function OrgWorkspaceProvider({ children }: Props) {
  const { isLoggedIn, user } = useSupabase();
  const { activeOrgID, setActiveOrg, setOrgs, setLoaded, clearContext } =
    useOrgWorkspaceStore();

  const prevOrgID = useRef(activeOrgID);

  // Fetch orgs when logged in
  useEffect(() => {
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
  }, [isLoggedIn, user]);

  // Clear React Query cache when org switches
  useEffect(() => {
    if (prevOrgID.current !== activeOrgID && prevOrgID.current !== null) {
      const queryClient = getQueryClient();
      queryClient.clear();
    }
    prevOrgID.current = activeOrgID;
  }, [activeOrgID]);

  return <>{children}</>;
}
